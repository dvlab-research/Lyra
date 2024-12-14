import argparse
import torch

from lyra.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_SPEECH_TOKEN
from lyra.conversation import conv_templates, SeparatorStyle
from lyra.model.builder import load_pretrained_model
from lyra.utils import disable_torch_init
from lyra.mm_utils import process_highres_image, process_anyres_image, tokenizer_image_speech_token, tokenizer_speech_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from qwen_vl_utils import process_vision_info, fetch_video
from qwen_vl_utils.vision_process import _read_video_decord
from torchaudio.transforms import Resample
import torchaudio

import requests
from PIL import Image, ImageDraw
from io import BytesIO
from transformers import TextStreamer
from transformers import WhisperFeatureExtractor

from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import soundfile as sf
import os
import json


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def preprocess_image_qwen2vl(image, image_resolution):
    if max(image.width, image.height) > image_resolution:
        resize_factor = image_resolution / max(image.width, image.height)
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")
        
    if min(image.width, image.height) < 28:
        width, height = max(image.width, 28), max(image.height, 28)
        image = image.resize((width, height), resample=Image.NEAREST)

    if image.width / image.height > 200:
        width, height = image.height * 180, image.height
        image = image.resize((width, height), resample=Image.NEAREST)

    if image.height / image.width > 200:
        width, height = image.width, image.width * 180
        image = image.resize((width, height), resample=Image.NEAREST)

    return image

def dump_speech(args, pred_wav, round_num):
    if not os.path.exists(args.out_speech_path):
        os.makedirs(args.out_speech_path)
    sf.write(f"{args.out_speech_path}/pred_round{round_num}.wav", pred_wav.detach().cpu().numpy(), 16000, )
    
def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if args.extractor:
        if 'extractor' not in model_name.lower():
            model_name = model_name + '_extractor'

    tokenizer, model, image_processor, _, speech_processor = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, model_lora_path=args.lora_path, use_flash_attn=True, device=args.device)
    
    model_lora_path = f'{args.model_path}/speech_lora'
    model.load_adapter(model_lora_path, adapter_name="speech")
    print(f"Loading LoRA weights from {model_lora_path}")
    model.to(torch.float16)
    
    model_lora_path = f'{args.model_path}/long_speech_lora'
    model.load_adapter(model_lora_path, adapter_name="long_speech")
    print(f"Loading LoRA weights from {model_lora_path}")
    model.to(torch.float16)
    
    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(args.vocoder_ckpt, vocoder_cfg).cuda()

    conv_mode = 'qwen2vl'
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    # prepare visual file
    if args.image_file is not None:

        images = []
        if ',' in args.image_file:
            images = args.image_file.split(',')
        else:
            images = [args.image_file]
        
        image_convert = []
        for _image in images:
            image_convert.append(load_image(_image))
        if len(image_convert) == 1:
            image_convert = image_convert[0]
            
        if model.config.image_aspect_ratio == "highres":
            image_tensor = process_highres_image(image_convert, image_processor, model.config.image_grid_pinpoints)
        elif model.config.image_aspect_ratio == "anyres" or "anyres_max" in model.config.image_aspect_ratio:
            image_tensor = process_anyres_image(image_convert, image_processor, model.config.image_grid_pinpoints)
        elif model.config.image_aspect_ratio == "qwen2vl":
            image_tensor = preprocess_image_qwen2vl(image_convert, image_resolution=2880)
            image_tensor = image_processor(images=image_tensor, return_tensors="pt")
        else:
            raise NotImplementedError
    
    elif args.video_file is not None:
        
        if model.config.image_aspect_ratio == "qwen2vl":
            total_pixels = 28 * 28 * 2048 * 32
            print('Visual tokens:', int(total_pixels / 28 / 28))
            min_pixels = 28 * 28 * 4
            max_frames = 16
            messages = [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": args.video_file,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    "max_frames": max_frames
                }]
            }]
            image_inputs, video_inputs = process_vision_info(messages)
            image_tensor = image_processor(images=None, videos=video_inputs[0], return_tensors="pt")
        else:
            raise NotImplementedError
        
    else:
        image_tensor = None


    # prepare speech file
    if args.audio_file is not None:
        target_sample_rate = 16000
        wav, sample_rate = torchaudio.load(args.audio_file)
        resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        wav = resample_transform(wav)
        if wav.ndim != 1: # convert to mono
            wav = wav[0]
        
        speech_tensor = []
        whipser_len = target_sample_rate * 30
        speech_num = wav.shape[0] // whipser_len + 1
        for i in range(speech_num):
            temp_wav = wav[i*whipser_len:(i+1)*whipser_len]
            _speech_tensor = speech_processor(raw_speech=temp_wav, 
                                sampling_rate=target_sample_rate, 
                                return_tensors="pt", 
                                return_attention_mask=True)["input_features"].squeeze() # (128, 3000)
            speech_tensor.append(_speech_tensor)

        speech_tensor = torch.stack(speech_tensor, dim=0).squeeze()

    else:
        speech_tensor = None


    if image_tensor is not None:
        if isinstance(image_tensor, dict):
            for key in image_tensor.keys():
                image_tensor[key] = image_tensor[key].to(dtype=model.dtype, device=model.device, non_blocking=True)
        else:
            image_tensor = image_tensor.to(dtype=model.dtype, device=model.device, non_blocking=True)
        image_tensor = [image_tensor]
        images = image_tensor
    else:
        images = None
    
    if speech_tensor is not None:
        speech_tensor = [speech_tensor.to(dtype=model.dtype, device=model.device, non_blocking=True)]
        speeches = speech_tensor
    else:
        speeches = None
    
    round_num = 0
    while True:

        if speeches is not None and images is not None:
            inp = DEFAULT_SPEECH_TOKEN
        else:
            try:
                inp = input(f"{roles[0]}: ")
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break

        print(f"{roles[1]}: ", end="")

        if images is not None:
            # first message
            image_tokens = DEFAULT_IMAGE_TOKEN
            if model.config.image_aspect_ratio == "qwen2vl":
                image_tokens = '<|vision_start|>' + image_tokens + '<|vision_end|>'
                inp = image_tokens + inp
            else:
                inp = image_tokens + "\n" + inp
        elif speeches is not None:
            speech_tokens = DEFAULT_SPEECH_TOKEN
            inp = speech_tokens + "\n" + inp
        else:
            pass
            # later messages
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if images is not None:
            input_ids = tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        else:
            input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        if speeches is None:
            model.disable_adapters()
            print("No load new projectors & LoRA module!")
        elif len(speeches[0].shape) == 2:
            model.set_adapter(["speech"])
            model_lora_path = f'{args.model_path}/speech_lora'
            mm_projector_weights = torch.load(os.path.join(model_lora_path, 'non_lora_trainables.bin'), map_location='cpu')
            mm_projector_weights = {k[6:]: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            print("load new projectors....", mm_projector_weights.keys())
            status = model.load_state_dict(mm_projector_weights, strict=False)
            print('load pretrain_mm_mlp_adapter, unexpected_keys:{}'.format(status.unexpected_keys))
        elif len(speeches[0].shape) == 3:
            model.set_adapter(["long_speech"])
            model_lora_path = f'{args.model_path}/long_speech_lora'
            mm_projector_weights = torch.load(os.path.join(model_lora_path, 'non_lora_trainables.bin'), map_location='cpu')
            mm_projector_weights = {k[6:]: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            print("load new projectors....", mm_projector_weights.keys())
            status = model.load_state_dict(mm_projector_weights, strict=False)
            print('load pretrain_mm_mlp_adapter, unexpected_keys:{}'.format(status.unexpected_keys))
        
        with torch.inference_mode():
            if args.generate_speech: # include speech generation
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    speeches=speech_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    bos_token_id=151643,  # Begin of sequence token
                    eos_token_id=[151645,151643],  # End of sequence token
                    pad_token_id=151643,  # Pad token
                    streamer=streamer,
                    use_cache=True)
                output_ids, output_units = outputs
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                conv.messages[-1][-1] = output_text
                output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
                output_units = list(map(int, output_units.strip().split()))
                x = {"code": torch.LongTensor(output_units).view(1, -1).cuda()}
                wav = vocoder(x, True)
                dump_speech(args, wav, round_num)
            else:
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    speeches=speech_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    bos_token_id=151643,  # Begin of sequence token
                    eos_token_id=[151645,151643],  # End of sequence token
                    pad_token_id=151643,  # Pad token
                    streamer=streamer,
                    use_cache=True)
                output_ids, _ = outputs
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                conv.messages[-1][-1] = outputs

        images = None
        speeches = None
        round_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="work_dirs/Lyra_Base_9B")
    parser.add_argument("--vocoder-ckpt", type=str, default="model_zoo/audio/vocoder/g_00500000")
    parser.add_argument("--vocoder-cfg", type=str, default="model_zoo/audio/vocoder/config.json")
    parser.add_argument("--out-speech-path", type=str, default="examples")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--audio-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="qwen2vl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--extractor", action="store_true")
    parser.add_argument("--generate-speech", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)