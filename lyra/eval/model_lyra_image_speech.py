import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
from typing import Dict, Optional, Sequence, List
import re

from lyra.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_SPEECH_TOKEN
from lyra.conversation import conv_templates, SeparatorStyle
from lyra.model.builder import load_pretrained_model
from lyra.utils import disable_torch_init
from lyra.mm_utils import tokenizer_image_token, process_highres_image, process_anyres_image, process_highres_image_crop_split, get_model_name_from_path, load_image_from_base64
from lyra.mm_utils import tokenizer_image_speech_token

from transformers import WhisperFeatureExtractor
from torchaudio.transforms import Resample
import torchaudio

from PIL import Image
import math
import pdb


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len, speech_processor = load_pretrained_model(model_path, args.model_base, model_name, use_flash_attn=args.use_flash_attn, model_lora_path=args.model_lora_path, eval_bench=True)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        if 'speech' in line:
            speech_file = "{}.mp3".format(line['speech'])
        else:
            speech_file = "{}.mp3".format(idx)

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + DEFAULT_SPEECH_TOKEN
        else:
            if 'textvqa' in args.image_folder.lower():
                qs = DEFAULT_IMAGE_TOKEN + '\n' + line["text"]
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + DEFAULT_SPEECH_TOKEN

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        prompt = prompt.replace('<image>\n', '<|vision_start|><image><|vision_end|>')
        input_ids = tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()


        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        if model.config.image_aspect_ratio == "qwen2vl":
            def _preprocess_image_qwen2vl(image, **kwargs):
                image_resolution = 1920
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
            
            image = _preprocess_image_qwen2vl(image, image_resolution=2880)
            image = image_processor(images=image, return_tensors="pt")
        elif model.config.image_aspect_ratio == "highres":
            image = process_highres_image(image, image_processor, model.config.image_grid_pinpoints)
        elif model.config.image_aspect_ratio == "anyres" or "anyres_max" in model.config.image_aspect_ratio:
            image = process_anyres_image(image, image_processor, model.config.image_grid_pinpoints)
        
        image_tensor = image

        images = image_tensor.to(dtype=model.dtype, device='cuda', non_blocking=True)
        
        speech_file = os.path.join(args.speech_folder, speech_file)
        if not os.path.exists(speech_file):
            speech_file = speech_file.replace(".mp3", ".wav")
        target_sample_rate = 16000
        wav, sample_rate = torchaudio.load(speech_file)
        resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        wav = resample_transform(wav)
        if wav.ndim != 1: # convert to mono
            wav = wav[0]
        
        
        speech_tensor = speech_processor(raw_speech=wav, 
                            sampling_rate=target_sample_rate, 
                            return_tensors="pt", 
                            return_attention_mask=True)["input_features"] # (1, 128,3000)

        if "qwen2vl" == args.conv_mode:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[images],
                    speeches=[speech_tensor.to(dtype=model.dtype, device=model.device, non_blocking=True)],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    bos_token_id=151643,  # Begin of sequence token
                    eos_token_id=[151645,151643],  # End of sequence token
                    pad_token_id=151643,  # Pad token
                    use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        print(outputs)
        prompt = line["prompt"] if 'textvqa' in args.question_file else line["text"]
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-lora-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--speech-folder", type=str, default="")
    parser.add_argument("--speech-processor-path", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--use_flash_attn', type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)