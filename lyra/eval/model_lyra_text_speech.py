import argparse
import torch
import os
import json
from tqdm import tqdm

from jiwer import wer
import jiwer
import json
import re
import math

from lyra.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from lyra.conversation import conv_templates
from lyra.model.builder import load_pretrained_model
from lyra.utils import disable_torch_init
from lyra.mm_utils import tokenizer_speech_token, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor
from torchaudio.transforms import Resample
import torchaudio

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# for evaluate speech captions
def save_result(result, result_file, remove_duplicate='speech'):
    if remove_duplicate:
        result_new = []
        id_list = []    
        for res in result:
            if res[remove_duplicate] not in id_list:
                id_list.append(res[remove_duplicate])
                result_new.append(res)
        result = result_new             
            
    json.dump(result,open(result_file,'w'),ensure_ascii=False, indent=4)            
    print('result file saved to %s'%result_file)
    return result_file


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, speech_test_file, speech_processor, tokenizer, num_chunks, chunk_idx, speech_folder, conv_mode):
        self.speech_test_file = speech_test_file
        self.tokenizer = tokenizer
        self.speech_processor = speech_processor
        self.target_sample_rate = 16000
        self.speech_folder = speech_folder
        self.conv_mode = conv_mode

        self.speech_prompt = "Record the spoken words as text."
        
        test_data = []
        with open(speech_test_file, 'r') as f:
            for i, line in enumerate(f):
                item = json.loads(line.strip()) # it contains {'audio': speech_path, 'text': gt_text}
                test_data.append(item)
                
        self.test_data = get_chunk(test_data, num_chunks, chunk_idx)

    def __getitem__(self, index):
        if "speech" in self.test_data[index]:
            speech_file = self.test_data[index]["speech"]
        else:
            speech_file = self.test_data[index]["audio"]
        wav, sample_rate = torchaudio.load(os.path.join(self.speech_folder, speech_file))
        resample_transform = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
        wav = resample_transform(wav)
        if wav.ndim != 1: # convert to mono
            wav = wav[0]
        
        speech_tensor = self.speech_processor(raw_speech=wav, 
                            sampling_rate=self.target_sample_rate, 
                            return_tensors="pt", 
                            return_attention_mask=True)["input_features"].squeeze() # (128,3000)

        qs = DEFAULT_SPEECH_TOKEN + '\n' + '\n' + self.speech_prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, SPEECH_TOKEN_INDEX, return_tensors='pt')
        gt_text = self.test_data[index]["text"]
        return speech_file, input_ids, speech_tensor, gt_text
    
    def __len__(self):
        return len(self.test_data)


# DataLoader
def create_data_loader(speech_test_file, speech_processor_path, tokenizer, speech_folder, conv_mode, batch_size=1, num_workers=4, num_chunks=4, chunk_idx=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(speech_test_file, speech_processor_path, tokenizer, num_chunks, chunk_idx, speech_folder, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len, speech_processor = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=args.load_8bit, model_lora_path=args.model_lora_path, eval_bench=True)
    model.config.train_modality = "text_speech"
    model.config.asr_align = False
    data_loader = create_data_loader(args.speech_test_file, speech_processor, tokenizer, 
                                     args.speech_folder, args.conv_mode,
                                     num_workers=args.num_workers, 
                                     num_chunks=args.num_chunks, chunk_idx=args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for speech_file, input_ids, speech_tensor, gt_text in tqdm(data_loader, total=len(data_loader)):
        input_ids = input_ids.to(device=model.device, non_blocking=True)
        if args.conv_mode == "qwen2vl":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    speeches=[speech_tensor.to(dtype=model.dtype, device=model.device, non_blocking=True)],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    bos_token_id=151643,  # Begin of sequence token
                    eos_token_id=[151645,151643],  # End of sequence token
                    pad_token_id=151643,  # Pad token
                    use_cache=True)
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_file.write(json.dumps({"speech": speech_file[0],
                                   "text": outputs,
                                   "gt_text": gt_text[0], 
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-lora-path", type=str, default=None)
    parser.add_argument("--speech-test-file", type=str, default="")
    parser.add_argument("--speech-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--load_8bit', type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
