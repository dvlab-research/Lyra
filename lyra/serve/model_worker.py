"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import os

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from lyra.constants import WORKER_HEART_BEAT_INTERVAL
from lyra.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from lyra.model.builder import load_pretrained_model
from lyra.mm_utils import process_highres_image, process_anyres_image, load_image_from_base64, tokenizer_image_speech_token, tokenizer_speech_token
from lyra.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_SPEECH_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread

from torchaudio.transforms import Resample
from transformers import WhisperFeatureExtractor
import torchaudio

from qwen_vl_utils import process_vision_info
from lyra.conversation import SeparatorStyle
from moviepy import VideoFileClip

import io
import base64
from PIL import Image
import numpy as np

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None

def build_unit_tokenizer(vocab_size):
    import os
    from transformers import BertTokenizer
    with open("unit_vocab.txt", "w") as f:
        for i in range(vocab_size + 1):
            f.write(str(i) + "\n")
    tokenizer = BertTokenizer(vocab_file="unit_vocab.txt")
    os.remove("unit_vocab.txt")
    return tokenizer


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_lora_path, model_base, model_name,
                 load_8bit, device, use_flash_attn=False):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len, self.speech_processor = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, model_lora_path=model_lora_path, device=self.device, use_flash_attn=True)
        self.unit_tokenizer = build_unit_tokenizer(self.model.config.unit_vocab_size)
        self.is_multimodal = True

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=30)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
    def add_content(self, prompt, new_content):
        if '[INST]' in prompt:
            split_index = prompt.rfind(' [/INST]')
        elif '<|im_end|>' in prompt:
            split_index = prompt.rfind('<|im_end|>')
        else:
            split_index = prompt.rfind('###Assistant:')
        left_prompt = prompt[:split_index]
        right_prompt = prompt[split_index:]
        prompt = left_prompt + new_content + right_prompt
        return prompt
    
    def preprocess_image_qwen2vl(self, image, image_resolution):
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

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor, speech_processor = self.tokenizer, self.model, self.image_processor, self.speech_processor
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        speeches = params.get("speeches", None)
        videos = params.get("videos", None)

        # prepare image file
        if images[-1] is not None:
            image_file = images[-1]
            image = Image.open(image_file).convert('RGB')
            if model.config.image_aspect_ratio == "qwen2vl":
                image_tensor = self.preprocess_image_qwen2vl(image, image_resolution=2880)
                image_tensor = image_processor(images=image_tensor, return_tensors="pt")
            else:
                raise NotImplementedError
        # prepare video file
        elif videos[-1] is not None:
            new_frames = dict()
            video_file = videos[-1]
            if model.config.image_aspect_ratio == "qwen2vl":
                total_pixels = 28 * 28 * 2048 * 32
                print('Visual tokens:', int(total_pixels / 28 / 28))
                min_pixels = 28 * 28 * 4
                max_frames = 16
                messages = [{
                    "role": "user",
                    "content": [{
                        "type": "video",
                        "video": video_file,
                        "total_pixels": total_pixels,
                        "min_pixels": min_pixels,
                        "max_frames": max_frames
                    }]
                }]
                image_inputs, video_inputs = process_vision_info(messages)
                image_tensor = image_processor(images=None, videos=video_inputs[0], return_tensors="pt")
            else:
                raise NotImplementedError
            
            # extract mp3 file from video
            if speeches[-1] is None:
                try:
                    video = VideoFileClip(video_file)
                    audio = video.audio
                    audio.write_audiofile('/tmp/gradio/tmp.mp3')
                    speeches = ['/tmp/gradio/tmp.mp3']
                    print(f"save audio at: /tmp/gradio/tmp.mp3")
                except:
                    pass
        else:
            image_tensor = None
        
        # prepare speech file
        if speeches[-1] is not None:
            audio_file = speeches[-1]
            target_sample_rate = 16000
            wav, sample_rate = torchaudio.load(audio_file)
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
        
        # convert to cuda
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

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        prompt = prompt.replace('<image>\n', '<|vision_start|><image><|vision_end|>')
        print("model prompt: ", prompt)
        if '<image>' in prompt:
            input_ids = tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)
        else:
            input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30)
        streamer_unit = TextIteratorStreamer(self.unit_tokenizer, skip_prompt=False, skip_special_tokens=True, timeout=15)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return
        
        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            streamer_unit=streamer_unit,
            use_cache=True,
            images=images,
            speeches=speeches,
            bos_token_id=151643,  # Begin of sequence token
            eos_token_id=[151645,151643],  # End of sequence token
            pad_token_id=151643,  # Pad token
            streaming_unit_gen=True,
        ))
            
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            generated_unit = " ".join(map(str, streamer_unit.token_cache))
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "unit": generated_unit, "error_code": 0}).encode() + b"\0"
        torch.cuda.empty_cache()

        if os.path.exists('/tmp/gradio/tmp.mp3'):
            os.remove('/tmp/gradio/tmp.mp3') 


    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="work_dirs/Lyra_Base_9B")
    parser.add_argument("--model-lora-path", type=str, default="work_dirs/Lyra_Base_9B/speech_lora")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_lora_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.device,
                         use_flash_attn=args.use_flash_attn)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
