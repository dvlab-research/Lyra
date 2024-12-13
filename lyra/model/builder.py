#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
#    Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
#    Copyright 2024 Zhisheng Zhong, Chengyao Wang
# ------------------------------------------------------------------------

import os
import warnings

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, GenerationMixin
import torch
from lyra.model import *
from lyra.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import pdb


def initialize_input_ids_for_generation(
    self,
    inputs: Optional[torch.Tensor] = None,
    bos_token_id: Optional[torch.Tensor] = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.LongTensor:
    """Initializes input ids for generation, if necessary."""
    if inputs is not None:
        return inputs

    encoder_outputs = model_kwargs.get("encoder_outputs")
    if self.config.is_encoder_decoder and encoder_outputs is not None:
        # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
        shape = encoder_outputs.last_hidden_state.size()[:-1]
        return torch.ones(shape, dtype=torch.long, device=self.device) * -100

    # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
    # soft-prompting or in multimodal implementations built on top of decoder-only language models.
    batch_size = 1

    if "inputs_embeds" in model_kwargs:
        return torch.ones((batch_size, 0), dtype=torch.long, device=self.device)

    for value in model_kwargs.values():
        if isinstance(value, torch.Tensor):
            batch_size = value.shape[0]
            break

    if bos_token_id is None:
        raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

    return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id


GenerationMixin._maybe_initialize_input_ids_for_generation = initialize_input_ids_for_generation


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, 
                          model_lora_path=None, merge_lora_path=None, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    if 'extractor' in model_name.lower():
        from lyra.model.qwen2vl_top_attn import replace_qwen2vl_attn_with_top_attn
        replace_qwen2vl_attn_with_top_attn()
    if 'lyra' in model_name.lower():        
        # Load MGM model
        if model_base is not None:
            raise NotImplementedError
        else:
            # Qwen2-VL based
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            if "extractor" in model_name.lower():
                model = LyraQwen2VLForCausalLMExtractor.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif "Base" in model_name or "Mini" in model_name or "Pro" in model_name:
                model = Lyra2SQwen2VLForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                model = LyraQwen2VLForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            # todo: Qwen2 & LLaMA3

                
            if model_lora_path is not None: # add lora projector
                mm_projector_weights = torch.load(os.path.join(model_lora_path, 'non_lora_trainables.bin'), map_location='cpu')
                mm_projector_weights = {k[6:]: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                print("load new projectors....", mm_projector_weights.keys())
                status = model.load_state_dict(mm_projector_weights, strict=False)
                print('load pretrain_mm_mlp_adapter, unexpected_keys:{}'.format(status.unexpected_keys))
            
            if merge_lora_path is not None: # add lora projector
                mm_projector_weights = torch.load(os.path.join(merge_lora_path, 'non_lora_trainables.bin'), map_location='cpu')
                mm_projector_weights = {k[6:]: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                print("load new projectors....", mm_projector_weights.keys())
                status = model.load_state_dict(mm_projector_weights, strict=False)
                print('load pretrain_mm_mlp_adapter, unexpected_keys:{}'.format(status.unexpected_keys))

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    
    speech_tower = model.get_speech_tower()
    if speech_tower is not None:
        if not speech_tower.is_loaded:
            speech_tower.load_model()
        speech_tower.to(device=device, dtype=torch.float16)
        
    if model_lora_path is not None:
        model.load_adapter(model_lora_path)
        print(f"Loading LoRA weights from {model_lora_path}")
        model.to(torch.float16)
    
    if merge_lora_path is not None:
        # model.load adapter(model lora path)
        print(f"Loading LoRA weights from {merge_lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, merge_lora_path)
        print(f"Merging weights")
        model = model.merge_and_unload()
        print('convert to FP16...')
        model.to(torch.float16)
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
        
    return tokenizer, model, image_processor, context_len
