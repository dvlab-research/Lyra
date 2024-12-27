# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import re

import torch
import numpy as np
import torchaudio
from torchaudio.transforms import Resample

import transformers
import whisper
import tokenizers

from qwen_vl_utils import process_vision_info

from lyra.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_SPEECH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from torch.utils.data import Dataset
from lyra.train.lyra_trainer import LyraTrainer

from lyra import conversation as conversation_lib
from lyra.model import *
from lyra.mm_utils import tokenizer_image_token, tokenizer_image_speech_token, tokenizer_speech_token, process_highres_image, process_anyres_image, process_highres_image_crop_split

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_speech_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    speech_tower: Optional[str] = field(default=None)
    optimize_vision_tower: bool = field(default=False) # whether to optimize vision tower
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_speech_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_speech_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    compress: bool = field(default=False)
    compress_gap: int = field(default=33)
    keep_rate: float = field(default=1.0)
    keep_global: bool = field(default=False)
    down_factor_1: int = field(default=1)
    down_factor_2: int = field(default=1)
    train_modality: Optional[str] = field(default='text_image')
    asr_align: bool = field(default=False)
    check_data_modality: bool = field(default=False)
    align_temperature: float = field(default=1.0)
    weight_lambda: float = field(default=0.5)
    speech_norm: bool = field(default=True)
    speech_learn: bool = field(default=False)
    align_norm: bool = field(default=True)
    align_type: Optional[str] = field(default='dtw')
    speech_encoder_ds_rate: int = field(default=5)
    speech_encoder_hidden_size: int = field(default=1280)
    generate: bool = field(default=False)
    ctc_decoder_config: str = "(2,4096,32,11008)"
    ctc_upsample_factor: int = 25
    ctc_loss_weight: float = 1.0
    unit_vocab_size: int = 1000
    tune_speech_generator_only: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    is_speech_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    speech_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    speech_overlap: bool = False
    speech_overlap_time: int = field(default=30)
    image_grid_pinpoints: Optional[str] = field(default=None)
    overlap_ratio: float = field(default=0.0)
    image_grid: Optional[int] = field(default=1)
    image_global: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_speech_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    target_modules: str = 'v_proj,o_proj,down_proj,gate_proj,q_proj,k_proj,up_proj'
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler', 'vlm_uni']
        # add vision tower
        keys_to_match.extend(['vision_tower'])
        
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
    if getattr(trainer.args, "tune_mm_mlp_adapter", False) or getattr(trainer.args, "tune_mm_speech_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_speech_projector']
        
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_speech_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_speech_projector.bin'))
        # return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            replace_token = '<|vision_start|>' + replace_token + '<|vision_end|>'
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN + '\n', replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1

        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt

        # include <bos> for all rounds
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            # include <bos> for all rounds
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # include <|eot_id|> for all rounds
            round_len += 1
            instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_3_1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt

        # include <bos> for all rounds
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            # include <bos> for all rounds
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # include <|eot_id|> for all rounds
            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len += 2
                instruction_len += 2
                # # Hermes-3-Llama-3.1-8B
                # round_len += 1
                # instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen2vl(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image and has_speech:
        input_ids = torch.stack([tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    
    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN2VL
    # assert conv.sep_style == conversation_lib.SeparatorStyle.CHATML

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        
        # include <bos> for all rounds
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep
            # include <bos> for all rounds
            if has_image and has_speech:
                round_len = len(tokenizer_image_speech_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_speech_token(parts[0], tokenizer)) - 2
            elif has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            elif has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # include <|eot_id|> for all rounds
            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len += 3
                instruction_len += 3

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len += 1

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image and has_speech:
        input_ids = torch.stack([tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    
    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN2


    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        
        # include <bos> for all rounds
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            # include <bos> for all rounds
            if has_image and has_speech:
                round_len = len(tokenizer_image_speech_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_speech_token(parts[0], tokenizer)) - 2
            elif has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            elif has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # include <|eot_id|> for all rounds
            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len += 3
                instruction_len += 3

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len += 1

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # not included <|im_end|>
            if has_image: 
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # include <|im_end|> for all rounds
            # if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
            if getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)

def preprocess_speech_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_SPEECH_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_SPEECH_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_speech_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_speech: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.SPEECH_PLAIN:
        return preprocess_speech_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1:
        return preprocess_llama_3_1(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN2:
        return preprocess_qwen2(sources, tokenizer, has_image=has_image, has_speech=has_speech)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN2VL:
        return preprocess_qwen2vl(sources, tokenizer, has_image=has_image, has_speech=has_speech)
    
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if 'image' in sample and 'speech' in sample:
                cur_len = cur_len + 0.5
            elif 'image' in sample:
                cur_len = cur_len
            else:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list
    
    def _preprocess_image_qwen2vl(self, image, image_resolution):

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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 10
        while attempt < max_attempt:
            try:
                # sample an item
                data_dict = self._sample_item(i)
                break
            except:
                attempt += 1
                print(f"Error in loading {i}, retrying...")
                i = random.randint(0, len(self.list_data_dict) - 1)
    
        return data_dict

    def _sample_item(self, i) -> Dict[str, torch.Tensor]:

        image = None
        sources = self.list_data_dict[i]
        suffix = None
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        if 'speech_asr' in sources[0]:
            speech_asr = self.list_data_dict[i]['speech_asr']
            speech_asr = self.tokenizer(speech_asr, return_tensors="pt").input_ids
        
        if 'image' in sources[0]:
            image_files = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            
            image_total = []
            if not isinstance(image_files, list):
                image_files = [image_files]
            
            for image_file in image_files:

                image_path = os.path.join(image_folder, image_file)
                if not os.path.exists(image_path):
                    image_path = image_path.replace(".jpg", ".gif")

                if not os.path.exists(image_path):
                    image_path = image_path.replace(".jpg", ".png")
                    image_path = image_path.replace(".gif", ".png")

                image = Image.open(image_path).convert('RGB')
                    
                if self.data_args.image_aspect_ratio == "qwen2vl":
                    image = self._preprocess_image_qwen2vl(image, image_resolution=2560)
                    image = self.data_args.image_processor(images=image, return_tensors="pt")
                elif self.data_args.image_aspect_ratio == "highres":
                    image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
                elif self.data_args.image_aspect_ratio == "anyres" or "anyres_max" in self.data_args.image_aspect_ratio:
                    image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints, self.data_args.overlap_ratio)
                elif self.data_args.image_aspect_ratio == "crop_split":
                    image = process_highres_image_crop_split(image, self.data_args)
                elif self.data_args.image_aspect_ratio == "plain_llava":
                    def make_image_pieces(image, up_scale):
                        if hasattr(self.data_args.image_processor, "crop_size"):
                            newsize = (self.data_args.image_processor.crop_size['height'] * up_scale, self.data_args.image_processor.crop_size['width'] * up_scale)
                        elif hasattr(self.data_args.image_processor, "size"):
                            newsize = (self.data_args.image_processor.size['height'] * up_scale, self.data_args.image_processor.size['width'] * up_scale)
                        image_up = image.resize(newsize)

                        M = image_up.size[0] // up_scale
                        N = image_up.size[1] // up_scale
                        image_up_array = np.array(image_up)
                        tiles = [Image.fromarray(image_up_array[x:x + M, y:y + N])
                                for x in range(0, image_up.size[0], M)
                                for y in range(0, image_up.size[1], N)]

                        return tiles
                    image = make_image_pieces(image, 1)
                    image = [processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image]

        elif 'video' in sources[0]:
            video_files = self.list_data_dict[i]['video']
            video_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            
            if not isinstance(video_files, list):
                video_files = [video_files]
            
            for video_file in video_files:
                video_path = os.path.join(video_folder, video_file)
                if self.data_args.image_aspect_ratio == "qwen2vl":
                    total_pixels = 8192 * 28 * 28
                    suffix = video_file.split('.')[-1]
                    if suffix == 'mp4' or suffix == 'mkv':
                        messages = [{"role": "user", "content": [{"type": "video", "video": video_path, "total_pixels": total_pixels, "min_pixels": 28 * 28}]}]
                        image_inputs, video_inputs = process_vision_info(messages)
                        image = processor(images=None, videos=video_inputs[0], return_tensors="pt")
                    elif suffix == 'jpg' or suffix == 'png' or suffix == 'jpeg':
                        image = Image.open(video_path).convert('RGB')
                        
                        image = np.array(image)
                        non_white_rows = np.any(image.sum(axis=-1) < 720, axis=1)
                        non_white_cols = np.any(image.sum(axis=-1) < 720, axis=0)
                        cat_image = image[non_white_rows][:, non_white_cols]

                        img_height, img_width, _ = cat_image.shape
                        piece_width, piece_height = img_width // 4, img_height // 4
                        cat_image = Image.fromarray(cat_image)

                        images = []
                        for row in range(4):
                            for col in range(4):
                                left = col * piece_width
                                upper = row * piece_height
                                right = left + piece_width
                                lower = upper + piece_height
                                cropped_img = cat_image.crop((left, upper, right, lower))
                                images.append(cropped_img)
                        images = [processor.preprocess(img, return_tensors='pt') for img in images]
                        
                        pixel_values_videos = torch.cat([x['pixel_values'] for x in images])
                        video_grid_thw = images[0]['image_grid_thw']
                        video_grid_thw[0,0] = len(images)
                        image = dict(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw)
                    else:
                        video_root = video_path
                        video_frames = os.listdir(video_path)
                        video_path = [os.path.join(video_root, x) for x in video_frames]
                        max_pixels = int(total_pixels / len(video_frames) * 2)
                        messages = [{"role": "user", "content": [{"type": "video", "video": video_path, "max_pixels": max_pixels}]}]  
                        image_inputs, video_inputs = process_vision_info(messages)
                        image = processor(images=None, videos=video_inputs[0], return_tensors="pt")
                else:
                    raise NotImplementedError
        
        # preprocess speech generation part:
        if 'tgt_unit' in sources[0]:
            tgt_units = self.list_data_dict[i]['tgt_unit']
            numbers = re.findall(r'<(\d+)>', tgt_units)
            numbers = [int(num) for num in numbers]
        # preprocess speech file:   
        if 'speech' in sources[0]:
            speech_files = self.list_data_dict[i]['speech']
            speech_folder = self.data_args.speech_folder
            processor = self.data_args.speech_processor

            if not isinstance(speech_files, list):
                speech_files = [speech_files]

            speech_files = [os.path.join(speech_folder, speech_file) for speech_file in speech_files]
            
            speech_total = []
            for speech_file in speech_files:
                wav, sample_rate = torchaudio.load(speech_file)
                target_sample_rate = 16000
                resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                wav = resample_transform(wav)
                if wav.ndim != 1:
                    wav = wav[0]
                    
                speech = processor(raw_speech=wav, 
                                sampling_rate=target_sample_rate, 
                                return_tensors="pt", 
                                return_attention_mask=True)
                speech_total.append(speech["input_features"].squeeze())  # (mels, lengths), e.g. (128, 3000)

            if len(speech_total) > 1:
                speech = torch.stack(speech_total, dim=0)
            else: # always in this branch in current version
                speech = speech_total[0]  
        
        if 'long_speech' in sources[0]:
            speech_files = self.list_data_dict[i]['long_speech']
            speech_folder = self.data_args.speech_folder
            processor = self.data_args.speech_processor

            if not isinstance(speech_files, list):
                speech_files = [speech_files]

            speech_files = [os.path.join(speech_folder, speech_file) for speech_file in speech_files]
            
            speech_total = []
            for speech_file in speech_files:
                
                speech_total = []
                wav, sample_rate = torchaudio.load(speech_file)
                target_sample_rate = 16000
                resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                wav = resample_transform(wav)
                if wav.ndim != 1:
                    wav = wav[0]
                if self.data_args.speech_overlap:
                    temp_wav = wav.unfold(0, target_sample_rate * 30, target_sample_rate * self.data_args.speech_overlap_time)
                    temp_wav = [wav_clip for wav_clip in temp_wav]
                    clip_num = len(temp_wav)
                    temp_wav.append(wav[target_sample_rate * self.data_args.speech_overlap_time * clip_num:])
                    for wav_clip in temp_wav:
                        speech_tensor = processor(raw_speech=wav_clip, 
                                            sampling_rate=target_sample_rate, 
                                            return_tensors="pt", 
                                            return_attention_mask=True)["input_features"].squeeze() # (128,3000)
                        speech_total.append(speech_tensor)
                else:
                    whipser_len = target_sample_rate * 30
                    speech_num = wav.shape[0] // whipser_len + 1
                    for speech_num_idx in range(speech_num):
                        temp_wav = wav[speech_num_idx*whipser_len:(speech_num_idx+1)*whipser_len]
                        speech_tensor = processor(raw_speech=temp_wav, 
                                            sampling_rate=target_sample_rate, 
                                            return_tensors="pt", 
                                            return_attention_mask=True)["input_features"].squeeze() # (128,3000)
                        speech_total.append(speech_tensor)
                
                
            if len(speech_total) > 1:
                speech = torch.stack(speech_total, dim=0)
            else: # always in this branch in current version
                speech = speech_total[0]   
            # print('speech input shape is :', speech.shape)
        if 'image' in sources[0] or 'speech' in sources[0] or 'long_speech' in sources[0]:
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
                
        has_image = ('image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i])
        has_speech = ('speech' in self.list_data_dict[i]) or ('long_speech' in self.list_data_dict[i])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image,
            has_speech=has_speech)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # image exist in the data
        if has_image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            if self.data_args.image_aspect_ratio == "qwen2vl":
                tmp_image = 'examples/placeholder.jpg'
                tmp_image = Image.open(tmp_image).convert('RGB')
                data_dict['image'] = self.data_args.image_processor(images=tmp_image, return_tensors="pt")
            elif hasattr(self.data_args.image_processor, "crop_size"):
                data_dict['image'] = torch.stack([torch.zeros(3, self.data_args.image_processor.crop_size['height'], self.data_args.image_processor.crop_size['width'])] * 2)
            elif hasattr(self.data_args.image_processor, "size"):
                data_dict['image'] = torch.stack([torch.zeros(3, self.data_args.image_processor.size['height'], self.data_args.image_processor.size['width'])] * 2)
                
        # append speech
        if 'speech' in self.list_data_dict[i] or 'long_speech' in self.list_data_dict[i]:
            data_dict['speech'] = speech
        elif self.data_args.is_speech_multimodal:
            data_dict['speech'] = torch.zeros(128, 3000)
        
        if 'speech_asr' in self.list_data_dict[i]:
            data_dict['speech_asr'] = speech_asr.squeeze()
        elif self.data_args.is_speech_multimodal:
            data_dict['speech_asr'] = torch.zeros(100).to(device=data_dict["input_ids"].device, dtype=data_dict["input_ids"].dtype)
        
        if 'tgt_unit' in self.list_data_dict[i]:
            data_dict['tgt_unit'] = torch.tensor(numbers)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['images'] = images

        if 'speech' in instances[0]:
            speeches = [instance['speech'] for instance in instances]
            batch['speeches'] = speeches
        
        if 'speech_asr' in instances[0]:
            speeches_asr = [instance['speech_asr'] for instance in instances]
            batch['speeches_asr'] = speeches_asr
        
        if 'tgt_unit' in instances[0]:
            target_units = [instance['tgt_unit'] for instance in instances]
            if all(x is not None and x.shape == target_units[0].shape for x in target_units):
                batch['tgt_units'] = torch.stack(target_units)

            else:
                target_units = torch.nn.utils.rnn.pad_sequence(target_units,
                                                        batch_first=True,
                                                        padding_value=IGNORE_INDEX)
                batch['tgt_units'] = target_units

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    if model_args.vision_tower is not None or \
        model_args.speech_tower is not None:
        if "qwen2vl" in model_args.model_name_or_path.lower():
            if model_args.compress:
                model = LyraQwen2VLForCausalLMExtractor.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            elif model_args.generate:
                model = Lyra2SQwen2VLForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                model = LyraQwen2VLForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
        elif "qwen" in model_args.model_name_or_path.lower():
            model = LyraQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.target_modules.split(','),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        # fix bugs after special token with use_fast=True
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if "qwen2vl" == model_args.version:
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen2vl"]
    elif "qwen_2" == model_args.version:
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2"]
    elif "llama_3" == model_args.version:
        # set unknown token and pad token to the first reserved special token
        if tokenizer.unk_token is None:
            tokenizer.unk_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3"]
    elif "llama_3_1" == model_args.version:
        # set unknown token and pad token to the first reserved special token
        if tokenizer.unk_token is None:
            tokenizer.unk_token = "<|reserved_special_token_2|>"
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3_1"]
    else:
        #  for Qwen-2 plain training
        if tokenizer.unk_token is None:
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = copy.deepcopy(vision_tower.image_processor)
        
        data_args.video_processor = copy.deepcopy(vision_tower.image_processor)
        data_args.is_multimodal = True

        model.config.image_grid = data_args.image_grid
        model.config.image_global = data_args.image_global
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.speech_overlap = data_args.speech_overlap
        model.config.speech_overlap_time = data_args.speech_overlap_time
        model.config.overlap_ratio = data_args.overlap_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
        model.config.compress = model_args.compress
        model.config.compress_gap = model_args.compress_gap
        model.config.keep_rate = model_args.keep_rate
        model.config.keep_global = model_args.keep_global
        model.config.down_factor_1 = model_args.down_factor_1
        model.config.down_factor_2 = model_args.down_factor_2
        model.config.train_modality = model_args.train_modality
        model.config.check_data_modality = model_args.check_data_modality
        
        model.config.asr_align = model_args.asr_align
        

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        if model_args.optimize_vision_tower:
            vision_tower.requires_grad_(True)
            # print('Optimize last 1/2 layers in vision tower')
            # total_num = len(vision_tower.vision_tower.vision_model.encoder.layers)
            # for _idx in range(total_num//2, total_num):
            #     vision_tower.vision_tower.vision_model.encoder.layers[_idx].requires_grad_(True)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Start: add speech module
    if model_args.generate:
        model.initialize_speech_generator(
            model_args=model_args
        )
        
        speech_generator = model.get_speech_generator()
        speech_generator.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        
        model.config.generate = model_args.generate
        model.config.ctc_decoder_config = model_args.ctc_decoder_config
        model.config.ctc_upsample_factor = model_args.ctc_upsample_factor
        model.config.ctc_loss_weight = model_args.ctc_loss_weight
        model.config.unit_vocab_size = model_args.unit_vocab_size
        
        model.config.tune_speech_generator_only = model_args.tune_speech_generator_only
        if model_args.tune_speech_generator_only:
            model.requires_grad_(False)
            for p in model.speech_generator.parameters():
                p.requires_grad = True
        
    if model_args.speech_tower is not None:
        speech_tower_name = model_args.speech_tower
        
        model.get_model().initialize_speech_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        speech_tower = model.get_speech_tower()
        speech_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.speech_processor = copy.deepcopy(speech_tower.speech_processor)
        # if "whisper" in speech_tower_name.lower():
        #     from transformers import WhisperFeatureExtractor
        #     data_args.speech_processor = WhisperFeatureExtractor.from_pretrained(model_args.speech_tower)
        # else:
        #     raise NotImplementedError
            
        data_args.speech_processor_type = speech_tower_name.lower()
        data_args.is_speech_multimodal = True
        
        model.config.train_modality = model_args.train_modality
        model.config.check_data_modality = model_args.check_data_modality
        model.config.asr_align = model_args.asr_align
        model.config.align_temperature = model_args.align_temperature
        model.config.weight_lambda = model_args.weight_lambda
        model.config.speech_encoder_ds_rate = model_args.speech_encoder_ds_rate
        model.config.speech_encoder_hidden_size = model_args.speech_encoder_hidden_size
        model.config.speech_norm = model_args.speech_norm
        model.config.speech_learn = model_args.speech_learn
        model.config.align_norm = model_args.align_norm
        model.config.align_type = model_args.align_type
        
        

        model.config.tune_mm_speech_mlp_adapter = training_args.tune_mm_speech_mlp_adapter = model_args.tune_mm_speech_mlp_adapter
        if model_args.tune_mm_speech_mlp_adapter:
            model.requires_grad_(False)  # BEAWARE: Need to remove in case vision tower become false
            for p in model.get_model().mm_speech_projector.parameters():
                p.requires_grad = True
        
        model.config.freeze_mm_speech_mlp_adapter = training_args.freeze_mm_speech_mlp_adapter
        if training_args.freeze_mm_speech_mlp_adapter:
            for p in model.get_model().mm_speech_projector.parameters():
                p.requires_grad = False
                
        if training_args.bits in [4, 8]:
            model.get_model().mm_speech_projector.to(dtype=compute_dtype, device=training_args.device)
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    # Check learnable module to a txt file
    # with open('model_requires_grad_status.txt', 'w') as f:
    #     for name, param in model.named_parameters():
    #         f.write(f"Parameter: {name}, requires_grad: {param.requires_grad}\n")
            
    trainer = LyraTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.base_model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
