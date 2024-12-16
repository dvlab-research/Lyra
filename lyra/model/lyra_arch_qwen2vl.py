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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import transformers
import safetensors
from transformers.deepspeed import is_deepspeed_zero3_enabled
import deepspeed

from .multimodal_encoder.builder import build_vision_tower, build_speech_tower, build_audio_tower
from .multimodal_projector.builder import build_vision_projector, build_speech_projector, build_audio_projector

from lyra.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, SPEECH_TOKEN_INDEX,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)

from lyra.model.soft_dtw_cuda import SoftDTW

class LyraMetaModel:

    def __init__(self, config):
        super(LyraMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        
        if hasattr(config, "mm_speech_tower"):
            self.speech_tower = build_speech_tower(config, delay_load=True)
            self.mm_speech_projector = build_speech_projector(config)
        
        if hasattr(config, "mm_audio_tower"):
            self.audio_tower = build_audio_tower(config, delay_load=True)
            self.mm_audio_projector = build_audio_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_speech_tower(self):
        speech_tower = getattr(self, 'speech_tower', None)
        if type(speech_tower) is list:
            speech_tower = speech_tower[0]
        return speech_tower

    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_use_mrope = getattr(model_args, 'mm_use_mrope', False)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}

            if 'model' in mm_projector_weights.keys():
                mm_projector_weights = mm_projector_weights['model']
                if is_deepspeed_zero3_enabled():
                    if len(mm_projector_weights) > 0:
                        with deepspeed.zero.GatheredParameters(mm_projector_weights, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                status = self.mm_projector.load_state_dict(mm_projector_weights)
                                print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_mlp_adapter, status.missing_keys))
                else:
                    status = self.mm_projector.load_state_dict(mm_projector_weights, strict=False)
                    print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_mlp_adapter, status.missing_keys))
            else:
                if is_deepspeed_zero3_enabled():
                    named_parameters = get_w(mm_projector_weights, 'mm_projector')
                    if len(named_parameters) > 0:
                        with deepspeed.zero.GatheredParameters(named_parameters, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                status = self.mm_projector.load_state_dict(named_parameters)
                                print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_mlp_adapter, status.missing_keys))
                else:
                    status = self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
                    print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_mlp_adapter, status.missing_keys))
            self.mm_projector = self.mm_projector.to(device='cuda')
            
    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        pretrain_mm_audio_mlp_adapter = model_args.pretrain_mm_audio_mlp_adapter

        self.config.mm_audio_tower = audio_tower

        if self.get_audio_tower() is None:
            audio_tower = build_audio_tower(model_args)
            audio_tower.requires_grad_(False)

            if fsdp is not None and len(fsdp) > 0:
                self.audio_tower = [audio_tower]
            else:
                self.audio_tower = audio_tower
        else:
            audio_tower.requires_grad_(False)
            if fsdp is not None and len(fsdp) > 0:
                self.audio_tower = [audio_tower]
            else:
                self.audio_tower = audio_tower
            audio_tower.load_model()

        
        self.config.mm_audio_projector_type = getattr(model_args, 'mm_audio_projector_type', 'linear')
        self.config.mm_audio_hidden_size = 1024  # for imagebind huge

        if getattr(self, 'mm_audio_projector', None) is None:
            self.mm_audio_projector = build_audio_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_audio_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_audio_mlp_adapter is not None:
            mm_audio_projector_weights = torch.load(pretrain_mm_audio_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}

            if 'model' in mm_audio_projector_weights.keys():
                mm_audio_projector_weights = mm_audio_projector_weights['model']
                if is_deepspeed_zero3_enabled():
                    if len(mm_audio_projector_weights) > 0:
                        with deepspeed.zero.GatheredParameters(mm_audio_projector_weights, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                self.mm_audio_projector.load_state_dict(mm_audio_projector_weights)
                else:
                    status = self.mm_audio_projector.load_state_dict(mm_audio_projector_weights, strict=False)
                    print('missing_keys:', status.missing_keys)
            else:
                if is_deepspeed_zero3_enabled():
                    named_parameters = get_w(mm_audio_projector_weights, 'mm_audio_projector')
                    if len(named_parameters) > 0:
                        with deepspeed.zero.GatheredParameters(named_parameters, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                self.mm_audio_projector.load_state_dict(named_parameters)
                else:
                    status = self.mm_audio_projector.load_state_dict(get_w(mm_audio_projector_weights, 'mm_audio_projector'), strict=False)
                    print('missing_keys:', status.missing_keys)
            self.mm_audio_projector = self.mm_audio_projector.to(device='cuda')
            
    def initialize_speech_modules(self, model_args, fsdp=None):  
        speech_tower = model_args.speech_tower
        pretrain_mm_speech_mlp_adapter = model_args.pretrain_mm_speech_mlp_adapter
        self.config.mm_speech_tower = speech_tower
        self.config.speech_encoder_ds_rate = model_args.speech_encoder_ds_rate
        self.config.speech_encoder_hidden_size = model_args.speech_encoder_hidden_size
        if self.get_speech_tower() is None:
            speech_tower = build_speech_tower(model_args)
            speech_tower.requires_grad_(False)

            if fsdp is not None and len(fsdp) > 0:
                self.speech_tower = [speech_tower]
            else:
                self.speech_tower = speech_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                speech_tower = self.speech_tower[0]
            else:
                speech_tower = self.speech_tower
            speech_tower.load_model()

        self.config.mm_speech_projector_type = getattr(model_args, 'mm_speech_projector_type', 'linear')
        if 'encodec' in model_args.speech_tower.lower():
            self.config.mm_speech_hidden_size = speech_tower.config.codebook_dim
        elif 'whisper' in model_args.speech_tower.lower():
            self.config.mm_speech_hidden_size = speech_tower.config.d_model # d_model for whisper and codebook_dim for encodec
        else:
            raise NotImplementedError

        self.soft_dtw = SoftDTW(use_cuda=True, gamma=0.1, contrastive_temp=model_args.align_temperature)
        
        if getattr(self, 'mm_speech_projector', None) is None:
            self.mm_speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_speech_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_speech_mlp_adapter is not None:
            mm_speech_projector_weights = torch.load(pretrain_mm_speech_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}

            if 'model' in mm_speech_projector_weights.keys():
                mm_speech_projector_weights = mm_speech_projector_weights['model']
                if is_deepspeed_zero3_enabled():
                    if len(mm_speech_projector_weights) > 0:
                        with deepspeed.zero.GatheredParameters(mm_speech_projector_weights, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                status = self.mm_speech_projector.load_state_dict(mm_speech_projector_weights)
                                print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_speech_mlp_adapter, status.missing_keys))
                else:
                    status = self.mm_speech_projector.load_state_dict(mm_speech_projector_weights, strict=False)
                    print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_speech_mlp_adapter, status.missing_keys))
            else:
                if is_deepspeed_zero3_enabled():
                    named_parameters = get_w(mm_speech_projector_weights, 'mm_speech_projector')
                    if len(named_parameters) > 0:
                        with deepspeed.zero.GatheredParameters(named_parameters, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                status = self.mm_speech_projector.load_state_dict(named_parameters)
                                print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_speech_mlp_adapter, status.missing_keys))
                else:
                    status = self.mm_speech_projector.load_state_dict(get_w(mm_speech_projector_weights, 'mm_speech_projector'), strict=False)
                    print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_speech_mlp_adapter, status.missing_keys))
            self.mm_speech_projector = self.mm_speech_projector.to(device='cuda')
    
class LyraMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_speech_tower(self):
        return self.get_model().get_speech_tower()

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def encode_images(self, images, is_video=False):
        if "qwen2vl" in self.config.mm_vision_tower.lower():
            image_feature = self.get_model().get_vision_tower()(images)
            return image_feature
        
        if isinstance(images, list):
            image_features = []
            for img in images:
                if len(img.shape) == 3:
                    image_feature = self.get_model().get_vision_tower()(img[None, :])
                else:
                    image_feature = self.get_model().get_vision_tower()(img)
                image_feature = self.get_model().mm_projector(image_feature)
                single_img_token_num = image_feature.shape[-2]
                llm_dim = image_feature.shape[-1]
                # [image_original_resize] + patches
                if getattr(self.config, 'video_inference', False) is False:
                    global_image_feature = image_feature[0, :]
                    grid_image_feature = image_feature[1:, :]
                
                    grid_image_feature = grid_image_feature.reshape(-1, int(single_img_token_num ** 0.5), int(single_img_token_num ** 0.5), llm_dim)
                    downsample = torch.nn.Upsample(scale_factor=self.config.down_factor_1 / self.config.down_factor_2, mode='bilinear')
                    grid_image_feature = grid_image_feature.permute(0, 3, 1, 2)
                    grid_image_feature = downsample(grid_image_feature)
                    grid_image_feature = grid_image_feature.flatten(-2, -1).contiguous()
                    grid_image_feature = grid_image_feature.permute(0, -1, 1).flatten(0, 1).contiguous()
                    image_feature = torch.cat([global_image_feature, grid_image_feature], dim=0)
                else:
                    grid_image_feature = image_feature[0:, :]
                    grid_image_feature = grid_image_feature.reshape(-1, int(single_img_token_num ** 0.5), int(single_img_token_num ** 0.5), llm_dim)
                    downsample = torch.nn.Upsample(scale_factor=self.config.down_factor_1 / self.config.down_factor_2, mode='bilinear')
                    grid_image_feature = grid_image_feature.permute(0, 3, 1, 2)
                    grid_image_feature = downsample(grid_image_feature)
                    grid_image_feature = grid_image_feature.flatten(-2, -1).contiguous()
                    grid_image_feature = grid_image_feature.permute(0, -1, 1).flatten(0, 1).contiguous()
                    image_feature = torch.cat([grid_image_feature], dim=0)

                image_features.append(image_feature)
        
        else:
            batch_size = images.shape[0]
            images = images.flatten(0, 1).contiguous()
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            single_img_token_num = image_features.shape[-2]
            llm_dim = image_features.shape[-1]
            image_features = image_features.reshape(batch_size, -1, single_img_token_num, llm_dim)
            image_features = image_features.flatten(1, 2).contiguous()
        return image_features
 
    def encode_speeches(self, speeches):
        if isinstance(speeches, list):
            if len(speeches[0].shape) == 2:
                speeches = torch.stack(speeches, dim=0)
                speech_features = self.get_model().get_speech_tower()(speeches)
                speech_features = self.get_model().mm_speech_projector(speech_features)
                if self.config.speech_norm:
                    speech_features = torch.nn.functional.normalize(speech_features, dim=-1)
                return speech_features
            
            # long speech case
            if len(speeches[0].shape) == 3:
                speech_features = []
                for speech in speeches:
                    speech_feature = self.get_model().get_speech_tower()(speech)
                    speech_feature = self.get_model().mm_speech_projector(speech_feature)
                    if self.config.speech_norm:
                        speech_feature = torch.nn.functional.normalize(speech_feature, dim=-1)
                    speech_features.append(speech_feature.flatten(0, 1).contiguous())
                return speech_features
        else:
            speech_features = self.get_model().get_speech_tower()(speeches)
            speech_features = self.get_model().mm_speech_projector(speech_features)
            if self.config.speech_norm:
                speech_features = torch.nn.functional.normalize(speech_features, dim=-1)
            return speech_features

    
    def align_speech_with_asr_embed(self, speech_features, speeches_asr):
        loss_align = 0.
        if self.config.speech_learn:
            speeches_asr_embeds = [self.get_model().embed_tokens(speech_asr) for speech_asr in speeches_asr]
        else:
            with torch.no_grad():
                speeches_asr_embeds = [self.get_model().embed_tokens(speech_asr) for speech_asr in speeches_asr]
        
        count = 0
        for idx in range(len(speeches_asr_embeds)):            
            if self.config.align_type == 'dtw':
                speech_feature = speech_features[idx]
                speeches_asr_embed = speeches_asr_embeds[idx]
                if self.config.align_norm:
                    speech_feature = torch.nn.functional.normalize(speech_feature, dim=-1)
                    speeches_asr_embed = torch.nn.functional.normalize(speeches_asr_embed, dim=-1)
                loss_align_item = (self.get_model().soft_dtw(speech_feature[None, :].to(torch.float32), speeches_asr_embed[None, :].to(torch.float32))[0] / speech_feature.shape[0]).to(dtype=speech_feature.dtype)
                
                if torch.all(speeches_asr[idx] == 0).item():
                    loss_align += loss_align_item * 0.
                else:
                    loss_align += loss_align_item
                    count += 1
                    
            elif self.config.align_type == 'truncated':
                speech_features_align = speech_features[idx, -speeches_asr_embeds[idx].shape[0]:]
                if self.config.align_norm:
                    logit = torch.nn.functional.normalize(speech_features_align, dim=-1) @ torch.nn.functional.normalize(speeches_asr_embeds[idx], dim=-1).T
                else:
                    logit = speech_features_align @ speeches_asr_embeds[idx].T
                logit /= self.config.align_temperature
                label = torch.arange(0, logit.shape[0], device=logit.device, dtype=torch.long)
                loss_align_item = nn.functional.cross_entropy(logit, label)
                
                if torch.all(speeches_asr[idx] == 0).item():
                    loss_align += loss_align_item * 0.
                else:
                    loss_align += loss_align_item
                    count += 1
                    
        if count > 0:    
            loss_align /= count

        return loss_align
    

    def prepare_inputs_labels_for_text_image_speech_qwen2vl(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images=None, speeches=None, speeches_asr=None, rope_deltas=None
    ):
        vision_tower = self.get_vision_tower()
        speech_tower = self.get_speech_tower()

        if ((vision_tower is None or images is None) and  # if do not have images
            (speech_tower is None or speeches is None) # and do not have speeches
            ) or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and input_ids.shape[1] == 1:

                target_shape = past_key_values[0][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

                if rope_deltas is not None:
                    for i in range(len(rope_deltas)):
                        position_ids[i] += rope_deltas[i]
                position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, rope_deltas

        if vision_tower is not None and images is not None:
            image_features = self.encode_images(images)

        loss_align = None
        if speech_tower is not None and speeches is not None:
            speech_features = self.encode_speeches(speeches)
            if self.config.asr_align and self.training:
                loss_align = self.align_speech_with_asr_embed(speech_features, speeches_asr)

        if self.config.check_data_modality:
            all_have = True
            all_not_have = True
            for input_id in input_ids:
                all_have = all_have and (IMAGE_TOKEN_INDEX in input_id)
                all_not_have = all_not_have and (IMAGE_TOKEN_INDEX not in input_id)
            print("image: all have", all_have, ", all not have", all_not_have)
            same_modality = all_have or all_not_have
            
            all_have = True
            all_not_have = True
            for input_id in input_ids:
                all_have = all_have and (SPEECH_TOKEN_INDEX in input_id)
                all_not_have = all_not_have and (SPEECH_TOKEN_INDEX not in input_id)
            print("speech: all have ", all_have, ", all not have", all_not_have)
            
            same_modality = same_modality and (all_have or all_not_have)
            print("same modality?", same_modality)
            if not same_modality:
                print("!!!!!!!!!!!!!!!!  not same modality  !!!!!!!!!!!!!!!!")
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_position_ids = []
        rope_deltas = []
        cur_im_idx = 0
        cur_sp_idx = 0
        cur_device = input_ids[0].device
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_mm = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() + (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            if num_mm == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                
                if images is not None:
                    cur_image_features = image_features[cur_im_idx]
                    cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0]], dim=0)
                    cur_im_idx += 1
                if speeches is not None:
                    cur_sp_features = speech_features[cur_sp_idx]
                    cur_input_embeds = torch.cat([cur_input_embeds, cur_sp_features[0:0]], dim=0)
                    cur_sp_idx += 1

                cur_position_ids = torch.arange(len(cur_input_ids), dtype=torch.long, device=cur_device)
                cur_position_ids = cur_position_ids.unsqueeze(0).repeat(3, 1)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_position_ids.append(cur_position_ids)
                continue
            
            # must <image>\n<speech> image frist speech last
            mm_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nomm = []
            cur_labels = labels[batch_idx]
            cur_labels_nomm = []
            for i in range(len(mm_token_indices) - 1):
                cur_input_ids_nomm.append(cur_input_ids[mm_token_indices[i]+1:mm_token_indices[i+1]])
                cur_labels_nomm.append(cur_labels[mm_token_indices[i]+1:mm_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nomm]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nomm))
            
            # gradient for all modalities
            if images is not None:
                cur_image_features = image_features[cur_im_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0].to(device=cur_input_embeds.device)], dim=0)
            if speeches is not None:
                cur_sp_features = speech_features[cur_sp_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_sp_features[0:0].to(device=cur_input_embeds.device)], dim=0)
                
            cur_input_embeds_no_mm = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_position_ids = []

            max_pos_id = 0
            mrope_delta = 0
            for i in range(num_mm + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_mm[i])
                cur_new_labels.append(cur_labels_nomm[i])
                _pos_ids = torch.arange(split_sizes[i], dtype=torch.long, device=cur_device) + max_pos_id
                _pos_ids = _pos_ids.unsqueeze(0).repeat(3, 1)
                cur_position_ids.append(_pos_ids)
                max_pos_id += cur_input_embeds_no_mm[i].shape[0]
                if i < num_mm:
                    if cur_input_ids[mm_token_indices[i+1]] == IMAGE_TOKEN_INDEX:
                        cur_image_features = image_features[cur_im_idx]
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_im_idx += 1

                        if hasattr(images[batch_idx], 'image_grid_thw'):
                            if self.training:
                                t, h, w = images[batch_idx]['image_grid_thw'][0]
                            else:
                                if len(images[batch_idx]['image_grid_thw'].shape) == 3:
                                    t = images[batch_idx]['image_grid_thw'][0, 0, 0].item()
                                    h = images[batch_idx]['image_grid_thw'][0, 0, 1].item()
                                    w = images[batch_idx]['image_grid_thw'][0, 0, 2].item()
                                elif len(images[batch_idx]['image_grid_thw'].shape) == 2:
                                    t = images[batch_idx]['image_grid_thw'][0, 0].item()
                                    h = images[batch_idx]['image_grid_thw'][0, 1].item()
                                    w = images[batch_idx]['image_grid_thw'][0, 2].item()
                                else:
                                    t, h, w = images[batch_idx]['image_grid_thw'][0]
                        else:
                            if self.training:
                                t, h, w = images[batch_idx]['video_grid_thw'][0]
                            else:
                                if len(images[batch_idx]['video_grid_thw'].shape) == 3:
                                    t = images[batch_idx]['video_grid_thw'][0, 0, 0].item()
                                    h = images[batch_idx]['video_grid_thw'][0, 0, 1].item()
                                    w = images[batch_idx]['video_grid_thw'][0, 0, 2].item()
                                elif len(images[batch_idx]['video_grid_thw'].shape) == 2:
                                    t = images[batch_idx]['video_grid_thw'][0, 0].item()
                                    h = images[batch_idx]['video_grid_thw'][0, 1].item()
                                    w = images[batch_idx]['video_grid_thw'][0, 2].item()
                                else:
                                    t, h, w = images[batch_idx]['video_grid_thw'][0]

                        h = h // 2
                        w = w // 2
                        t_index = torch.arange(t, device=cur_device, dtype=torch.long).view(-1, 1).expand(-1, h * w).flatten()
                        h_index = torch.arange(h, device=cur_device, dtype=torch.long).view(1, -1, 1).expand(t, -1, w).flatten()
                        w_index = torch.arange(w, device=cur_device, dtype=torch.long).view(1, 1, -1).expand(t, h, -1).flatten()
                        cur_position_ids.append(torch.stack([t_index, h_index, w_index]) + max_pos_id)
                        
                        max_pos_id = cur_position_ids[-1].max() + 1

                    elif cur_input_ids[mm_token_indices[i+1]] == SPEECH_TOKEN_INDEX:
                        cur_speech_features = speech_features[cur_sp_idx]
                        cur_new_input_embeds.append(cur_speech_features)
                        cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_position_ids.append(torch.arange(0, cur_speech_features.shape[0], dtype=position_ids.dtype, device=position_ids.device).repeat(3, 1) + max_pos_id)
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_sp_idx += 1
            
            if IMAGE_TOKEN_INDEX not in cur_input_ids: cur_im_idx += 1
            if SPEECH_TOKEN_INDEX not in cur_input_ids: cur_sp_idx += 1
            
            cur_new_input_embeds = [x.to(device=cur_input_embeds.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_position_ids = torch.cat(cur_position_ids, dim=-1)
            
            rope_delta = cur_position_ids.max() + 1 - len(cur_new_input_embeds)
            rope_deltas.append(rope_delta)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_position_ids.append(cur_position_ids)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)

        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_position_ids = [x[..., :tokenizer_model_max_length] for x in new_position_ids]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((3, batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_position_ids) in enumerate(zip(new_input_embeds, new_labels, new_position_ids)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[:, i, -cur_len:] = cur_position_ids
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[:, i, :cur_len] = cur_position_ids
                    
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, loss_align, rope_deltas


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
                
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
