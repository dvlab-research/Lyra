#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial

import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms

from .imagebind.data import get_clip_timepoints, waveform2melspec
from .imagebind.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize, SelectElement)
from .imagebind.multimodal_preprocessors import (AudioPreprocessor,
                                             PatchEmbedGeneric,
                                             SpatioTemporalPosEmbeddingHelper)
from .imagebind.transformer import MultiheadAttention, SimpleTransformer



from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

from typing import Optional, Union
from transformers.image_processing_utils import BatchFeature
from transformers.utils import TensorType
import logging

import pdb



class ImageBindAudioModel(nn.Module):
    def __init__(
        self,
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=1024,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_num_mel_bins=128,
        audio_target_len=204,
        audio_drop_path=0.1,
    ):
        super().__init__()

        self.audio_preprocessors = self._create_audio_preprocessors(
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
        )

        self.audio_trunks = self._create_audio_trunks(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
        )

        self.audio_heads = self._create_audio_heads(
            out_embed_dim,
            audio_embed_dim,
        )

        self.audio_postprocessors = self._create_audio_postprocessors()

    def _create_audio_preprocessors(
        self,
        audio_embed_dim=768,
        audio_kernel_size=16,
        audio_stride=10,
        audio_num_mel_bins=128,
        audio_target_len=204,
    ):

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )


        return audio_preprocessor

    def _create_audio_trunks(
        self,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        return instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )


    def _create_audio_heads(
        self,
        out_embed_dim,
        audio_embed_dim,
    ):
        
        return nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )


    def _create_audio_postprocessors(self):
        
        return nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        

    def forward(self, inputs):
        reduce_list = (
            inputs.ndim >= 5
        )  # Audio and Video inputs consist of multiple clips
        if reduce_list:
            B, S = inputs.shape[:2]
            inputs = inputs.reshape(
                B * S, *inputs.shape[2:]
            )
        # pdb.set_trace()
        if inputs is not None:
            inputs = self.audio_preprocessors(inputs)
            trunk_inputs = inputs["trunk"]
            head_inputs = inputs["head"]
            inputs = self.audio_trunks(**trunk_inputs)
            inputs = self.audio_heads(inputs, **head_inputs)

            inputs = self.audio_postprocessors(inputs)
            
            if reduce_list:
                inputs = inputs.reshape(B, S, -1)
                inputs = inputs.squeeze(0)
                # inputs = inputs.mean(dim=1)
            outputs = inputs

        return outputs


class AudioProcessor(nn.Module):
    def __init__(self, num_mel_bins=128, target_length=204, sample_rate=16000, 
                 clip_duration=2, clips_per_video=3, audio_mean=-4.268, audio_std=9.138, **kwargs) -> None:
        super().__init__()

        self.num_mel_bins=num_mel_bins,
        self.target_length=target_length,
        self.sample_rate=sample_rate,
        self.clip_duration=clip_duration,
        self.clips_per_video=clips_per_video,
        self.audio_mean=audio_mean,
        self.audio_std=audio_std,

        self.clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
    
    def __call__(self, audios, **kwargs) -> BatchFeature:
        """Preprocess an audio or a batch of audios."""
        return self.preprocess(audios, **kwargs)
    
    def preprocess(self, 
        audios,
        device='cpu',
        num_mel_bins=128,
        target_length=204,
        sample_rate=16000,
        clip_duration=2,
        clips_per_video=3,
        audio_mean=-4.268,
        audio_std=9.138,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        num_mel_bins = num_mel_bins if num_mel_bins is not None else self.num_mel_bins
        target_length = target_length if target_length is not None else self.target_length
        sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        clip_duration = clip_duration if clip_duration is not None else self.clip_duration
        clips_per_video = clips_per_video if clips_per_video is not None else self.clips_per_video
        audio_mean = audio_mean if audio_mean is not None else self.audio_mean
        audio_std = audio_std if audio_std is not None else self.audio_std

        if audios is None:
            return None

        audio_outputs = []

        if not isinstance(audios, list):
            audios = [audios]

        for audio_path in audios:
            # waveform, sr = librosa.load(audio_path, sample_rate=16000)
            waveform, sr = torchaudio.load(audio_path)
            if sample_rate != sr:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=sample_rate
                )
            all_clips_timepoints = get_clip_timepoints(
                self.clip_sampler, waveform.size(1) / sample_rate
            )
            all_clips = []
            for clip_timepoints in all_clips_timepoints:
                waveform_clip = waveform[
                    :,
                    int(clip_timepoints[0] * sample_rate): int(
                        clip_timepoints[1] * sample_rate
                    ),
                ]
                waveform_melspec = waveform2melspec(
                    waveform_clip, sample_rate, num_mel_bins, target_length
                )
                all_clips.append(waveform_melspec)

            normalize = transforms.Normalize(mean=audio_mean, std=audio_std)
            all_clips = [normalize(ac).to(device) for ac in all_clips]

            all_clips = torch.stack(all_clips, dim=0)
            audio_outputs.append(all_clips)
        
        data =torch.stack(audio_outputs, dim=0)
        return data

def imagebind_huge(pretrained=False):
    model = ImageBindAudioModel(
        out_embed_dim=1024,
        audio_drop_path=0.1,
    )

    if pretrained:
        if not os.path.exists("/dataset-vlm/vyuqiliu/model_zoo/Audio/ImageBind/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to /dataset-vlm/vyuqiliu/model_zoo/Audio/ImageBind/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                "/dataset-vlm/vyuqiliu/model_zoo/Audio/ImageBind/imagebind_huge.pth",
                progress=True,
            )

        model.load_state_dict(torch.load("/dataset-vlm/vyuqiliu/model_zoo/Audio/ImageBind/imagebind_huge.pth"))

    return model



class ImageBindAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.is_optimize = getattr(args, 'optimize_audio_tower', False)
        
        if not delay_load:
            self.load_model()

    def load_model(self):
        self.audio_processor = AudioProcessor()
        self.audio_tower = ImageBindAudioModel(out_embed_dim=1024, audio_drop_path=0.1)
        
        for name, param in self.audio_processor.named_parameters():
            print(f"参数名: {name}")
            print(f"参数值: {param.data}")
            print(f"是否需要梯度计算: {param.requires_grad}")
        # pdb.set_trace()
        # aaaa = torch.load(self.audio_tower_name)
        
        self.audio_tower.load_state_dict(torch.load(self.audio_tower_name))
        # self.audio_tower.requires_grad_(False)
        self.is_loaded = True


    # @torch.no_grad()
    def forward(self, audios):
        if type(audios) is list:
            audio_features = []
            for audio in audios:
                audio_embeds = self.audio_tower(audio)
                audio_features.append(audio_embeds)
        else:
            audio_features = self.audio_tower(audios)
        return audio_features

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

