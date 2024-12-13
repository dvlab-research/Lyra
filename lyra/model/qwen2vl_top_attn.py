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
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from einops import rearrange, repeat
from dataclasses import dataclass

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import repeat_kv, apply_multimodal_rotary_pos_emb
from transformers.utils import is_flash_attn_2_available

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers.utils import logging
logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


def forward_attn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    text_labels: Optional[torch.BoolTensor] = None,
    context_labels: Optional[torch.BoolTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")
    
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None
    
    if output_attentions:
        with torch.no_grad():
            attn_weights = []
            for query_state, key_state, context_label, text_label in zip(query_states, key_states, context_labels, text_labels):
                text_state = query_state[text_label].transpose(0, 1).to(dtype=torch.float32)
                vis_state = key_state.transpose(0, 1).to(dtype=torch.float32)
                attn_weight = torch.matmul(text_state, vis_state.transpose(1, 2)) / math.sqrt(self.head_dim)
                attn_weight = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weight = attn_weight[..., context_label]
                attn_weights.append(attn_weight.mean(dim=0).mean(dim=0))
    else:
        attn_weights = None
    
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


def forward_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    context_labels: Optional[torch.BoolTensor] = None,
    text_labels: Optional[torch.BoolTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    kwargs['context_labels'] = context_labels
    kwargs['text_labels'] = text_labels

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


@dataclass
class CompressModelOutputWithPast(BaseModelOutputWithPast):
    labels: Optional[Tuple[torch.LongTensor, ...]] = None


def forward_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    text_labels: Optional[torch.BoolTensor] = None,
    context_labels: Optional[torch.BoolTensor] = None,
    labels: Optional[torch.BoolTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CompressModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    # all_self_attns = () if output_attentions else None
    all_self_attns = ()
    next_decoder_cache = None

    # set compression
    compress_gap = self.config.compress_gap
    keep_rate = self.config.keep_rate
    num_layers = len(self.layers)

    for i, decoder_layer in enumerate(self.layers):
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # set output_attentions
        compress_tokens_layer = ((context_labels is not None) and (i+1) % compress_gap == 0 and keep_rate < 1 and i+1 < num_layers)
        output_attentions_layer = (output_attentions | compress_tokens_layer)

        # change attention_mask and position_ids at inference
        if causal_mask is not None and not self.training and len(past_key_values[i][0]) > 0:
            length = past_key_values[i][0].shape[2] + hidden_states.shape[1]
            causal_mask = causal_mask[:, :length]

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions_layer,
                use_cache,
                cache_position,
                position_embeddings,
                context_labels,
                text_labels
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions_layer,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                context_labels=context_labels,
                text_labels=text_labels
            )

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions_layer else 1]

        if output_attentions_layer:
            all_self_attns += (layer_outputs[1],)

        # compress visual token
        if context_labels is not None and output_attentions_layer:
            attn_weights = layer_outputs[1]
            hidden_states, context_labels, text_labels, labels, causal_mask, position_ids, position_embeddings = \
                self.merge_visual_tokens(hidden_states, context_labels, text_labels, labels, \
                                         attn_weights, causal_mask, position_ids, position_embeddings, keep_rate)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    return CompressModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        labels=labels,
    )


def merge_visual_tokens(self, hidden_states, context_labels, text_labels, labels, attn_weights, attention_mask, position_ids, position_embeddings, keep_rate=0.5):
    batch_size = hidden_states.shape[0]
    new_hidden_states = []
    new_context_labels = []
    new_text_labels = []
    new_labels = []
    new_position_ids = []
    new_attention_mask = []
    new_cos = []
    new_sin = []
    
    cos, sin = position_embeddings

    for i in range(batch_size):
        vis_count = context_labels[i].sum()
        attn_weight = attn_weights[i].to(dtype=torch.float32)

        select_tokens = (vis_count * keep_rate).clamp(min=32).long()
        
        if vis_count < 32:
            select_tokens = vis_count

        topk_indices = attn_weight.topk(k=select_tokens, dim=-1).indices
        select_idxs = torch.zeros_like(attn_weight, dtype=torch.bool)
        select_idxs[topk_indices] = True
        keep_idxs = torch.ones_like(context_labels[i], dtype=torch.bool)
        keep_idxs[context_labels[i]] = select_idxs

        new_hidden_states.append(hidden_states[i, keep_idxs])
        new_context_labels.append(context_labels[i, keep_idxs])
        new_text_labels.append(text_labels[i, keep_idxs])
        new_position_ids.append(position_ids[:, i, keep_idxs])
        new_cos.append(cos[:, i, keep_idxs])
        new_sin.append(sin[:, i, keep_idxs])
        if labels is not None:
            new_labels.append(labels[i, keep_idxs])

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                new_attention_mask.append(attention_mask[i, keep_idxs])
            else:
                cur_attention_mask = attention_mask[i, :, keep_idxs]
                cur_attention_mask = cur_attention_mask[:, :, keep_idxs]
                new_attention_mask.append(cur_attention_mask)
    
    if attention_mask is not None:
        max_length = max([attn_mask.sum() for attn_mask in new_attention_mask])
    else:
        max_length = max(len(x) for x in new_context_labels)
    for i in range(batch_size):
        pad_length = max_length - len(new_context_labels[i])
        if pad_length > 0:
            new_hidden_states[i] = F.pad(new_hidden_states[i], (0, 0, 0, pad_length))
            new_context_labels[i] = F.pad(new_context_labels[i], (0, pad_length))
            new_text_labels[i] = F.pad(new_text_labels[i], (0, pad_length))
            if labels is not None:
                new_labels[i] = F.pad(new_labels[i], (0, pad_length), value=-100)
            if attention_mask is not None:
                new_attention_mask[i] = F.pad(new_attention_mask[i], (0, pad_length))
            new_position_ids[i] = F.pad(new_position_ids[i], (0, pad_length))
            new_cos[i] = F.pad(new_cos[i], (0, 0, 0, pad_length))
            new_sin[i] = F.pad(new_sin[i], (0, 0, 0, pad_length))
        elif pad_length < 0:
            new_hidden_states[i] = new_hidden_states[i][:max_length]
            new_context_labels[i] = new_context_labels[i][:max_length]
            new_text_labels[i] = new_text_labels[i][:max_length]
            if labels is not None:
                new_labels[i] = new_labels[i][:max_length]
            if attention_mask is not None:
                new_attention_mask[i] = new_attention_mask[i][:max_length]
            new_position_ids[i] = new_position_ids[i][:, :max_length]
            new_cos[i] = new_cos[i][:, :max_length]
            new_sin[i] = new_sin[i][:, :max_length]

    new_hidden_states = torch.stack(new_hidden_states)
    new_context_labels = torch.stack(new_context_labels)
    new_text_labels = torch.stack(new_text_labels)
    if labels is not None:
        new_labels = torch.stack(new_labels)
    else:
        new_labels = None
    if attention_mask is not None:
        new_attention_mask = torch.stack(new_attention_mask)
    else:
        new_attention_mask = None
    new_position_ids = torch.stack(new_position_ids, dim=1)
    new_cos = torch.stack(new_cos, dim=1)
    new_sin = torch.stack(new_sin, dim=1)
    new_position_embeddings = (new_cos, new_sin)

    return new_hidden_states, new_context_labels, new_text_labels, new_labels, new_attention_mask, new_position_ids, new_position_embeddings


def replace_qwen2vl_attn_with_top_attn():
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = forward_model
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLDecoderLayer.forward = forward_layer
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.merge_visual_tokens = merge_visual_tokens
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = forward_attn

