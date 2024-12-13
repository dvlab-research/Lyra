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
#    and LLaMA-Omni (https://github.com/ictnlp/LLaMA-Omni)
#    Copyright 2024 Zhisheng Zhong, Chengyao Wang
# ------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.generation.utils import GenerateOutput
from lyra.model.language_model.lyra_qwen2vl import LyraQwen2VLForCausalLM
from lyra.model.multimodal_generator.builder import build_speech_generator
from lyra.model.multimodal_generator.generation import GenerationWithCTC
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

@dataclass
class LyraCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

class LyraConfig(Qwen2VLConfig):
    model_type = "Lyra_Qwen2VL_SpeechGenerator"


class Lyra2SQwen2VLForCausalLM(LyraQwen2VLForCausalLM, GenerationWithCTC):
    config_class = LyraConfig
    def __init__(self, config):
        super().__init__(config)
        self.post_init()
        if hasattr(config, "speech_generator_type"):
            self.speech_generator = build_speech_generator(config)
    
    def initialize_speech_generator(self, model_args):
        self.config.speech_generator_type = getattr(model_args, 'speech_generator_type', 'ctc')
        self.config.ctc_decoder_config = getattr(model_args, 'ctc_decoder_config', '(4,4096,32,11008)')
        self.config.ctc_upsample_factor = getattr(model_args, 'ctc_upsample_factor', 25)
        self.config.ctc_loss_weight = getattr(model_args, 'ctc_loss_weight', 1.0)
        self.config.unit_vocab_size = getattr(model_args, 'unit_vocab_size', 1000)
        self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', False)
        if getattr(self, "speech_generator", None) is None:
            self.speech_generator = build_speech_generator(self.config)
    
    def get_speech_generator(self):
        speech_generator = getattr(self, 'speech_generator', None)
        if type(speech_generator) is list:
            speech_generator = speech_generator[0]
        return speech_generator
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        speeches: Optional[torch.FloatTensor] = None,
        speeches_asr: Optional[torch.FloatTensor] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            # Generator training should be text & speech
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                loss_align,
                rope_deltas
            ) = self.prepare_inputs_labels_for_text_image_speech_qwen2vl(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                speeches,
                speeches_asr,
                rope_deltas
            )

        if self.training:
            if self.tune_speech_generator_only:
                with torch.no_grad():
                    qwen2vl_output = super(LyraQwen2VLForCausalLM, self).forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=True,
                        return_dict=return_dict
                    )
                loss = self.speech_generator(qwen2vl_output['hidden_states'][-1], labels, tgt_units)
            else:
                qwen2vl_output = super(LyraQwen2VLForCausalLM, self).forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict
                )
                lm_loss = qwen2vl_output.loss
                ctc_loss = self.speech_generator(qwen2vl_output['hidden_states'][-1], labels, tgt_units)
                loss = lm_loss + ctc_loss * self.config.ctc_loss_weight
        else:
            qwen2vl_output = super(LyraQwen2VLForCausalLM, self).forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict
            )
            loss = qwen2vl_output.loss

        return LyraCausalLMOutputWithPast(
            loss=loss,
            logits=qwen2vl_output.logits,
            past_key_values=qwen2vl_output.past_key_values,
            hidden_states=qwen2vl_output.hidden_states,
            attentions=qwen2vl_output.attentions,
            rope_deltas=rope_deltas,
        )
        
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        speeches: Optional[torch.Tensor] = None,
        speeches_asr: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        streaming_unit_gen=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None or speeches is not None:
            # Generator training should be text & speech
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                rope_deltas
            ) = self.prepare_inputs_labels_for_text_image_speech_qwen2vl(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                speeches,
                speeches_asr,
                rope_deltas
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        outputs = GenerationWithCTC.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            streaming_unit_gen=streaming_unit_gen,
            rope_deltas=rope_deltas,
            **kwargs
        )

        hidden_states = outputs['hidden_states']
        hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)
        ctc_pred = self.speech_generator.predict(hidden_states.squeeze(0))

        return outputs.sequences, ctc_pred
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        speeches = kwargs.pop("speeches", None)
        speeches_asr = kwargs.pop("speeches_asr", None)
        rope_deltas = kwargs.pop("rope_deltas", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if speeches is not None:
            _inputs['speeches'] = speeches
        if speeches_asr is not None:
            _inputs['speeches_asr'] = speeches_asr
        if rope_deltas is not None:
            _inputs['rope_deltas'] = rope_deltas
        return _inputs

AutoConfig.register("Lyra_Qwen2VL_SpeechGenerator", LyraConfig)
AutoModelForCausalLM.register(LyraConfig, Lyra2SQwen2VLForCausalLM)