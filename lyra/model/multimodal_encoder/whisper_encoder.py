import torch
import torch.nn as nn
import os

from transformers import WhisperModel, WhisperFeatureExtractor, WhisperConfig
import pdb

class WhisperTower(nn.Module):
    def __init__(self, speech_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.speech_tower_name = speech_tower
        self.is_optimize = getattr(args, 'optimize_speech_tower', False)
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_speech_tower', False):
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.speech_tower_name)

    def load_model(self):
        self.speech_processor = WhisperFeatureExtractor.from_pretrained(self.speech_tower_name)
        self.speech_tower = WhisperModel.from_pretrained(self.speech_tower_name).encoder
        self.speech_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, speeches):
        speech_features = self.speech_tower(speeches.to(device=self.speech_tower.device), return_dict=True).last_hidden_state
        return speech_features


    @property
    def dtype(self):
        return self.speech_tower.dtype

    @property
    def device(self):
        return self.speech_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.speech_tower.config
        else:
            return self.cfg_only