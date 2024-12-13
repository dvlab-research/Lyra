import torch
import torch.nn as nn

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLVisionConfig
from transformers.models.qwen2_vl import Qwen2VLImageProcessor


class Qwen2VLVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.is_optimize = getattr(args, 'optimize_vision_tower', False)
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = Qwen2VLVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = Qwen2VisionTransformerPretrainedModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True


    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                if hasattr(image, 'pixel_values'):
                    image_embeds = self.vision_tower(image['pixel_values'].to(device=self.vision_tower.device), grid_thw=image['image_grid_thw'].to(device=self.vision_tower.device))
                else:
                    image_embeds = self.vision_tower(image['pixel_values_videos'].to(device=self.vision_tower.device), grid_thw=image['video_grid_thw'].to(device=self.vision_tower.device))
                image_features.append(image_embeds)
        else:
            image_features = []
            if len(images['image_grid_thw'].shape) == 3:
                grid_thw_ = images['image_grid_thw'][0]
            elif len(images['image_grid_thw'].shape) == 2:
                grid_thw_ = images['image_grid_thw']
            image_embeds = self.vision_tower(images['pixel_values'].to(dtype=self.vision_tower.dtype, device=self.vision_tower.device, non_blocking=True).squeeze(), grid_thw=grid_thw_)
            image_features.append(image_embeds)
        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
    