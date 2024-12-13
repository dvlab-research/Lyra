import os
from .clip_encoder import CLIPVisionTower
# from .eva_encoder import EVAVisionTower
# from .openclip_encoder import OpenCLIPVisionTower
from .siglip_encoder import SiglipVisionTower
from .qwen2vl_encoder import Qwen2VLVisionTower
from .whisper_encoder import WhisperTower
# from .imagebind_encoder import ImageBindAudioTower
from transformers import WhisperModel
import pdb

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', "../processor/clip-patch14-224"))
    if not os.path.exists(vision_tower):
        raise ValueError(f'Not find vision tower: {vision_tower}')
    # if "openai" in vision_tower.lower() or "ShareGPT4V" in vision_tower:
    #     return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "clip-vit" in vision_tower.lower() or "ShareGPT4V" in vision_tower or "InternViT" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower.lower():
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "qwen2vl" in vision_tower.lower():
        return Qwen2VLVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
    #     return EVAVisionTower(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')

    
def build_speech_tower(speech_tower_cfg, **kwargs):
    speech_tower = getattr(speech_tower_cfg, 'mm_speech_tower', getattr(speech_tower_cfg, 'speech_tower', None))

    if not os.path.exists(speech_tower):
        raise ValueError(f'Not find speech tower: {speech_tower}')
    
    if "whisper" in speech_tower.lower():
        # return WhisperModel.from_pretrained(speech_tower).encoder
        return WhisperTower(speech_tower, args=speech_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown speech tower: {speech_tower}')
    
def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))

    if not os.path.exists(audio_tower):
        raise ValueError(f'Not find audio tower: {audio_tower}')
    
    if "imagebind" in audio_tower.lower():
        # return WhisperModel.from_pretrained(audio_tower).encoder
        return ImageBindAudioTower(audio_tower, args=audio_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown audio tower: {audio_tower}')
    