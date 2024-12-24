#!/bin/bash
file=${0##*/}
file=${file%.sh}
echo $file

FINETUNE_NAME=$file
PRETRAIN_AUDIO_NAME=Stage1_Lyra_Base_qwen2vl_9B_Pretrain

deepspeed --hostfile hostfile/hostfile_4 \
    lyra/train/train.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path model_zoo/LLM/Qwen2VL_7B_LLM \
    --version qwen2vl \
    --data_path data/Lyra_SFT/multi_modality_speech/lyra_multimodal.json \
    --image_folder data/Lyra_SFT/multi_modality_image/ \
    --vision_tower model_zoo/vision/Qwen2VL_7B_ViT \
    --mm_projector_type identity \
    --speech_folder data/Lyra_SFT/multi_modality_speech \
    --speech_tower model_zoo/audio/whisper-large-v3 \
    --pretrain_mm_speech_mlp_adapter work_dirs/$PRETRAIN_AUDIO_NAME/mm_speech_projector.bin \
    --mm_speech_projector_type simple_mlp \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio qwen2vl \
    --speech_norm False \
    --speech_learn True \
    --align_norm False \
    --train_modality 'text_image_speech' \
    --asr_align True \
    --weight_lambda 0.5 \
    --align_type 'dtw' \
    --align_temperature 0.06 \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir work_dirs/$FINETUNE_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --lora_enable True \
    --freeze_mm_mlp_adapter True
