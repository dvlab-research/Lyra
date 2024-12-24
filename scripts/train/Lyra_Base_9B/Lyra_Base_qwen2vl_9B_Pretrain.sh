#!/bin/bash
file=${0##*/}
file=${file%.sh}
echo $file

PRETRAIN_AUDIO_NAME=$file


deepspeed --hostfile hostfile/hostfile_4 \
    lyra/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path model_zoo/LLM/Qwen2VL_7B_LLM \
    --version qwen2vl \
    --data_path data/Lyra_Pretrain/lyra_pretrain.json \
    --speech_folder data/Lyra_Pretrain \
    --speech_tower model_zoo/audio/whisper-large-v3 \
    --mm_speech_projector_type simple_mlp \
    --speech_norm False \
    --speech_learn False \
    --asr_align False \
    --weight_lambda 0.0 \
    --tune_mm_speech_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio qwen2vl \
    --train_modality 'text_speech' \
    --bf16 True \
    --output_dir work_dirs/$PRETRAIN_AUDIO_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none