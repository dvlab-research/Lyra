#!/bin/bash
file=${0##*/}
file=${file%.sh}
echo $file

FINETUNE_NAME=$file

deepspeed --hostfile ./hostfile/hostfile_22 \
    lyra/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /dataset-v2/zszhong/checkpoints/Audio_Lyra_125_qwen2vl_7b_stage2_english_asr_lora_moredata_datarefine3_wholemodel \
    --version qwen2vl \
    --data_path /dataset-vlm/sq/audio/generate_data/Cambrian_puretext_no_speech_noexceed3000.json \
    --image_folder /dataset-vlm/vyuqiliu/data/Dataset/MGM-Finetune \
    --vision_tower /dataset-v2/pretrained-models/Qwen2vl_vit \
    --mm_projector_type identity \
    --speech_folder /dataset-vlm/sq/audio/generate_data/tmp/MGM_Instruction_Clear \
    --speech_tower /dataset-vlm/vyuqiliu/model_zoo/Audio/whisper-large-v3 \
    --mm_speech_projector_type simple_mlp \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio qwen2vl \
    --speech_norm False \
    --speech_learn False \
    --align_norm False \
    --asr_align False \
    --weight_lambda 0.0 \
    --align_type 'dtw' \
    --align_temperature 0.06 \
    --generate True \
    --tune_speech_generator_only True \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /dataset-v2/zszhong/checkpoints/$FINETUNE_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.09 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to none \
    --freeze_mm_mlp_adapter True \
    --freeze_mm_speech_mlp_adapter True \