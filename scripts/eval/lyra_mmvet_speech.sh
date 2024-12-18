#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


CKPT="Lyra_Base_9B"
LORA_PATH="Lyra_Base_9B/speech_lora"

echo $CKPT
echo $LORA_PATH

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m lyra.eval.model_lyra_image_speech \
    --model-path work_dirs/$CKPT \
    --model-lora-path work_dirs/$LORA_PATH \
    --question-file data/Lyra_Eval/MM_vet_speech/lyra_mm_vet.jsonl \
    --image-folder data/Lyra_Eval/MM_vet_speech/images \
    --speech-folder data/Lyra_Eval/MM_vet_speech/lyra_audios \
    --answers-file work_dirs/MM_vet_speech/$CKPT/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --use_flash_attn True \
    --conv-mode qwen2vl &
done

wait

output_file=work_dirs/MM_vet_speech/$CKPT/merge.jsonl
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat work_dirs/MM_vet_speech/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_mmvet_for_eval.py \
    --src $output_file \
    --dst work_dirs/MM_vet_speech/$CKPT/$CKPT.json

echo $CKPT
echo $LORA_PATH
