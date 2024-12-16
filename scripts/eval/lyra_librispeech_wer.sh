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
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m lyra.eval.model_lyra_text_speech \
    --model-path work_dirs/$CKPT \
    --model-lora-path work_dirs/$LORA_PATH \
    --speech-test-file data/Lyra_Eval/LibriSpeech/test-clean.jsonl \
    --speech-folder data/Lyra_Eval/LibriSpeech \
    --answers-file work_dirs/LibriSpeech/$CKPT/${CHUNKS}_${IDX}.jsonl \
    --temperature 0 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode qwen2vl & 
done

wait


output_file=work_dirs/LibriSpeech/$CKPT/merge.jsonl
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat work_dirs/LibriSpeech/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python -m lyra.eval.eval_asr_wer \
    --result-file $output_file

echo $CKPT
echo $LORA_PATH