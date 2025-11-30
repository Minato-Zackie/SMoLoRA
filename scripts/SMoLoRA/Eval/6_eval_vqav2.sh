#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./'
else
    MODELPATH=$2
fi

if [ ! -n "$3" ] ;then
    RESULT_DIR="./results/SMoLoRA/VQAv2_single"
else
    RESULT_DIR=$3
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.SMoLoRA.MET.model_vqa_cc_instruction \
        --model-path $MODELPATH \
        --model-base /data/zqwang/moe_cl_data/checkpoint/vicuna-7b-v1.5 \
        --ins-type 5 \
        --question-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/VQAv2/val.json \
        --image-folder /data/zqwang/moe_cl_data/dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/SMoLoRA/Eval/eval_vqav2.py \
    --result-file $output_file \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/VQAv2/val.json \
    --output-dir $RESULT_DIR/$STAGE \

