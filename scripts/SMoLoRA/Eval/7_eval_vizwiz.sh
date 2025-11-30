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

RESULT_DIR="./results/SMoLoRA/VizWiz_single"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.SMoLoRA.MET.model_vizwiz \
        --model-path $MODELPATH \
        --model-base /data/zqwang/moe_cl_data/checkpoint/vicuna-7b-v1.5 \
        --ins-type 1 \
        --question-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/VizWiz/val.json \
        --image-folder /data/zqwang/moe_cl_data/dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --mm_pretrain /data/zqwang/moe_cl_data/checkpoint/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin &

done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/SMoLoRA/Eval/eval_vizwiz.py \
    --result-file $output_file \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/VizWiz/val.json \
    --output-dir $RESULT_DIR/$STAGE \