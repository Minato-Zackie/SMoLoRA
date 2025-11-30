################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################


################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

deepspeed --include localhost:4,5,6,7 --master_port 29600 llava/train/train_mem_SMoLoRA.py \
    --deepspeed ./scripts/zero3_offload.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --expert_num 8 \
    --ins_type 3 \
    --ins_emb playground/ins_emb_single.pkl \
    --model_name_or_path /data_ssd/zqwang/model/vicuna-7b-v1.5 \
    --pretrain_mm_mlp_adapter /data/zqwang/moe_cl_data/checkpoint/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --previous_task_model_path /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/Flickr30k/llava-1.5-7b-lora \
    --version $PROMPT_VERSION \
    --data_path /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/ImageNet/train.json \
    --image_folder /data/zqwang/moe_cl_data/dataset \
    --vision_tower /data/zqwang/moe_cl_data/checkpoint/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/ImageNet/llava-1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit  1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none