# sequential-finetune test
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/1_eval_sqa.sh ScienceQA_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/ScienceQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/2_eval_textqa.sh TextVQA_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/TextVQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/3_eval_flickr.sh Flickr30k_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/Flickr30k/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/4_eval_imagenet.sh ImageNet_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/ImageNet/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/5_eval_gqa.sh GQA_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/GQA/llava-1.5-7b-lora
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/6_eval_vqav2.sh VQAv2_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/VQAv2/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/1_eval_sqa.sh TextVQA_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/TextVQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/1_eval_sqa.sh Flickr30k_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/Flickr30k/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/1_eval_sqa.sh ImageNet_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/ImageNet/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/1_eval_sqa.sh GQA_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/GQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/1_eval_sqa.sh VQAv2_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/VQAv2/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/2_eval_textqa.sh Flickr30k_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/Flickr30k/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/2_eval_textqa.sh ImageNet_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/ImageNet/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/2_eval_textqa.sh GQA_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/GQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/2_eval_textqa.sh VQAv2_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/VQAv2/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/3_eval_flickr.sh ImageNet_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/ImageNet/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/3_eval_flickr.sh GQA_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/GQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/3_eval_flickr.sh VQAv2_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/VQAv2/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/4_eval_imagenet.sh GQA_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/GQA/llava-1.5-7b-lora
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/4_eval_imagenet.sh VQAv2_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/VQAv2/llava-1.5-7b-lora

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/CoIN/Eval_newbench_met/5_eval_gqa.sh VQAv2_MET /data/zqwang/moe_cl_data/checkpoint/Coin/Instruction_newbench_type1/MoeLoRA_expert8_top2/VQAv2/llava-1.5-7b-lora