# Upstream tasks

sh ./scripts/SMoLoRA/Train_Eval/1_Science.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/1_eval_sqa.sh ScienceQA_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/ScienceQA/llava-1.5-7b-lora ./results/SMoLoRA/ScienceQA_single
sh ./scripts/SMoLoRA/Train_Eval/2_TextVQA.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/2_eval_textqa.sh TextVQA_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/TextVQA/llava-1.5-7b-lora ./results/SMoLoRA/TextVQA_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/1_eval_sqa.sh TextVQA_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/TextVQA/llava-1.5-7b-lora ./results/SMoLoRA/ScienceQA_single
sh ./scripts/SMoLoRA/Train_Eval/3_Flickr30k.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/3_eval_flickr.sh Flickr30k_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/Flickr30k/llava-1.5-7b-lora ./results/SMoLoRA/Flickr30k_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/2_eval_textqa.sh Flickr30k_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/Flickr30k/llava-1.5-7b-lora ./results/SMoLoRA/TextVQA_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/1_eval_sqa.sh Flickr30k_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/Flickr30k/llava-1.5-7b-lora ./results/SMoLoRA/ScienceQA_single
sh ./scripts/SMoLoRA/Train_Eval/4_ImageNet.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/4_eval_imagenet.sh ImageNet_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/ImageNet/llava-1.5-7b-lora ./results/SMoLoRA/ImageNet_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/3_eval_flickr.sh ImageNet_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/ImageNet/llava-1.5-7b-lora ./results/SMoLoRA/Flickr30k_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/2_eval_textqa.sh ImageNet_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/ImageNet/llava-1.5-7b-lora ./results/SMoLoRA/TextVQA_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/1_eval_sqa.sh ImageNet_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/ImageNet/llava-1.5-7b-lora ./results/SMoLoRA/ScienceQA_single
sh ./scripts/SMoLoRA/Train_Eval/5_GQA.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/5_eval_gqa.sh GQA_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/GQA/llava-1.5-7b-lora ./results/SMoLoRA/GQA_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/4_eval_imagenet.sh GQA_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/GQA/llava-1.5-7b-lora ./results/SMoLoRA/ImageNet_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/3_eval_flickr.sh GQA_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/GQA/llava-1.5-7b-lora ./results/SMoLoRA/Flickr30k_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/2_eval_textqa.sh GQA_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/GQA/llava-1.5-7b-lora ./results/SMoLoRA/TextVQA_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/1_eval_sqa.sh GQA_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/GQA/llava-1.5-7b-lora ./results/SMoLoRA/ScienceQA_single
sh ./scripts/SMoLoRA/Train_Eval/6_vqav2.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/5_eval_gqa.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/GQA_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/4_eval_imagenet.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/ImageNet_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/3_eval_flickr.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/Flickr30k_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/2_eval_textqa.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/TextVQA_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/1_eval_sqa.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/ScienceQA_single
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/6_eval_vqav2.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/VQAv2_single


# Downstream tasks

sh ./scripts/SMoLoRA/Train_Eval/7_Place365_5shot.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/10_eval_place365.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/Place365_5shot/llava-1.5-7b-lora ./results/SMoLoRA/Places365_single_5shot
sh ./scripts/SMoLoRA/Train_Eval/7_Place365_10shot.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/10_eval_place365.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/Place365_10shot/llava-1.5-7b-lora ./results/SMoLoRA/Places365_single_10shot


CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/7_eval_vizwiz.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/VizWiz_single_10shot
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/8_eval_textcaps.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/Textcaps_single_10shot
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./scripts/SMoLoRA/Eval/9_eval_ocrvqa.sh VQAv2_smolora_single /data/zqwang/moe_cl_data/checkpoint/SMoLoRA/Instruction_single_type/VQAv2/llava-1.5-7b-lora ./results/SMoLoRA/OCRVQA_single_10shot



