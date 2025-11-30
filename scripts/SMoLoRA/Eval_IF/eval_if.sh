

python llava/eval/SMoLoRA/Eval/eval_scienceqa_instruction.py \
    --result-file ./results/SMoLoRA/ScienceQA_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/ScienceQA_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_vqa_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/TextVQA/val.json \
    --result-file ./results/SMoLoRA/TextVQA_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/TextVQA_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_caption_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/Flickr30k/val.json \
    --result-file ./results/SMoLoRA/Flickr30k_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/Flickr30k_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_vqa_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/ImageNet/test.json \
    --result-file ./results/SMoLoRA/ImageNet_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/ImageNet_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_vqa_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/GQA/test.json \
    --result-file ./results/SMoLoRA/GQA_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/GQA_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_vqa_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/VQAv2/val.json \
    --result-file ./results/SMoLoRA/VQAv2_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/VQAv2_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_vqa_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/VizWiz/val.json \
    --result-file ./results/SMoLoRA/VizWiz_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/VizWiz_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_caption_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/TextCaps/val.json \
    --result-file ./results/SMoLoRA/TextCaps_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/TextCaps_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_vqa_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/OCRVQA/test.json \
    --result-file ./results/SMoLoRA/OCRVQA_single/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/OCRVQA_single/VQAv2_smolora_single/eval_if_output.jsonl \

python llava/eval/SMoLoRA/Eval/eval_vqa_instruction.py \
    --annotation-file /data/zqwang/moe_cl_data/CVIT_benchmark/Instructions_Single/Places365/val.json \
    --result-file ./results/SMoLoRA/Places365_single_5shot/VQAv2_smolora_single/merge.jsonl \
    --output-file ./results/SMoLoRA/Places365_single_5shot/VQAv2_smolora_single/eval_if_output.jsonl \




