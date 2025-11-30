import os
import numpy as np
import argparse
import json
from cider import Cider  # 使用 CIDEr 计算器

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./cl_dataset/annotations.json')  # 验证集的真实标签文件
    parser.add_argument('--result-file', type=str, default='./results/predictions.jsonl')  # 模型预测结果文件
    parser.add_argument('--output-dir', type=str)  # 输出结果的目录
    return parser.parse_args()

def eval_cider(annotation_file, result_file):
    # 读取 annotation-file (真实标签)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # 将 annotations 列表转换为字典，方便根据 question_id 查找
    annotations_dict = {str(item['question_id']): item for item in annotations}

    # 读取 result-file (模型预测结果)
    results = [json.loads(line) for line in open(result_file)]

    # 准备 CIDEr 评估的输入数据
    res = {}  # 模型生成的文本，修改为字典形式
    gts = {}  # 参考答案的字典

    for result in results:
        question_id = str(result['question_id'])
        if question_id in annotations_dict:
            # 构建 res 的结构，符合 cider.py 的要求
            res[question_id] = [result['text']]  # 使用 question_id 作为唯一标识，预测结果放入列表中

            # 将所有参考答案作为一组放入 gts 中
            # 这里确保 gts 的值是一个列表，即使只有一个参考答案
            answer = annotations_dict[question_id]['answer']
            gts[question_id] = [answer] if isinstance(answer, str) else answer  # 确保参考答案是列表


    # 计算 CIDEr 分数
    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(gts, res)
    # 对分数进行乘以100
    score *= 100
    scores = [s * 100 for s in scores]

    # 输出总的 CIDEr 分数并加上百分号
    print(f"总 CIDEr 分数: {score:.2f}%")

    # 将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'cider_result.txt')
        with open(output_file, 'w') as f:
            f.write(f"总 CIDEr 分数: {score:.2f}%\n")
            for question_id, cider_score in zip(gts.keys(), scores):
                f.write(f"Question ID: {question_id}, CIDEr Score: {cider_score:.2f}%\n")

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_cider(args.annotation_file, args.result_file)