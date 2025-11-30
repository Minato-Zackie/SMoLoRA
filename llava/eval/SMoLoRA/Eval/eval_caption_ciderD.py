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
            gts[question_id] = annotations_dict[question_id]['answer']  # 多个参考答案的列表

    # 初始化 cider scorer
    cider_scorer = Cider()

    # 初始化最终的参考答案字典
    final_gts = {}

    # 遍历模型预测的结果
    for question_id, res_text in res.items():
        if question_id in gts:
            temp_gts = gts[question_id]  # 对应的五个参考答案

            # 扩展 temp_res 成为五条相同的内容
            temp_res = res_text * len(temp_gts)

            # 使用不同的id（1, 2, 3, 4, 5）与 gts 进行计算
            temp_res_dict = {str(i + 1): [temp_res[i]] for i in range(len(temp_res))}
            temp_gts_dict = {str(i + 1): [temp_gts[i]] for i in range(len(temp_gts))}
            
            # 计算 CIDEr 分数，将每个 temp_res 与对应的 gts 进行计算
            score, scores = cider_scorer.compute_score(temp_gts_dict, temp_res_dict)

            # 从 scores 数组中选取最高分的索引
            best_gts_index = np.argmax(scores)  # 找到分数最高的参考答案的索引
            
            # 获取对应的参考答案
            best_gts = temp_gts[best_gts_index]

            # 将分数最高的 gts 加入 final_gts
            final_gts[question_id] = [best_gts]

    # 计算最终的 CIDEr 分数
    score, scores = cider_scorer.compute_score(res, final_gts)

    # 对分数进行乘以100
    score *= 100
    scores = [s * 100 for s in scores]

    # 输出总的 CIDEr 分数并加上百分号
    print(f"Avg CIDEr Score: {score:.2f}%")

    # 将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'cider_result.txt')
        with open(output_file, 'w') as f:
            f.write(f"Avg CIDEr Score: {score:.2f}%\n")
            for question_id, cider_score in zip(gts.keys(), scores):
                f.write(f"Question ID: {question_id}, CIDEr Score: {cider_score:.2f}%\n")

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_cider(args.annotation_file, args.result_file)
