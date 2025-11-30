import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='./cl_dataset/ScienceQA')
    parser.add_argument('--result-file', type=str, default='./results/CoIN/Qwen/ScienceQA/Finetune/merge.jsonl')
    parser.add_argument('--output-file', type=str, default='./results/CoIN/Qwen/ScienceQA/Finetune/output.jsonl')
    parser.add_argument('--annotation-file', type=str, default='./annotations.json')  # 添加参数来读取正确答案
    return parser.parse_args()

def count_words(sentence):
    """计算句子中的单词数量"""
    return len(sentence.split())

if __name__ == "__main__":
    args = get_args()

    # 读取预测文件和注释文件
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}

    annotations = json.load(open(args.annotation_file))  # 读取答案注释文件
    answers = {item['question_id']: item['answer'] for item in annotations}

    results = {'positive': [], 'negative': []}

    for pred in predictions.values():
        question_id = pred['question_id']
        pred_text = pred['text'].strip()  # 去除首尾空格

        # 计算单词数
        word_count = count_words(pred_text)

        # 判断正负样本
        if word_count <= 2:
            results['positive'].append({
                'question_id': question_id,
                'pred_text': pred_text,
                'valid': True
            })
        else:
            # 检查是否是正确答案
            correct_answer = answers.get(question_id, "")
            if pred_text.lower() == correct_answer.lower():  # 进行不区分大小写的比较
                results['positive'].append({
                    'question_id': question_id,
                    'pred_text': pred_text,
                    'valid': True
                })
            else:
                results['negative'].append({
                    'question_id': question_id,
                    'pred_text': pred_text,
                    'valid': False
                })

    # 计算准确率
    correct = len(results['positive'])
    total = correct + len(results['negative'])
    accuracy = correct / total * 100 if total > 0 else 0

    # 保存结果到输出文件
    with open(args.output_file, 'w') as f:
        # 写入准确率在顶部
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        json.dump(results, f, indent=2)

    # 打印摘要
    print(f"Total Positive Samples: {len(results['positive'])}, Total Negative Samples: {len(results['negative'])}")
    print(f"Accuracy: {accuracy:.2f}%")
