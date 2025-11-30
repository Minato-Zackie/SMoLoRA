import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='/data/zqwang/moe_cl_data/dataset/ScienceQA')
    parser.add_argument('--result-file', type=str, default='./results/CoIN/Qwen/ScienceQA/Finetune/merge.jsonl')
    parser.add_argument('--output-file', type=str, default='./results/CoIN/Qwen/ScienceQA/Finetune/output.jsonl')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=[chr(i) for i in range(65, 91)])  # A-Z
    return parser.parse_args()


def is_single_option(pred_text, options):
    """
    Check if the predicted text is a single option (e.g., "A", "B", "C", ..., "Z").
    Punctuation and whitespace will be removed from the predicted text.

    Parameters:
        pred_text (str): The predicted text to check.
        options (list): The list of valid options.

    Returns:
        bool: True if pred_text is a single option, False otherwise.
    """
    # Strip leading/trailing whitespaces and convert to uppercase
    pred_text = pred_text.strip().upper()

    # Ensure that the cleaned text is exactly one alphabetic character and is in the list of options
    return pred_text in options and len(pred_text) == 1


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'positive': [], 'negative': []}

    for prob_id, prob in split_problems.items():
        if prob_id not in predictions:
            pred_text = 'FAILED'
        else:
            pred = predictions[prob_id]
            pred_text = pred['text'].strip()  # 去除首尾空格

        # Check if the prediction is a single option
        if is_single_option(pred_text, args.options):
            result = {
                'question_id': prob_id,
                'parsed_ans': pred_text,
                'question': pred['prompt'],
                'valid': True
            }
            results['positive'].append(result)
        else:
            result = {
                'question_id': prob_id,
                'parsed_ans': pred_text,
                'question': pred['prompt'],
                'valid': False
            }
            results['negative'].append(result)

    # Calculate accuracy
    correct = len(results['positive'])
    total = correct + len(results['negative'])
    accuracy = correct / total * 100 if total > 0 else 0

    # Save results to output file
    with open(args.output_file, 'w') as f:
        # Write accuracy at the top
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        json.dump(results, f, indent=2)

    # Print summary
    print(f"Total Positive Samples: {len(results['positive'])}, Total Negative Samples: {len(results['negative'])}")
    print(f"Accuracy: {accuracy:.2f}%")
