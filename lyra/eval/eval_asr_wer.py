import os 
import jiwer
import json
import re
import argparse

def remove_punc(text):
    punc = '[,.!\']+\"<>?~*'
    return re.sub(punc, '', text)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str)
    return parser.parse_args()


def load_jsonl_to_dict(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data

if __name__ == "__main__":
    args = get_args()
    error = 0
    references = []
    hypothesises = []

    results = load_jsonl_to_dict(args.result_file)
    for result in results:
        references.append(remove_punc(result['gt_text']))
        hypothesises.append(remove_punc(result['text']))

    error = jiwer.wer(references, hypothesises,
                reference_transform=jiwer.wer_standardize_contiguous,
                hypothesis_transform=jiwer.wer_standardize_contiguous
                )

    print(f"WER: {error * 100:.3f}%")