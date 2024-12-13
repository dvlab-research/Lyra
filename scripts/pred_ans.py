import pandas as pd
import argparse


def calculate_accuracy(file_path):
    df = pd.read_excel(file_path)

    df['question_id'] = df['index'] % 1000000

    grouped = df.groupby('question_id')

    total_questions = 0
    correct_questions = 0

    for name, group in grouped:
        total_questions += 1
        correct = True

        for idx, row in group.iterrows():
            if row['prediction'] != row['answer']:
                correct = False
                break
        if correct:
            correct_questions += 1

    accuracy = correct_questions / total_questions if total_questions > 0 else 0
    return accuracy


parser = argparse.ArgumentParser()
parser.add_argument("--file-path", type=str, required=True)
args = parser.parse_args()
accuracy = calculate_accuracy(args.file_path)
print(f"The accuracy of the answers is: {accuracy:.2%}")
