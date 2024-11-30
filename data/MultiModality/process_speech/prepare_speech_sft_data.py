import pandas as pd
import json
import random
import os
import csv

speech2text_instructions = [
    "Analyze the spoken words and transcribe them accurately.",
    "Can you please convert this speech into written text?",
    "Listen carefully and transform the spoken words into text.",
    "Provide a textual representation of the spoken content.",
    "Convert the speech into a written transcript.",
    "Recognize and write down what is being said.",
    "Generate a text version of the spoken words.",
    "Transform the spoken language into text form.",
    "Please transcribe this speech into written text.",
    "Write down the spoken words accurately.",
    "Turn the spoken words into written format.",
    "Capture the spoken content as text.",
    "Transform the speech into a textual representation.",
    "Generate a transcript of the spoken words.",
    "Provide a written version of the spoken words.",
    "Transcribe the speech for better understanding.",
    "Change the spoken language into text.",
    "Record the spoken words in textual form.",
    "Please transcribe this spoken content.",
    "Recognize the spoken words and write them down.",
    "Convert the speech into written format.",
    "Generate a textual representation of the speech.",
    "Transform the spoken words into text form.",
    "Capture the spoken content as written text.",
    "Convert the spoken language into text.",
    "Provide a transcript of the spoken words.",
    "Convert the speech into a written document.",
    "Recognize and write out what is being said.",
    "Please transform this speech into written text.",
    "Generate a text version of the spoken content.",
    "Transform the spoken language into text form.",
    "Capture the spoken content as a transcript.",
    "Provide a textual representation of the speech.",
    "Transcribe the spoken words with precision.",
    "Convert the speech into a written transcript.",
    "Recognize and transcribe the spoken content.",
    "Write down the spoken words accurately.",
    "Change the spoken language into text.",
    "Generate a transcript of the spoken words.",
    "Capture the spoken content as written text.",
    "Transform the speech into a textual representation.",
    "Recognize the speech and write it out.",
    "Convert the spoken words into written format.",
    "Write down what you hear in the speech.",
    "Transcribe the speech for better understanding.",
    "Convert the spoken language into text.",
    "Record the spoken words in textual form.",
    "Please transcribe this spoken content accurately.",
    "Recognize the spoken words and transcribe them.",
    "Convert the speech into a text document.",
    "Generate a text version of the spoken content.",
    "Capture the spoken content as text.",
    "Convert the spoken language into written format.",
    "Provide a transcript of the spoken words.",
    "Convert the speech into a written script.",
    "Recognize and write out what is being spoken.",
    "Transform the spoken words into text format.",
    "Change the spoken language into written text.",
    "Generate a textual representation of the speech.",
    "Transform the speech into written format.",
    "Capture the spoken content as a transcript.",
    "Provide a written version of the spoken words.",
    "Transcribe the spoken words accurately.",
    "Convert the speech into a written transcript.",
    "Write down the spoken words clearly.",
    "Convert the spoken language into text form.",
    "Generate a transcript of the spoken words.",
    "Capture the spoken content as written text.",
    "Transform the speech into a textual representation.",
    "Recognize the speech and write it down.",
    "Convert the spoken words into written form.",
    "Write down what you hear in the speech.",
    "Transcribe the speech for better understanding.",
    "Convert the spoken language into text.",
    "Record the spoken words as text.",
    "Please transcribe this spoken content clearly.",
    "Recognize the spoken words and transcribe them accurately.",
    "Convert the speech into a text document.",
    "Generate a text version of the spoken content.",
    "Capture the spoken content as text.",
    "Convert the spoken language into written format.",
    "Provide a transcript of the spoken words.",
    "Convert the speech into a written transcript.",
    "Recognize and write out what is being spoken.",
    "Transform the spoken words into text format.",
    "Change the spoken language into written text.",
    "Generate a textual representation of the speech.",
    "Transform the speech into written format.",
    "Capture the spoken content as a transcript.",
    "Provide a written version of the spoken words.",
    "Transcribe the spoken words accurately.",
    "Convert the speech into a written script.",
    "Write down the spoken words clearly.",
    "Convert the spoken language into text form.",
    "Generate a transcript of the spoken words.",
    "Capture the spoken content as written text.",
    "Transform the speech into a textual representation.",
    "Recognize the speech and write it down.",
    "Convert the spoken words into written form.",
    "Write down what you hear in the speech.",
    "Transcribe the speech for better understanding.",
    "Convert the spoken language into text.",
    "Record the spoken words as text.",
    "Please transcribe this spoken content clearly.",
    "Recognize the spoken words and transcribe them accurately.",
    "Convert the speech into a text document.",
    "Generate a text version of the spoken content.",
    "Capture the spoken content as text.",
    "Convert the spoken language into written format.",
    "Provide a transcript of the spoken words.",
    "Convert the speech into a written transcript.",
    "Recognize and write out what is being spoken.",
    "Transform the spoken words into text format.",
    "Change the spoken language into written text.",
    "Generate a textual representation of the speech.",
    "Transform the speech into written format.",
    "Capture the spoken content as a transcript.",
    "Provide a written version of the spoken words.",
    "Transcribe the spoken words accurately.",
    "Convert the speech into a written script.",
    "Write down the spoken words clearly.",
    "Convert the spoken language into text form.",
    "Generate a transcript of the spoken words.",
    "Capture the spoken content as written text.",
    "Transform the speech into a textual representation.",
    "Recognize the speech and write it down.",
    "Convert the spoken words into written form.",
    "Write down what you hear in the speech.",
    "Transcribe the speech for better understanding.",
    "Convert the spoken language into text.",
    "Record the spoken words as text.",
    "Please transcribe this spoken content clearly.",
    "Recognize the spoken words and transcribe them accurately.",
    "Convert the speech into a text document.",
    "Generate a text version of the spoken content.",
    "Capture the spoken content as text.",
    "Convert the spoken language into written format.",
    "Provide a transcript of the spoken words.",
    "Convert the speech into a written transcript.",
    "Recognize and write out what is being spoken.",
    "Transform the spoken words into text format.",
    "Change the spoken language into written text.",
    "Generate a textual representation of the speech.",
    "Transform the speech into written format.",
    "Capture the spoken content as a transcript.",
    "Provide a written version of the spoken words.",
    "Transcribe the spoken words accurately.",
    "Convert the speech into a written script.",
    "Write down the spoken words clearly.",
    "Convert the spoken language into text form.",
    "Generate a transcript of the spoken words.",
    "Capture the spoken content as written text.",
    "Transform the speech into a textual representation.",
    "Recognize the speech and write it down.",
    "Convert the spoken words into written form.",
    "Write down what you hear in the speech.",
    "Transcribe the speech for better understanding.",
    "Convert the spoken language into text.",
    "Record the spoken words as text.",
    "Please transcribe this spoken content clearly.",
    "Recognize the spoken words and transcribe them accurately.",
    "Convert the speech into a text document.",
    "Generate a text version of the spoken content.",
    "Capture the spoken content as text.",
    "Convert the spoken language into written format.",
    "Provide a transcript of the spoken words.",
    "Convert the speech into a written transcript.",
    "Recognize and write out what is being spoken.",
    "Transform the spoken words into text format.",
    "Change the spoken language into written text.",
    "Generate a textual representation of the speech.",
    "Transform the speech into written format.",
    "Capture the spoken content as a transcript.",
    "Provide a written version of the spoken words.",
    "Transcribe the spoken words accurately.",
    "Convert the speech into a written script.",
    "Write down the spoken words clearly.",
    "Convert the spoken language into text form.",
    "Generate a transcript of the spoken words.",
    "Capture the spoken content as written text.",
    "Transform the speech into a textual representation.",
    "Recognize the speech and write it down.",
    "Convert the spoken words into written form.",
    "Write down what you hear in the speech.",
    "Transcribe the speech for better understanding.",
    "Convert the spoken language into text.",
    "Record the spoken words as text.",
    "Please transcribe this spoken content clearly.",
    "Recognize the spoken words and transcribe them accurately.",
    "Convert the speech into a text document.",
    "Generate a text version of the spoken content.",
    "Capture the spoken content as text.",
    "Convert the spoken language into written format.",
    "Provide a transcript of the spoken words.",
    "Convert the speech into a written transcript.",
    "Recognize and write out what is being spoken.",
    "Transform the spoken words into text format.",
    "Change the spoken language into written text.",
    "Generate a textual representation of the speech.",
    "Transform the speech into written format.",
    "Capture the spoken content as a transcript.",
    "Provide a written version of the spoken words.",
    "Transcribe the spoken words accurately.",
    "Convert the speech into a written script.",
    "Write down the spoken words clearly.",
    "Convert the spoken language into text form.",
    "Generate a transcript of the spoken words.",
    "Capture the spoken content as written text.",
    "Transform the speech into a textual representation.",
    "Recognize the speech and write it down.",
    "Convert the spoken words into written form.",
    "Write down what you hear in the speech.",
    "Transcribe the speech for better understanding.",
    "Convert the spoken language into text.",
    "Record the spoken words as text.",
    "Please transcribe this spoken content clearly.",
    "Recognize the spoken words and transcribe them accurately.",
    "Convert the speech into a text document.",
    "Generate a text version of the spoken content.",
    "Capture the spoken content as text.",
    "Convert the spoken language into written format.",
    "Provide a transcript of the spoken words.",
    "Convert the speech into a written transcript.",
    "Recognize and write out what is being spoken.",
    "I need you to figure out the words spoken in this recording.",
]

speech_instructions_length = len(speech2text_instructions)


def convert_librispeech_to_json(annotation_path, split, raw_file_type='jsonl'):
    new_json_data = []
    with open(os.path.join(annotation_path, split+'.'+raw_file_type), 'r') as f:
        for line in f:
            item= json.loads(line.strip())
            id = item['audio'].split('/')[-1].split('.')[0]
            if len(item["text"].split()) <= 10 or len(item["text"].split()) >= 150: # drop EN txt length < 10 and length >150
                continue
            new_item = {
                'id': 'librispeech_' + id,
                'speech': item['audio'][27:], # TODO change this slice to ignore prefix in data path
                'speech_asr': item["text"]
            }
            new_item["conversations"] = [
                {"from": "human", "value": "<speech>\n"+speech2text_instructions[random.randint(0,speech_instructions_length-1)]},
                {"from": "gpt", "value": item["text"]}
            ]
            new_json_data.append(new_item)

    print(new_json_data[0])
    print(split, "len: ", len(new_json_data))
    return new_json_data



if __name__ == '__main__':
    speech_data = []

    # librispeech ['train-clean-100', 'train-clean-360', 'train-other-500']
    annotation_path = 'data/LibriSpeech' # TODO Change here
    split = 'train-clean-100'
    speechs = convert_librispeech_to_json(annotation_path, split)
    speech_data.extend(speechs)

    split = 'train-clean-360'
    speechs = convert_librispeech_to_json(annotation_path, split)
    speech_data.extend(speechs)

    split = 'train-other-500'
    speechs = convert_librispeech_to_json(annotation_path, split)
    speech_data.extend(speechs)
    
    # json to file
    with open('librispeech_filter_train.json', 'w', encoding='utf-8') as json_file:
        json.dump(speech_data, json_file, ensure_ascii=False, indent=4) 
    
    print("convert all to Lyra json format, total: ", len(speech_data)) # totally 281k pairs
