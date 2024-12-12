import pandas as pd
import json
import random
import os
import csv

sound2text_instructions = [
    "Provide a brief description of the given sound.",
    "Share a concise interpretation of the sound provided.",
    "Summarize the key points from the sound clip.",
    "Describe the main events or actions occurring in the sound.",
    "Identify and explain the primary theme or message of the sound.",
    "Highlight any significant sounds or voices heard in the sound.",
    "Give an overview of the tone and mood conveyed by the sound.",
    "Note any background noises or ambient sounds in the sound.",
    "Describe the setting or context in which the sound takes place."
]

sound_instructions_length = len(sound2text_instructions)

def convert_audiocaps_csv_to_json(annotation_path, split='train'):
    # read csv
    df = pd.read_csv(annotation_path)

    # convert to json
    json_str = df.to_json(orient='records', force_ascii=False)

    json_data = json.loads(json_str)
    
    max_len = 0

    new_json_data = []
    for item in json_data:
        cap_len = len(item["caption"].split())
        max_len = max(max_len, cap_len)
        if cap_len < 2:
            continue
        new_item = {}
        new_item["id"] = "audiocaps_" + str(item["audiocap_id"])
        new_item["sound"] = "audiocaps/sounds/{}/".format(split) + str(item["audiocap_id"]) + ".wav"
        new_item["conversations"] = [
            {"from": "human", "value": "<sound>\n"+ sound2text_instructions[random.randint(0,sound_instructions_length-1)]},
            {"from": "gpt", "value": item["caption"]}
        ]
        new_item["youtube_id"] = item["youtube_id"]
        new_item["caption"] = item["caption"]

        new_json_data.append(new_item)
    
    print(new_json_data[0])

    print("convert audiocaps to json, total: ", len(new_json_data))
    print("max_len: ", max_len)

    return new_json_data 



if __name__ == '__main__':
    
    sound_data = []

    # audiocaps
    annotation_path = 'audiocaps/annotations/train.csv' # replace with your own path
    audiocaps_sound = convert_audiocaps_csv_to_json(annotation_path,'train')
    sound_data.extend(audiocaps_sound)

    annotation_path = 'audiocaps/annotations/val.csv' # replace with your own path
    audiocaps_sound = convert_audiocaps_csv_to_json(annotation_path,'val')
    sound_data.extend(audiocaps_sound)

    # json to file
    save_path = 'audiocaps_train_val.json' # replace with your own save path
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(sound_data, json_file, ensure_ascii=False, indent=4)
    
    print("convert all to Lyra annotation format, total: ", len(sound_data))