import asyncio
import edge_tts
import json
from tqdm import tqdm
import os
import argparse
import re

def replace_choice(string):
    return string.replace("\nA.", "\nOption A is ").replace("\nB.", "\nOption B is ").replace("\nC.", "\nOption C is ").replace("\nD.", "\nOption D is ")

def is_only_english(text):
    """
    Check if the text contains only English letters, numbers, punctuation, and spaces.
    It does not contain characters from languages like Chinese, Japanese, etc.
    """
    # Define the allowed character set: A-Z, a-z, 0-9, common punctuation marks, and spaces
    pattern = re.compile(r'^[A-Za-z0-9\s.,!?\'"()\-:;]+$')
    return bool(pattern.match(text))


async def generate_audio(text, output_file, voice, max_retries=10, delay=100):
    """
    Generate audio and save it to the specified file with a retry mechanism.
    
    Parameters:
        text (str): The text to convert into speech.
        output_file (str): The path to save the audio file.
        voice (str): The voice type to use.
        max_retries (int): The maximum number of retries.
        delay (int): The delay (in seconds) before retrying.
    """
    for attempt in range(1, max_retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            # print(f"Successfully generated audio: {output_file}")
            break  # Break the loop after success
        except Exception as e:
            if attempt == max_retries:
                print(f"Failed: Unable to generate audio {output_file}, reached max retry attempts. Error: {e}")
                raise  # Raise exception after the last failure
            else:
                print(f"Warning: Failed to generate audio {output_file}, attempt {attempt}/{max_retries}. Error: {e}. Retrying in {delay} seconds.")
                print(f"Text that caused the error: {text}")
                await asyncio.sleep(delay)  # Wait before retrying


async def generate_all_audio(all_text, speech_ids_all, save_path, voice, batch_size=10):
    """
    Generate audio in batches to limit concurrency.
    """
    all_audio_path = []
    tasks = []

    for i in tqdm(range(len(all_text)), desc="Preparing audio tasks"):
        try:
            text = all_text[i]
            speech_id = speech_ids_all[i]
            cur_save_path = os.path.join(save_path, f"{speech_id}.mp3")

            if os.path.exists(cur_save_path):
                try:
                    wav,sr = torchaudio.load(cur_save_path)
                    # with open(cur_save_path, 'rb') as f:
                    #     f.read(10)  # Try reading the first 10 bytes
                    # File exists and is readable, skipping generation
                    all_audio_path.append(cur_save_path)
                    continue
                except Exception as e:
                    print(f"Existing file {cur_save_path} is not readable. Regenerating. Error: {e}")

            # Create task and add to the task list
            tasks.append(generate_audio(text, cur_save_path, voice))
            all_audio_path.append(cur_save_path)
        except Exception as e:
            print(f"Error preparing audio for ID {speech_id}: {e}")

    # Execute tasks in batches to avoid excessive concurrency
    for i in tqdm(range(0, len(tasks), batch_size), desc="Generating audio"):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)

    return all_audio_path

if __name__ == "__main__":
    
    # ============================================Read Text================================================= #

    parser = argparse.ArgumentParser(description="TTS generation script")
    parser.add_argument(
        '--sft_data_path', 
        type=str, 
        help='Path to SFT data'
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        help='Path to save the generated audio files'
    )
    parser.add_argument(
        '--save_path_json', 
        type=str, 
        help='Path to save the generated JSON file'
    )
    args = parser.parse_args()

    sft_data_path = args.sft_data_path
    save_path = args.save_path
    save_path_json = args.save_path_json

    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    with open(sft_data_path, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)

    print(f"Total data entries: {len(sft_data)}")

    # Check the data source
    for i in range(len(sft_data)):
        if sft_data[i]['conversations'][:2][1]['from'] != 'gpt':
            print(f"Index {i} source is not 'gpt'")

    All_text = []
    speech_ids_all = []
    data_idx = []
    
    for i in range(len(sft_data)):
        try:
            cur_text = replace_choice(sft_data[i]['conversations'][:2][1]['value'])
            if is_only_english(cur_text):
                All_text.append(cur_text)
                speech_ids_all.append(sft_data[i]['id'])
                data_idx.append(i)
        except KeyError as e:
            print(f"Data index {i} is missing key: {e}")

    ### =================== VOICE Param ==================== ###
    VOICE = "en-GB-SoniaNeural"

    # ============================================END================================================= #
        
    # Generate all audio
    All_audio_path = asyncio.run(generate_all_audio(All_text, speech_ids_all, save_path, VOICE, batch_size=10))

    # Construct a new data structure
    New_data = []
    min_num = min(len(All_audio_path), len(data_idx))
    for i in range(min_num):
        idx = data_idx[i]
        cur_dict = sft_data[idx].copy()
        cur_dict['conversations'] = cur_dict['conversations'][:2]
        cur_dict['conversations'][:2][1]['value'] = replace_choice(sft_data[idx]['conversations'][:2][1]['value'])
        cur_dict['speech'] = All_audio_path[i]
        New_data.append(cur_dict)

    # Save the new JSON file
    with open(save_path_json, 'w', encoding='utf-8') as json_file:
        json.dump(New_data, json_file, ensure_ascii=False, indent=4)

    print(f"Audio generation complete, saved to {save_path_json}")
