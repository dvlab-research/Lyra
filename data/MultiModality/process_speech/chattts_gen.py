import ChatTTS
from IPython.display import Audio
import torchaudio
import torch
import json
import random
from tqdm import tqdm
import sys

# Example: python test.py [0,1,2,3,4,5,6]

chat = ChatTTS.Chat()
chat.load_models(compile=True, source='local', local_path='./', device='cuda:{}'.format(int(sys.argv[1])+1)) # Set to True for better performance

sft_text_path = "sft_ques_total_with_modified_options.json" # total is 1511341
# sft_text_path = "sft_ques_textOCR.json" # total is 2755

partition = 2000 # @TODO change this 
sft_text = json.load(open(sft_text_path, 'r'))[int(sys.argv[1])*partition: (int(sys.argv[1])+1)*partition]

texts_all = []
ids_all = []
for item in sft_text:
    texts_all.append(item["question"])
    ids_all.append(item["speech_id"])

split = 150 # @TODO change this 
for i in tqdm(range(len(texts_all) // split + 1)):
    try:
        texts = texts_all[i*split:(i+1)*split]
        ids = ids_all[i*split:(i+1)*split]
        
        ###################################
        # Sample a speaker from Gaussian.

        rand_spk = chat.sample_random_speaker()

        params_infer_code = {
            'spk_emb': rand_spk, # add sampled speaker 
            'temperature': .2, # using custom temperature
            'top_P': 0.6, # top P decode
            'top_K': 17, # top K decode
        }

        wavs = chat.infer(
            texts, 
            params_infer_code=params_infer_code
        )

        for wav_id in range(len(wavs)):
            torchaudio.save("mgm_sft_speech/Lyra_speech/{}.mp3".format(ids[wav_id]), torch.from_numpy(wavs[wav_id]), 24000)
        
    except Exception as e:
        print("Error: ", e)
        continue