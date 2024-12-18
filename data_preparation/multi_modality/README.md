## Audio Data Preparation
Lyra supports both sound and speech. The audio data preparation scripts are organized as follows.

```
|──data_preparation
|   |──multi_modality
|   |   |──README.md
|   |   |──process_speech
|   |   |──process_sound
|   |──... 
```

### Speech

Speech data in Lyra contains two parts: the pre-training data and the supervised fine-tuning data. 

#### Pre-training Data
- Install necessary packages:
```angular2html
# Install torchaudio
pip install torchaudio
# Install whisper
pip install -U openai-whisper
```

- Download Librispeech speeches from [here](https://www.openslr.org/12). You can download train-clean-100.tar.gz, train-clean-360.tar.gz, train-other-500.tar.gz and test-clean.tar.gz.

- Generate annotation for Librispeech, using [prepare_librispeech.py](process_speech/prepare_librispeech.py). Remember changing --input_dir, --output_file and --splits in the python file.
```angular2html
python process_speech/prepare_librispeech.py --input_dir '[YOUR Librispeech SPEECHES PATH]' \
    --output_file '[OUTPUT ANNOTATION PATH]' \
    --splits=[SPLITS]

# For example
python process_speech/prepare_librispeech.py --input_dir 'dataset/librispeech' \
     --output_file 'dataset/librispeech/annotation/test-clean.jsonl' \
     --splits='test-clean'
```

- Once you get the raw annotation of Librispeech, you should modify it to supervised fine-tuning format, using [prepare_speech_sft_data.py](process_speech/prepare_speech_sft_data.py). Remember changing line 248 and 267 in the python file.

#### Supervised Fine-tuning Data

We use [ChatTTS](https://github.com/2noise/ChatTTS) to convert text-image SFT data to speech-image SFT data.

- Add unique speech id and choose the text to be converted. Please refer to [prepare_speech_image_sft_data.ipynb](process_speech/prepare_speech_image_sft_data.ipynb) (Part 1. Add speech ID and filter text).

- Generate speeches using [chattts_gen.py](process_speech/chattts_gen.py)


- Process the original SFT annotation. Please refer to [prepare_speech_image_sft_data.ipynb](process_speech/prepare_speech_image_sft_data.ipynb) (Part 2. Reorganize SFT annotation data).


- The final speech's folder architecture should be 
```
├── data
│   ├── Lyra_SFT
│   │   ├── multi_modality_speech
│   │   │   ├── lyra_multimodal.json
│   │   │   ├── Lyra_MM
│   │   │   ├── LibriSpeech
```

### Sound 
- Download `csv` file [here](https://github.com/cdjkim/audiocaps/tree/master/dataset). The header of the CSV file are:
```angular2html
audiocap_id,youtube_id,start_time,caption
```

- Install necessary packages
```angular2html
# Install ffmpeg
sudo apt install ffmpeg
# Install audiocaps-download
pip install audiocaps-download
```

- Download audio. Be careful to replace sound_store_path with your own path.
You can follow [here](https://pypi.org/project/audiocaps-download/) for detailed usage of `audiocaps_download.Downloader`.:
```angular2html
sound_store_path = [YOUR STORE PATH]  # change to your own path
from audiocaps_download import Downloader
d = Downloader(root_path=sound_store_path, n_jobs=16)
d.download(format = 'wav')
```


- Process the raw sound data.
Once you get the raw annotation of AudioCaps, you should modify it to supervised fine-tuning format, using [prepare_sound_sft_data.py](process_sound/prepare_sound_sft_data.py). Remember changing annotation_path and save_path in the python file.



