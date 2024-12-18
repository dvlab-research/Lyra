## Pipeline
Our entire process consists of two parts: converting QA data into audio and converting audio data into discrete units.

## Text-to-Audio
We use Microsoft's open-source [Edge-TTS](https://github.com/rany2/edge-tts) to convert the Answer into audio. To ensure consistency in the generated voices, we select only one person's voice. Specifically, in our experiments, we choose the default voice "en-GB-SoniaNeural".  

We provide an example script, which includes functionalities such as checking if the text is entirely in English, and transforming multiple-choice questions into a more readable format.  

The code is as follows:

```python
python generate_audio.py \
    --sft_data_path <Path to SFT data> \
    --save_path <Generated Audio Save Path> \
    --save_path_json <Path to save the generated audio JSON file>
```

## Audio-to-Units
To adapt to the current auto-regressive training approach for LLMs, we convert the audio into units. Specifically, we follow the [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni) and [SpeechGPT](https://github.com/0nutation/SpeechGPT) methods.

We employ [mHuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md) as the speech tokenizer to discretize speech data into discrete units and remove the repetitive units of adjacent frames to get reduced units.

### Download
```bash
cd data_preparation/speech_generation
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
```

### Discretize
```python
python3 speech2unit.py --wav path/to/wav
```
## Acknowledgements
We extend our gratitude to the incredible open-source contributions of [Edge-TTS](https://github.com/rany2/edge-tts), [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni), [SpeechGPT](https://github.com/0nutation/SpeechGPT), and [mHuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md).