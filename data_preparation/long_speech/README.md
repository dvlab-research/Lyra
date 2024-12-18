### Download TED-LIUM dataset

1. Download `TEDLIUM_release-3.tgz` from https://www.openslr.org/51/ and put it to `data_preparation/long_speech/`.

2. Extract files from `TEDLIUM_release-3.tgz`.

3. Run `data_preparation/long_speech/audio_convert.py` to convert `.sph` files to `.mp3` files. (Package `sphfile` needed)

```bash
pip install sphfile
cd data_preparation/long_speech
python audio_convert.py
```

4. Move `data_preparation/long_speech/TEDLIUM_release-3/data/mp3/*` to `data_preparation/long_speech/Audios/`.

5. `data_preparation/long_speech/TEDLIUM_release-3/` can be deleted now.

### Download remaining videos

Run `audio_download.py` to download audio files from YouTube. (Package `yt_dlp` needed)

```bash
pip install yt_dlp
cd /data_preparation/long_speech
python audio_download.py
```

### Download QA data

Download `Lyra-LongSpeech.json` from https://huggingface.co/datasets/zszhong/Lyra-Data/tree/main/Lyra_SFT/long_speech, and put it to `data_preparation/long_speech`.

