### Download TED-LIUM dataset

1. Download `TEDLIUM_release-3.tgz` from https://www.openslr.org/51/ and put it to `data/LongSpeech/`.

2. Extract files from `TEDLIUM_release-3.tgz`.

3. Run `data/LongSpeech/audio_convert.py` to convert `.sph` files to `.mp3` files. (Package `sphfile` needed)

```bash
pip install sphfile
cd data/LongSpeech
python audio_convert.py
```

4. Move `data/LongSpeech/TEDLIUM_release-3/data/mp3/*` to `data/LongSpeech/Audios/`.

5. `data/LongSpeech/TEDLIUM_release-3/` can be deleted now.

### Download remaining videos

Run `audio_download.py` to download audio files from YouTube. (Package `yt_dlp` needed)

```bash
pip install yt_dlp
cd data/LongSpeech
python audio_download.py
```

### Download QA data

Download `Lyra-LongSpeech.json` from ***URL[TODO]***, and put it to `data/LongSpeech/`

