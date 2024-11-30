import os
import yt_dlp
import json
import time

with open(os.path.join("download_ID.json"), "r") as f:
    video_ids = json.load(f)

save_folder = "Audios"

ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True, 
        'audioformat': 'mp3',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'age_limit': None,
        'noplaylist': True,
        'outtmpl': os.path.join(save_folder, '%(id)s.%(ext)s'),
        'extr act_flat': True,
    }

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for video_id in video_ids:
        if os.path.isfile(os.path.join(save_folder, video_id+".mp3")):
            continue
        try:
            ydl.download(video_id)
        except:
            continue
        time.sleep(1)
