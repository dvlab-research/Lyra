import os
from sphfile import SPHFile

input_dir = 'TEDLIUM_release-3/data/sph/'
output_dir = 'TEDLIUM_release-3/data/mp3/'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cnt = 0
for filename in os.listdir(input_dir):
    if filename.endswith('.sph'):
        sph_path = os.path.join(input_dir, filename)
        
        sph = SPHFile(sph_path)
        
        wav_filename = filename.replace('.sph', '.mp3')
        wav_path = os.path.join(output_dir, wav_filename)
        
        sph.write_wav(wav_path)
        cnt += 1

print(f"{cnt} files processed.")