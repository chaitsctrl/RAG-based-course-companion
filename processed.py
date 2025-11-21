# convert mp4 files to mp3 files - can be passed on to whisper for transcription
import os
import subprocess

files = os.listdir('videos')
for file in files:
    vid_name = file.split('_')[1].split('.')[0]
    print(vid_name)
    subprocess.run(['ffmpeg', '-i', f'15/{file}', '-q:a', '0', '-map', 'a', f'audios/{vid_name}.mp3'])

