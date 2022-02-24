"""
This script have 1 function:
1. categorize the file (.npy & .wav) according to the speaker's name.
@zhangpeng
"""


import os
import shutil

if __name__ == "__main__":
    source_video_path = "./source_video/train_npy_split"
    source_audio_path = "./source_video/train_audio_16khz_split"
    samples = os.listdir(source_video_path)
    for i, sample in enumerate(samples):
        print("process {}, count: {}".format(sample, i))
        speaker_label = sample[:-28]
        speaker_video_path = os.path.join(source_video_path, speaker_label)
        if not os.path.exists(speaker_video_path):
            os.mkdir(speaker_video_path)
        speaker_audio_path = os.path.join(source_audio_path, speaker_label)
        if not os.path.exists(speaker_audio_path):
            os.mkdir(speaker_audio_path)
        audio_path = sample.replace('npy', 'wav')
        if os.path.exists((os.path.join(speaker_video_path, sample))) or os.path.exists((os.path.join(speaker_audio_path, audio_path))):
            continue
        if not os.path.exists(os.path.join(source_video_path, sample)) or not os.path.exists(os.path.join(source_audio_path, audio_path)):
            continue
        shutil.move(os.path.join(source_video_path, sample), speaker_video_path)
        shutil.move(os.path.join(source_audio_path, audio_path), speaker_audio_path)