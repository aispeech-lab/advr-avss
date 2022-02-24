"""
This script have 1 function:
1. split the audio and visual features to 3 second, when split visual features, we need take the fps into account;
@zhangpeng
"""


import os
import soundfile as sf
import numpy as np


def split_audio_video(video_path, audio_path, save_video_path, save_audio_path, video_name, fps):
    """
    Parameters:
    video_path: the path of visual features;
    audio_path: the path of audio;
    save_video_path: the path of saving the split visual features;
    save_audio_path: the path of saving the split audio;
    fps: the FPS of video(visual features)
    """
    t_video_len = 3 * np.ceil(fps)
    t_audio_len = 3 * 16000
    visual_features = np.load(video_path)
    audio, sr = sf.read(audio_path)
    num = int(np.float(audio.shape[0] / t_audio_len))
    if num >= 1:
        for i in range(num):
            sub_visual_features = visual_features[int(i*t_video_len):int((i+1)*t_video_len), :]
            sub_audio = audio[int(i*t_audio_len):int((i+1)*t_audio_len)]
            sub_visual_features_path = os.path.join(save_video_path, video_name[:-4]+'_{}.npy'.format(str(i)))
            np.save(sub_visual_features_path, sub_visual_features)
            sub_audio_path = os.path.join(save_audio_path, video_name[:-4]+'_{}.wav'.format(str(i)))
            sf.write(sub_audio_path, sub_audio, 16000)
    return num


if __name__ == "__main__":
    source_video_path = "./source_video/train_npy"
    source_audio_path = "./source_video/train_audio_16khz"
    target_video_path = "./source_video/train_npy_split"
    target_audio_path = "./source_video/train_audio_16khz_split" 
    sample_list_path = "./source_video/train_fps_list_1.txt"
    if not os.path.exists(target_video_path):
        os.mkdir(target_video_path)
    if not os.path.exists(target_audio_path):
        os.mkdir(target_audio_path)
    samples = open(sample_list_path, 'r').readlines()
    for i, sample in enumerate(samples):
        sample = sample.strip('\n').split('/')
        video_name, fps = sample[0], float(sample[1])
        print("Begin to process {}, count:{}!".format(video_name, i))
        video_path = os.path.join(source_video_path, video_name.replace('mp4', 'npy'))
        audio_path = os.path.join(source_audio_path, video_name.replace('mp4', 'wav'))
        if not os.path.exists(audio_path):
            continue
        if not os.path.exists(video_path):
            continue
        num = split_audio_video(video_path, audio_path, target_video_path, target_audio_path, video_name, fps)
        print("This video/audio is splited into {} clips successfully!".format(num))
        print("END", '\n')