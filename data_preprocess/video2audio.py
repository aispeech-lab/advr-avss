"""
This scripts is used to extract audio from video
@zhangpeng
"""


from moviepy.video.io.VideoFileClip import VideoFileClip
from resampy import resample
import os
import soundfile as sf


def video2audio(source_path, target_path):
    """
    Parameters:
    source_path: the path of video;
    target_path: the path of saving the audio.
    """
    try:
       video = VideoFileClip(source_path)
       audio = video.audio
       audio.write_audiofile(target_path)
       return True
    except Exception as e:
        print(e)
        return False


def resample_audio(source_path, target_path):
    """
    Function:
    Resample the audio's sample rate to 16 khz.
    Parameters:
    source_path: the path of original audio;
    target_path: the path of resampled audio.
    """
    try:
        original_audio, sr = sf.read(source_path)
        resampled_audio = resample(original_audio[:, 0], sr, 16000)
        sf.write(target_path, resampled_audio, 16000)
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    sample_list_path = "./source_video/train_fps_list_1.txt"
    source_path = './source_video/train'
    target_path = './source_video/train_audio'
    resampled_audio_path = './source_video/train_audio_16khz'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if not os.path.exists(resampled_audio_path):
        os.mkdir(resampled_audio_path)
    samples = open(sample_list_path, 'r').readlines()
    for i, sample in enumerate(samples):
        sample = sample.strip('\n').split('/')
        video_name = sample[0]
        print('Begin to process the video: {}, count: {}'.format(video_name, i))
        video_path = os.path.join(source_path, video_name)
        save_audio_path = os.path.join(target_path, video_name.replace('mp4', 'wav'))
        save_resampled_audio_path = os.path.join(resampled_audio_path, video_name.replace('mp4', 'wav'))
        l = video2audio(video_path, save_audio_path)
        if l:
            print("Extract audio from video successfully!")
            ll = resample_audio(save_audio_path, save_resampled_audio_path)
            if ll:
                print("Resample the sample rate of original audio successfully!")
        print('END', '\n')