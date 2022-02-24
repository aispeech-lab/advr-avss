"""
This script is used to generate train mixture list and return a batch of data;
Note: use cv2.resize to do upsample or downsample for 2-D array.
@zhangpeng
"""
import os
import numpy as np
import random
import config
import soundfile as sf
import cv2
from loss import cal_sisnr_order_loss
from torch.autograd import Variable
import torch


def generate_mixture_list(train_or_test, mix_spk_num, data_path, noise_path, all_spks, mixture_list_path, num_samples, scenario, e_id=None):
    """
    Generate mixture list for training;
    Parameters:
    train_or_test: 'train', 'test';
    mix_spk_num: 1, 2, 3;
    data_path: the path of clean audio;
    noise_path: the path of noise audio;
    all_spks: the number of speakers in data set (e.g., train or test);
    mixture_list_path: the path for saving generated train list;
    num_samples: the total number of samples for one epoch;
    scenario: '1':1speaker+noise, '2':2speaker, '3':2speaker+noise
    """
    file_name = open(os.path.join(mixture_list_path, 'mixture_{}_spk_{}_{}_online_{}.txt'.format(mix_spk_num, train_or_test, scenario, e_id)), 'w')
    for i in range(num_samples):
        selected_spks = random.sample(all_spks, mix_spk_num)
        if scenario == '1' or scenario == '3': # the scenario that need noise
            selected_noise = random.sample(os.listdir(noise_path), 1)[0]
            noise_ratio = round(-20*np.random.rand()-5, 3) # -25dB~-5dB
        line = ''
        if scenario == '1':
            ratio = 0.0
        else:
            ratio = round(5*np.random.rand()-2.5, 3)
        for spk in selected_spks:
            sample_name = random.sample(os.listdir(os.path.join(data_path, '{}_audio'.format(train_or_test), spk)), 1)[0]
            sample_name = sample_name[:-4]
            if spk == selected_spks[0]:
                line += '{}/{}/{}/'.format(spk, sample_name, ratio)
            elif spk == selected_spks[-1]:
                line += '{}/{}/{}/'.format(spk, sample_name, -1*ratio)
        if scenario == '1' or scenario == '3':
            line += '{}/{}'.format(selected_noise, noise_ratio)
        line += '\n'
        file_name.write(line)
    print('The train list is generated successfully !')


def process_audio(path):
    """
    Input: the path of audio;
    output: processed audio.
    """
    # load audio
    signal, sr = sf.read(path)  
    # if this audio has more than one channel, select the first channel
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    # do normalization for the audio
    signal -= np.mean(signal)  
    signal /= (np.max(np.abs(signal)) + np.spacing(1))
    """
    # make the audio has same length
    if signal.shape[0] < config.MAX_LEN_SPEECH: 
        signal = np.append(signal, np.zeros(config.MAX_LEN_SPEECH - signal.shape[0]))
    else:
        signal = signal[:config.MAX_LEN_SPEECH]
    """
    return signal


def parse_sample(sample, scenario):
    audio_spk = []
    audio_db = []
    audio_sample_name = []
    noise_name = []
    noise_db = []
    if scenario == '1':
        audio_spk.append(sample[0])
        audio_sample_name.append(sample[1])
        audio_db.append(float(sample[2]))
        noise_name.append(sample[3])
        noise_db.append(float(sample[4]))
    elif scenario == '2':
        audio_spk.append(sample[0])
        audio_spk.append(sample[3])
        audio_sample_name.append(sample[1])
        audio_sample_name.append(sample[4])
        audio_db.append(float(sample[2]))
        audio_db.append(float(sample[5]))
    else:
        audio_spk.append(sample[0])
        audio_spk.append(sample[3])
        audio_sample_name.append(sample[1])
        audio_sample_name.append(sample[4])
        audio_db.append(float(sample[2]))
        audio_db.append(float(sample[5]))
        noise_name.append(sample[6])
        noise_db.append(float(sample[7]))
    return audio_spk, audio_db, audio_sample_name, noise_name, noise_db


def prepare_data(train_or_test, scenario, e_id=None):
    """
    parameters:
    train_or_test: type = str, 'train', 'valid' or 'test'
    scenario: type = str, '1':1spk+noise; '2':2spk; '3':2spk+noise
    return: a batch of data
    """
    # load the video's fps for train or test data set
    with open(os.path.join(config.aim_path, config.DATASET, '{}_fps_list.txt'.format(train_or_test)), "r", encoding="utf-8") as f:
        fps_dict = {}
        for line in f:
            line = line.strip('\n')
            fps_dict[line.split('/')[0]] = float(line.split('/')[1])
    # set the batch size
    if train_or_test == 'train':
        BATCH_SIZE = config.BATCH_SIZE
    else:
        BATCH_SIZE = config.BATCH_SIZE_TEST
    # initialization
    mix_audio_list = []
    clean_audio_list = []
    visual_features_list = []
    """
    The length of audio and visual features have been pre-processed (3 second)
    visual_len = []
    clean_speech_len = []
    """
    # the path of data
    clean_data_path = os.path.join(config.aim_path, config.DATASET)
    noise_data_path = os.path.join(config.aim_path, 'AudioSet')
    all_speakers = os.listdir(os.path.join(clean_data_path, '{}_audio'.format(train_or_test))) # use audio as standard
    mixture_list_path = os.path.join(clean_data_path, 'mixture_list')
    if scenario == '1':
        mix_spk_num = 1
    else:
        mix_spk_num = 2 # include two scenarios: 2spk, 2spk+noise 
    print('The number of speakers in {} set is {}' .format(train_or_test, len(all_speakers)))
    # generate and determine the mixture list
    if train_or_test == 'train':
        if config.DATA_AUG == True: # generate the mixture data online (on the fly)
            # three scenarios: 1spk+noise, 2spk, 2spk+noise
            generate_mixture_list(train_or_test, mix_spk_num, clean_data_path, noise_data_path, all_speakers, 
                mixture_list_path, config.num_samples_per_epoch+1, scenario, e_id)
            aim_list_path = os.path.join(mixture_list_path, 'mixture_{}_spk_train_{}_online_{}.txt'.format(mix_spk_num, scenario, e_id)) 
        else:
            aim_list_path = os.path.join(mixture_list_path, 'mixture_{}_spk_train_{}.txt'.format(mix_spk_num, scenario))
    if train_or_test == 'test':
        aim_list_path = os.path.join(mixture_list_path, 'mixture_{}_spk_test_{}.txt'.format(mix_spk_num, scenario))
    # when in train phase, shuffle the mixture list
    all_samples_list = open(aim_list_path).readlines()
    if train_or_test == 'train' and config.SHUFFLE_BATCH:
        random.shuffle(all_samples_list)
        print('\nshuffle train list success!')
    # begin to load data 
    number_samples_all = len(all_samples_list)
    total_batch_mix = (number_samples_all-1) / BATCH_SIZE
    print('This epoch has {} batches'.format(total_batch_mix))
    sample_idx = 0
    batch_idx = 0
    for i in range(number_samples_all):
        if i == number_samples_all - 1: # loop termination condition
            yield False
        print('{}-speakers mixed sample {}/{}:'.format(mix_spk_num, sample_idx, batch_idx))
        sample = all_samples_list[sample_idx].strip('\n').split('/')
        audio_spk, audio_db, audio_sample_name, noise_name, noise_db = parse_sample(sample, scenario)
        for k, spk in enumerate(audio_spk):
            sample_name = audio_sample_name[k]
            # load audio
            audio_path = os.path.join(clean_data_path, '{}_audio'.format(train_or_test), spk, sample_name + '.wav')
            signal = process_audio(audio_path)
            clean_audio_list.append(signal)
            # load visual features
            visual_features_path = os.path.join(clean_data_path, '{}_npy'.format(train_or_test), spk, sample_name + '.npy')
            if not os.path.exists(visual_features_path):
                print("this sample's visual features are not exist!")
                visual_features = np.zeros([config.MAX_LEN_VIDEO, 256])
            else:
                visual_features = np.load(visual_features_path)
            """
            # make the visual_features have same length, need take fps into account
            fps = fps_dict[sample_name[:-2]] 
            if visual_features.shape[0] < config.MAX_LEN * fps:
                shape = visual_features.shape
                visual_features = np.vstack((visual_features, np.zeros((config.MAX_LEN * fps - shape[0], shape[1]))))
            else:
                visual_features = visual_features[:config.MAX_LEN * fps, :]
            """
            # do upsample or downsample for visual features
            visual_features = cv2.resize(visual_features, (256, config.MAX_LEN_VIDEO), interpolation=cv2.INTER_NEAREST)
            visual_features_list.append(visual_features)
            if k == 0:
                # for the first speaker
                ratio = 10 ** (audio_db[k] / 20.0)
                signal = ratio * signal
                mixture = signal
            else:
                ratio = 10 ** (audio_db[k] / 20.0)
                signal = ratio * signal
                mixture = mixture + signal
        # the condition that have noise
        if scenario == '1' or scenario == '3':
            noise_path = os.path.join(noise_data_path, noise_name[0])
            noise_audio = process_audio(noise_path)
            ratio = 10 ** (noise_db[0] / 20.0)
            mixture = mixture + noise_audio * ratio
        mix_audio_list.append(mixture)                                                 
        batch_idx += 1
        # return a batch of data
        if batch_idx == BATCH_SIZE:
            mix_audio_list = np.array(mix_audio_list)
            clean_audio_list = np.array(clean_audio_list)
            visual_features_list = np.array(visual_features_list)
            yield {'mixture': mix_audio_list,
                   'clean_audio': clean_audio_list,
                   'visual_features': visual_features_list
                   }
            # do initialization for next batch of data
            batch_idx = 0
            mix_audio_list = []
            clean_audio_list = []
            visual_features_list = []
        sample_idx += 1


if __name__ == "__main__":
    generator = prepare_data('test', '2')
    loss_list = []
    while True:
        data = generator.__next__()
        if data == False:
            break
        mixture = Variable(torch.from_numpy(data['mixture'].astype('float32'))).cuda()
        clean_audio = Variable(torch.from_numpy(data['clean_audio'].astype('float32'))).cuda()
        shape = clean_audio.shape
        clean_audio_len = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1])).cuda()
        clean_audio = clean_audio.unsqueeze(1)
        mixture = mixture.unsqueeze(1)
        shape = mixture.shape
        mixture = mixture.unsqueeze(1).expand(shape[0], 2, shape[1], shape[2]).contiguous().view(-1, shape[1], shape[2])
        loss = cal_sisnr_order_loss(clean_audio, mixture, clean_audio_len)
        loss_list.append(loss.item())
    loss_aver = np.array(loss_list).mean()
    print("mixture's loss_aver (SI-SNR): {}".format(loss_aver))
