# coding:utf-8
"""
This script is used for generate test list for SS model;
@zhangpeng
"""


import os
import numpy as np
import random
import config as config


"""
Generate test list for three single-channel SS tasks: 1 speaker + noise ('1')、2 speakers ('2')、2 speakers + noise ('3').
"""


def generate_mixture_list(train_or_test, mix_spk_num, data_path, noise_path, all_spks, mixture_list_path, num_samples, scenario):
    file_name = open(os.path.join(mixture_list_path, 'mixture_{}_spk_{}_{}.txt'.format(mix_spk_num, train_or_test, scenario)), 'w')
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
            sample_name = random.sample(os.listdir(os.path.join(data_path, spk)), 1)[0]
            sample_name = sample_name[:-4]
            if spk == selected_spks[0]:
                line += '{}/{}/{}/'.format(spk, sample_name, ratio)
            elif spk == selected_spks[-1]:
                line += '{}/{}/{}/'.format(spk, sample_name, -1*ratio)
        if scenario == '1' or scenario == '3':
            line += '{}/{}'.format(selected_noise, noise_ratio)
        line += '\n'
        file_name.write(line)


train_or_test = 'test'
clean_audio_path = os.path.join(config.aim_path, config.DATASET, train_or_test+'_audio')
noise_path = os.path.join(config.aim_path, 'AudioSet')
all_speakers = os.listdir(clean_audio_path)
mixture_list_path =  os.path.join(config.aim_path, config.DATASET, "mixture_list")
if not os.path.exists(mixture_list_path):
    os.mkdir(mixture_list_path)
# generate_mixture_list(train_or_test, 1, clean_audio_path, noise_path, all_speakers, mixture_list_path, 10000, '1')
generate_mixture_list(train_or_test, 2, clean_audio_path, noise_path, all_speakers, mixture_list_path, 10000, '2')
# generate_mixture_list(train_or_test, 2, clean_audio_path, noise_path, all_speakers, mixture_list_path, 10000, '3')