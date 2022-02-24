# coding:utf-8
"""
Online streaming inference strategy or offline inference.
@zhangpeng
"""

import time
import torch
from torch.autograd import Variable
import numpy as np 
import config
from dataloader import prepare_data
import os 
import soundfile as sf 
from model import AVMS
import argparse
from loss import cal_sisnr_order_loss
from separation import bss_eval_sources
from pypesq import pesq
from pystoi import stoi
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def compute_metric(source, predict, mix):
    source = source.squeeze(1).data.cpu().numpy()
    predict = predict.squeeze(1).data.cpu().numpy()
    mix = mix.data.cpu().numpy()
    B = source.shape[0]
    STOI = []
    STOIn = []
    PESQ = []
    PESQn = []
    for i in range(int(B)):
        source_idx = source[i,:]
        predict_idx = predict[i,:]    
        STOI_ = stoi(source_idx, predict_idx, 16000)
        STOI_n = stoi(source[i,:], mix[int(i/2),:], 16000)
        PESQ_ = pesq(source_idx, predict_idx, 16000)
        PESQ_n = pesq(source[i,:], mix[int(i/2),:], 16000)
        STOI.append(STOI_)
        PESQ.append(PESQ_)
        STOIn.append(STOI_n)
        PESQn.append(PESQ_n)
    STOI = np.array(STOI)
    PESQ = np.array(PESQ)
    STOIn = np.array(STOIn)
    PESQn = np.array(PESQn)
    STOI = STOI.mean()
    PESQ = PESQ.mean()
    STOIn = STOIn.mean()
    PESQn = PESQn.mean()
    print('STOI PESQ STOI_n PESQ_n this batch:{} {}'.format(STOI, PESQ, STOIn, PESQn))
    return STOI, PESQ, STOIn, PESQn


def compute_sdr(source, predict, mix):
    source = source.squeeze(1).data.cpu().numpy()
    predict = predict.squeeze(1).data.cpu().numpy()
    mix = mix.data.cpu().numpy()
    B = source.shape[0]
    SDR = []
    SDRn = []
    for i in range(B):
        if i == 24:
            x = 1
        source_idx = source[i,:]
        predict_idx = predict[i,:]
        mix_idx = mix[int(i/(config.mix_spk)),:]
        speech_metric = bss_eval_sources(source_idx, predict_idx)
        speech_metric_n = bss_eval_sources(source_idx, mix_idx)
        print(speech_metric)
        sdr = speech_metric[0].mean()
        sdrn = speech_metric_n[0].mean()
        SDR.append(sdr)
        SDRn.append(sdrn)
    SDR = np.array(SDR)
    SDRn = np.array(SDRn)
    SDR = SDR.mean()
    SDRn = SDRn.mean()
    print('SDR and SDRn this batch:{} and {}'.format(SDR, SDRn))
    return SDR, SDRn


def savewav(path, mix_wav, true_wav, predict_wav):
    predict_wav = predict_wav.squeeze(1).data.cpu().numpy()
    true_wav = true_wav.squeeze(1).data.cpu().numpy()
    mix_wav = mix_wav.data.cpu().numpy()
    print(mix_wav.shape, true_wav.shape, predict_wav.shape)
    BS = mix_wav.shape[0]
    for i in range(BS):
       # label = time.time()
       label = 1 
       sf.write(path+'{}_{}_mix.wav'.format(i,label), mix_wav[i,:], 16000)
       sf.write(path+'{}_{}_pre1.wav'.format(i,label), predict_wav[i*2]/np.max(predict_wav[i*2]), 16000)
       sf.write(path+'{}_{}_pre2.wav'.format(i,label), predict_wav[i*2+1]/np.max(predict_wav[i*2+1]), 16000)
       sf.write(path+'{}_{}_true1.wav'.format(i,label), true_wav[i*2], 16000)
       sf.write(path+'{}_{}_true2.wav'.format(i,label), true_wav[i*2+1], 16000)
    print('save wav completed!')
       

def inference(model, opt):
    model.eval()
    print("*" * 40 + 'test stage' + "*" * 40)
    test_data_gen = prepare_data('test', '2')
    SDR_SUM = np.array([])
    SDRn_SUM = np.array([])
    STOI_SUM = np.array([])
    PESQ_SUM = np.array([])
    STOIn_SUM = np.array([])
    PESQn_SUM = np.array([])
    loss_total = []
    while True:
        print('\n')
        test_data = test_data_gen.__next__()
        if test_data == False:
            break
        spk_num = 2 
        mixture = Variable(torch.from_numpy(test_data['mixture'].astype('float32'))).cuda()
        visual_features = Variable(torch.from_numpy(test_data['visual_features'].astype('float32'))).cuda()
        clean_audio = Variable(torch.from_numpy(test_data['clean_audio'].astype('float32'))).cuda() # [num_spk*batch_size, len]
        shape = clean_audio.shape
        clean_audio_len = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1])).cuda()
        clean_audio = clean_audio.unsqueeze(1)
        if config.inference:
            ######################################### Online streaming inference strategy ######################################################
            CACHE_audio = {}
            CACHE_visual = {} 
          
            for i in range(config.stack * config.layer):
                CACHE_audio[str(i)] = Variable(torch.from_numpy(np.zeros(shape=[spk_num*config.BATCH_SIZE, 512, 2*2**(i%(config.layer))], dtype=np.float32))).cuda() 
            for i in range(config.visual_stack * config.visual_layer):
                CACHE_visual[str(i)] = Variable(torch.from_numpy(np.zeros(shape=[spk_num*config.BATCH_SIZE, 512, 2*2**(i%(config.visual_layer))], dtype=np.float32))).cuda()
           
            chunk_size_audio = int(config.chunk_size / 1000 * config.FRAME_RATE)
            chunk_size_visual = int(config.chunk_size / 1000 * config.FRAME_RATE_VIDEO)
            
            chunk_num = int(mixture.shape[1] / chunk_size_audio)
          
            pad = Variable(torch.zeros(config.BATCH_SIZE, int(config.WIN_LEN/2)).type(mixture.type())).cuda()
            mix_wav = torch.cat((mixture, pad), dim=1)
            mix_wav_idx = Variable(torch.zeros(config.BATCH_SIZE, int(config.WIN_LEN/2))).type(mix_wav.type()).cuda()
            predict_wav = Variable(torch.from_numpy(np.zeros(shape=[spk_num*config.BATCH_SIZE, mix_wav.shape[1]], dtype=np.float32))).cuda()
            count = True 
            
            for idx in  range(chunk_num):
                mix_wav_idx = torch.cat((mix_wav_idx, mix_wav[:, idx*chunk_size_audio:(idx+1)*chunk_size_audio+int(config.WIN_LEN/2)]), dim=1)
                if count:
                    mix_wav_idx = mix_wav_idx[:, -(chunk_size_audio + config.WIN_LEN):]
                else:
                    mix_wav_idx = mix_wav_idx[:, -(chunk_size_audio + int(config.WIN_LEN/2)):]
                visual_features_idx = visual_features[:, idx*chunk_size_visual:(idx+1)*chunk_size_visual]
                predict_wav_idx, CACHE_audio, CACHE_visual = model(mix_wav_idx, visual_features_idx, spk_num, CACHE_audio, CACHE_visual)
                if count:
                    predict_wav_idx = predict_wav_idx[:, int(config.WIN_LEN/2):]
                predict_wav[:, idx*chunk_size_audio:(idx+1)*chunk_size_audio + int(config.WIN_LEN/2)] += predict_wav_idx
                count = False
            predict_wav = predict_wav[:, :-int(config.WIN_LEN/2)]
            #######################################################  END  ######################################################################
        else:
            predict_wav, _, _,_, _, _, _ = model(mixture, visual_features, spk_num, clean_audio)
        predict_wav = predict_wav.unsqueeze(1)
        loss = cal_sisnr_order_loss(clean_audio, predict_wav, clean_audio_len)
        loss_total += [loss.item()]
        print('loss:{}'.format(loss.item()))
        # sdr, sdrn = compute_sdr(clean_audio, predict_wav, mixture)
        sdr, sdrn = 0, 0
        if os.path.exists(opt.save_path):
            shutil.rmtree(opt.save_path)
        os.mkdir(opt.save_path)
        savewav(opt.save_path, mixture, clean_audio, predict_wav)
        STOI, PESQ, STOIn, PESQn = compute_metric(clean_audio, predict_wav, mixture)
        # STOI, PESQ, STOIn, PESQn = 0, 0, 0, 0
        SDR_SUM = np.append(SDR_SUM, sdr)
        SDRn_SUM = np.append(SDRn_SUM, sdrn)
        STOI_SUM = np.append(STOI_SUM, STOI)
        PESQ_SUM = np.append(PESQ_SUM, PESQ)
        STOIn_SUM = np.append(STOIn_SUM, STOIn)
        PESQn_SUM = np.append(PESQn_SUM, PESQn)
        print('SDR PESQ STOI', SDR_SUM.mean(), PESQ_SUM.mean(), STOI_SUM.mean())
    SDR_aver = SDR_SUM.mean()
    SDRn_aver = SDRn_SUM.mean()
    STOI_aver = STOI_SUM.mean()
    PESQ_aver = PESQ_SUM.mean() 
    STOIn_aver = STOIn_SUM.mean()
    PESQn_aver = PESQn_SUM.mean() 
    loss_aver = np.array(loss_total).mean()
    print('SDR{}, SDRn{}, PESQ{}, STOI{}, PESQn{}, STOIn{}' .format(SDR_aver, SDRn_aver, PESQ_aver, STOI_aver, PESQn_aver, STOIn_aver))
    print('test loss:{}' .format(loss_aver))
    print("*" * 40 + 'inference end' + "*" * 40)
    return loss_aver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ModelPath', type=str, default="./params_TF/params_BS32_fp8_vf64_cLN_sisnr_C/AVMS_AVSpeech_100.pth", help='the model we loaded')
    parser.add_argument('--gpus', type=int, default=2, help='number of gpu we use')
    parser.add_argument('--save_path', type=str, default="./test_audio_100/", help='the path of saving predicted audio')
    opt = parser.parse_args()
    model = AVMS(causal=config.causal, layer=config.layer, stack=config.stack).cuda()
    model = torch.nn.DataParallel(model, device_ids=range(opt.gpus))
    params_path = opt.ModelPath
    model.load_state_dict(torch.load(params_path)['state_dict'])
    print('Params:',params_path, 'loaded successfully!\n')
    with torch.no_grad():
        inference(model, opt)
