# coding:utf-8

"""
train.py is used to train or eval our AVMS model.
@zhangpeng
"""

import sys
import time
import torch
from torch import nn 
from torch.autograd import Variable
import numpy as np 
import random 
import config
from dataloader import prepare_data
from logger import Logger 
import os 
import shutil 
from model import AVMS
import argparse
from loss import cal_sisnr_order_loss
from utils import Adjust_lr, Early_stop
from torch.optim.lr_scheduler import ReduceLROnPlateau


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    

def test(model, logger, step):
    model.eval()
    print("*" * 40 + 'test stage' + "*" * 40)
    test_data_gen = prepare_data('test', '2')
    loss_total = []
    while True:
        test_data = test_data_gen.__next__()
        spk_num = 2
        if test_data == False:
            break
        mixture = Variable(torch.from_numpy(test_data['mixture'].astype('float32'))).cuda()
        visual_features = Variable(torch.from_numpy(test_data['visual_features'].astype('float32'))).cuda()
        clean_audio = Variable(torch.from_numpy(test_data['clean_audio'].astype('float32'))).cuda() # [num_spk*batch_size, len]
        shape = clean_audio.shape
        clean_audio_len = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1])).cuda()
        clean_audio = clean_audio.unsqueeze(1)
        predict_wav, _, _ = model(mixture, visual_features, spk_num)
        predict_wav = predict_wav.unsqueeze(1)
        loss = cal_sisnr_order_loss(clean_audio, predict_wav, clean_audio_len)
        loss_total += [loss.item()]
        print('loss:{}'.format(loss.item()))
    loss_aver = np.array(loss_total).mean()
    logger.log_test(loss_aver, step)
    print('test loss:{} in scenario: {}' .format(loss_aver, '2'))
    print("*" * 40 + 'eval end' + "*" * 40)
    return loss_aver


def train(epoch_idx, optimizer, init_lr, step, model, opt, logger):
    model.train()
    batch_idx = 0
    print('Generate training data ......')
    strat_time = time.time()
    # train_data_gen_1 = prepare_data('train', '1') # 1 speaker + noise
    train_data_gen_2 = prepare_data('train', '2') # 2 speakers
    # train_data_gen_3 = prepare_data('train', '3') # 2 speakers + noise
    end_time = time.time()
    print('The total time for generate train data is {} s'.format(round(end_time - strat_time, 2)))
    # stop_label = []
    while True:
        print("*" * 40, epoch_idx, batch_idx, "*" * 40)
        # train the model under multi-scenarios: 1 speaker + noise, 2 speakers, 2 speakers + noise
        """
        have_label = 0
        if '1' in stop_label and '2' in stop_label and '3' in stop_label:
            break
        p = random.random()
        if p <= 0.3 and p > 0 and '1' not in stop_label:
            train_data = train_data_gen_1.__next__()
            have_label = 1
            spk_num = 1
            print('The number of speakers in this batch is {}, scenario:{}'.format(spk_num, '1'))
            if train_data == False:
                stop_label.append('1')
                continue
        if p <= 0.7 and p > 0.3 and '2' not in stop_label:
            train_data = train_data_gen_2.__next__()
            have_label = 1
            spk_num = 2
            print('The number of speakers in this batch is {}, scenario:{}'.format(spk_num, '2'))
            if train_data == False:
                stop_label.append('2')
                continue
        if p <= 1 and p > 0.7 and '3' not in stop_label: 
            train_data = train_data_gen_3.__next__()
            have_label = 1
            spk_num = 2
            print('The number of speakers in this batch is {}, scenario:{}'.format(spk_num, '3'))
            if train_data == False:
                stop_label.append('3')
                continue
        if have_label == 0:
            continue
        """
        # train the model under 2 speakers scenario
        train_data = train_data_gen_2.__next__()
        spk_num = 2
        if train_data == False:
            break
        mixture = Variable(torch.from_numpy(train_data['mixture'].astype('float32'))).cuda()
        visual_features = Variable(torch.from_numpy(train_data['visual_features'].astype('float32'))).cuda()
        clean_audio = Variable(torch.from_numpy(train_data['clean_audio'].astype('float32'))).cuda() # [spk_num * batch_size, length]
        shape = clean_audio.shape
        clean_audio_len = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1])).cuda()
        clean_audio = clean_audio.unsqueeze(1)
        predict_audio, _, _ = model(mixture, visual_features, spk_num)
        predict_audio = predict_audio.unsqueeze(1)
        loss = cal_sisnr_order_loss(clean_audio, predict_audio, clean_audio_len)
        logger.log_train(loss.item(), step)
        optimizer.zero_grad()
        loss.backward()
        w_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        print('loss:{} grad_norm:{}'.format(loss.item(), w_norm))
        optimizer.step()
        batch_idx += 1
        step += 1
    return step


def main(opt):
    print("*" * 80 + '\n' + 'Build Audio-Visual Speech Separation Model.' + '\n' + "*" * 80)
    if config.DATA_AUG == True:
        print('Generate the mixture for training online (on the fly).')
    model = AVMS(causal=config.causal, layer=config.layer, stack=config.stack).cuda() # define the model
    print(model)
    model = nn.DataParallel(model, device_ids=range(opt.gpus)) # parallel setting
    # compute the total number of parameters of the SS model
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('Parameters size in the SS model is {} M'.format(param_count / 1e6))
    # the directory for saving pretrained model & the setting of tensorboad
    if not os.path.exists(opt.ParamsPath):
        os.mkdir(opt.ParamsPath)
    if os.path.exists(opt.TensorboardPath) and opt.ModelPath == None:
        shutil.rmtree(opt.TensorboardPath)
        os.mkdir(opt.TensorboardPath)
    if not os.path.exists(opt.TensorboardPath):
        os.mkdir(opt.TensorboardPath)
    logger = Logger(opt.TensorboardPath)
    # set the intial learning rate & optimizer & learning rate decay strategy
    init_lr = opt.lr
    print("The initial learning rate is {}".format(init_lr))
    optimizer = torch.optim.Adam([{'params':model.parameters()}], lr=init_lr)
    # lr_decay = Adjust_lr(init_lr, opt.lr_patient, optimizer) # learning rate decay strategy
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stop = Early_stop(opt.es_patient) # early stop strategy
    # the initialization of epoch and step
    epoch_idx = 0
    step = 0
    # load pretrained model for finetune (checkpoints, epoch, optimizer's parameters)
    if opt.ModelPath is not None:
        params_path = opt.ModelPath
        model.load_state_dict(torch.load(params_path)['state_dict'])
        epoch_idx = torch.load(params_path)['epoch_idx']
        optimizer.load_state_dict(torch.load(params_path)['optimizer_state'])
        scheduler.load_state_dict(torch.load(params_path)["scheduler"])
        step = int(config.num_samples_per_epoch / config.BATCH_SIZE) * epoch_idx
        print('Model:', params_path, 'load successfully', '\n')
    print('*' * 80 + '\n' + 'Begin to train the model.' + '\n' + '*' * 80)
    while True:
        step = train(epoch_idx, optimizer, init_lr, step, model, opt, logger)
        epoch_idx += 1
        # save model's parameters
        if epoch_idx >= 1 and epoch_idx % opt.save_epoch == 0:
            print("save model and optimizer state at iteration {} to {}/AVMS_{}_{}".format(
                epoch_idx, opt.ParamsPath, config.DATASET, epoch_idx))
            torch.save(
                    {'state_dict': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch_idx': epoch_idx
                    }, '{}/AVMS_{}_{}.pth'.format(opt.ParamsPath, config.DATASET, epoch_idx))
        # evaluation the model
        if 1 and epoch_idx >= 1 and epoch_idx % opt.eval_epoch == 0:
            with torch.no_grad():
                loss_test = test(model, logger, step)
        # decay the learning rate
        scheduler.step(loss_test)
        # early-stop
        key_value = early_stop.step(loss_test)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.log_lr(lr, epoch_idx)
        print('this epoch {} learning rate is {}' .format(epoch_idx, lr))
        if key_value == 0:
            print("End the whole train process in {} epoch!" .format(epoch_idx))
            sys.exit(0)
        # when the epoch equal to setting max epoch, end!
        if epoch_idx == config.MAX_EPOCH:
            sys.exit(0)


if __name__ == "__main__":
    ############### training settings ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus we demand')
    parser.add_argument('--TensorboardPath', type=str, default='./log/tb_test', help='path of saving tensorboard data')
    parser.add_argument('--ParamsPath', type=str, default='./params/params_test', help='path of saving pretrained model')
    parser.add_argument('--ModelPath', type=str, default=None, help='path of loading pretrained model')
    parser.add_argument('--eval_epoch', type=int, default=1, help='num of epochs to eval model')
    parser.add_argument('--save_epoch', type=int, default=1, help='num of epochs to save model')
    parser.add_argument('--seed', type=int, default=1, help='the seed')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_patient', type=int, default=10000, help='patient for lr decay')
    parser.add_argument('--es_patient', type=int, default=10000, help='patient for early stop')
    opt = parser.parse_args()
    # fix the random seed
    random.seed(opt.seed) # python
    np.random.seed(opt.seed) # numpy
    torch.manual_seed(opt.seed) # cpu
    torch.cuda.manual_seed(opt.seed) # current gpu
    torch.cuda.manual_seed_all(opt.seed) # all gpus
    print('seed now {}' .format(opt.seed))
    # set cudnn's parameters
    torch.backends.cudnn.benchmark = False # True can greatly improve the running speed of CNN
    torch.backends.cudnn.deterministic = True
    main(opt)