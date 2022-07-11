"""
This scripts have four functions:
1. filter videos (the length of video is less than 3 seconds and the number of faces in video 
   more than one) and record the name of correct videos;
2. video --> images and record the FPS of video in .txt file;
3. images --> face thumbnails, add variable control whether to save intermediate results (face thumbnails);
   reference: https://github.com/biubug6/Pytorch_Retinaface
4. face thumbnails --> visual features.
@zhangpeng
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import time

import network.FAN_feature_extractor as FAN_feature_extractor
from torch.autograd import Variable


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def video2images(path):
    """
    Function:
    video-->images;  
    Parameters:
    Input: path
    Return: images (or [] the length of video is less than 3 seconds), FPS
    """
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps == 0 or frames == 0:
        print("the video is error")
        return [], 0
    print("fps=", round(fps, 2), "frames=", int(frames), "lengths=", 1.0/fps*int(frames))
    images = []
    if 1.0/fps*int(frames) >= 3:
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            images.append(frame)
    else:
        print("the length of this video is less than 3 second")
    return images, round(fps, 2)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def detect_face(raw_imgs, net, video_name, device):
    """
    Function:
    1. detect face in images belong to a video;
    2. filter the video that the number of faces more than one;
    3. add variable control whether to save intermediate results (face thumbnails).
    Parameters:
    Input: raw images:raw_imgs, model:net, video name for saving face thumbnails
    Return: face thumbnails or [] (the numbe of faces in this video more than one)
    HyperParameters:
    1. save_face_thumbnails: control whether to save intermediate results;
    2. path_face_thumbnails: the path of folder for saving face thumbnails.
    """
    # detectting scale
    resize = 1
    # detecting begin
    face_thumbnails = []
    label_mt1 = False
    label_lt1 = False
    for i, img_raw in enumerate(raw_imgs):
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        loc, conf, landms = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        # scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        #                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        #                       img.shape[3], img.shape[2]])
        # scale1 = scale1.to(device)
        # landms = landms * scale1 / resize
        # landms = landms.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        # landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1] # max-->min
        boxes = boxes[order]
        # landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        # landms = landms[keep]
        # dets = np.concatenate((dets, landms), axis=1) # The final results: rectangle and landmarks
        # filter the video that the number of faces more than 1
        if dets.shape[0] == 0:
            label_lt1 = True
            break
        b = dets[0]
        b = list(map(int, b))
        b = np.maximum(b, 0) # prevent b[1] and b[0] < 0
        face_thumbnail = img_raw[b[1]:b[3],b[0]:b[2],:]
        # print('shape of face thumbnail: {}, b:{}'.format(face_thumbnail.shape, b))
        face_thumbnail = cv2.resize(face_thumbnail, (256, 256))
        face_thumbnails.append(face_thumbnail)
        if dets.shape[0] > 1:
            label_mt1 = True
            break
    if label_mt1:
        print('The number of faces in this video is more than one!')
    if label_lt1:
        print('There is no face in this video!')
    # save face thumbnails
    if not label_lt1 and not label_mt1 and args.save_face_thumbnails:
        dir_path = os.path.join(args.path_face_thumbnails, video_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for i, img_face in enumerate(face_thumbnails):
            img_face_path = os.path.join(dir_path, str(i)+'.jpg')
            cv2.imwrite(img_face_path, img_face)
    if not label_mt1 and not label_lt1:
        return face_thumbnails
    else:
        return []


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)
    return model


def load_checkpoint(resume_path, Model):
    resume_path = resume_path
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        Model = copy_state_dict(checkpoint['state_dict'], Model)
        return Model
    else:
        print("=> nocc checkpoint found at '{}'".format(resume_path))


def extract_visual_features(face_thumbnails, visual_encoder, device):
    """
    Function:
    Extract visual features.
    Parameters:
    Input: face_thumbnails;
    Return: Visual features
    """
    # TODO: parallel processing (batch)
    visual_features = []
    for i, img_face in enumerate(face_thumbnails):
        img_face_block = np.zeros((1, 256, 256, 3), dtype=np.float32)
        img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
        img_face = img_face.astype(np.float32)
        img_face /= 255
        img_face_block[0,:,:,:] = img_face[:, :, :]
        img_face_block = img_face_block.transpose(0, 3, 1, 2)
        img_face_block = torch.from_numpy(img_face_block)
        x = img_face_block.to(device)
        y = visual_encoder.forward(x)
        y = y.squeeze(0).squeeze(0)
        y = y.data.cpu().numpy()
        visual_features.append(y)
    visual_features = np.stack(visual_features, axis=0)
    return visual_features


if __name__ == "__main__":
    # configure Retinaface reference: https://github.com/biubug6/Pytorch_Retinaface
    parser = argparse.ArgumentParser(description='Retinaface and Visual encoder')
    parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--save_face_thumbnails', default=False, type=bool, help='Control whether to save intermediate results')
    parser.add_argument('--path_face_thumbnails', default='./source_video/test_face_thumbnails', type=str, help='Dir to save face thumbnails')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.8, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    # configure Visual encoder
    parser.add_argument('--trained_visual_encoder', default='./weights/visual_feature_encoder.pth',
                    type=str, help='Trained state_dict file path to open')
    parser.add_argument('--n_gpus', default=1, type=int, help='The number of gpus')
    # Source data path
    parser.add_argument('--s_data_path', default='./source_video/train', type=str, help='The path of source data (video)')
    parser.add_argument('--source_video_list', default='./source_video/train_list_2.txt', type=str, help='The path of source_video_list')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # Retinaface
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    # visual encoder
    visual_encoder = FAN_feature_extractor.FanFusion(args).to(device)
    visual_encoder = torch.nn.DataParallel(visual_encoder, device_ids=range(args.n_gpus))
    checkpoint_path = args.trained_visual_encoder
    print("=>loading checkpoints {} ......" .format(checkpoint_path))
    visual_encoder = load_checkpoint(checkpoint_path, visual_encoder)
    print("=>load checkpoint successfully")
    visual_encoder.eval()
    # begin to process source video
    print("begin to process source video......", '\n')
    source_video_path = args.s_data_path
    npy_dir = source_video_path + '_npy'
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)
    source_video_list = args.source_video_list
    samples_name = open(source_video_list, 'r').readlines()
    video_fps_txt = open(source_video_list[:-4] + '_fps_list.txt', 'w')
    for i, sample_name in enumerate(samples_name):
        sample_name = sample_name.strip('\n')
        if sample_name[-4:] != '.mp4':
            continue
        sample_path = os.path.join(source_video_path, sample_name)
        print("*"*15, "process video: {}, count: {}".format(sample_name, i), "*"*15)
        start = time.time()
        images, fps = video2images(sample_path)
        if images != []: # represents the lengths of the video is more than 3 seconds
            face_thumbnails = detect_face(images, net, sample_name, device)
            if face_thumbnails != []: # represents the number of face in this video is only one
                visual_features = extract_visual_features(face_thumbnails, visual_encoder, device)
                save_vf_path = os.path.join(npy_dir, sample_name[:-4] + '.npy')
                np.save(save_vf_path, visual_features)
                video_fps_txt.write(sample_name + "/" + str(fps) + '\n')
        end = time.time()
        cost_time = round(end - start, 2)
        print("*"*15, "process end, cost time: {} s ".format(cost_time), "*"*46, '\n')
    video_fps_txt.close()
