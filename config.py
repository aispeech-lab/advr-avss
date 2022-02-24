# coding:utf-8

"""
    Configuration File
"""

MODULE = 'TCN' # use TCN-based model
DATASET = 'AVSpeech'
aim_path='./data' # AVMS as the root
MAX_EPOCH = 10000
BATCH_SIZE = 32  
BATCH_SIZE_TEST = 32
num_steps_per_epoch = 3000
num_samples_per_epoch = int(num_steps_per_epoch * BATCH_SIZE)
FRAME_RATE = 16000
SHUFFLE_BATCH = True
dB = 5
MAX_LEN = 3 
MAX_LEN_SPEECH = int(FRAME_RATE*MAX_LEN)
mix_spk = 2
VIDEO_RATE = 25
MAX_LEN_VIDEO = int(MAX_LEN*VIDEO_RATE)
DATA_AUG = True
finetune = False
type_visual_encoder = 'TCN' # TCN or LSTM
inference = False

# The parameters for low-latency model
causal = False
mode_LN  = 'gLN' # cLN, BN, LN (only in causal settings), gLN (non-causal)

# Parameters for SS model
WIN_LEN = 16
layer = 8
stack = 3
visual_layer = 4
visual_stack = 1
MODAL_FUSION = 'CF' # CF or DCF
FUSION_POSITION = '8' # '0','8','16'
VISUAL_DIM = 64
SKIP = True

# Loss
loss_type = 'sisnr'

low_latency = False