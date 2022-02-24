import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config
import module


# Audio-Visual Speech Separation Networks
class AVMS(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, visual_dim=config.VISUAL_DIM, sr=16000, layer=8, stack=3, 
                 kernel=3, causal=False, num_spk=2, skip=config.SKIP):
        super(AVMS, self).__init__()
        """
        encoder output dim: speech encoder dim
        TCN hidden dim: feature dim
        """
        # hyper parameters
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.win = int(config.WIN_LEN)
        self.stride = self.win // 2  
        self.layer = layer
        self.skip = skip
        self.stack = stack
        self.kernel = kernel
        self.causal = causal
        if not self.causal and config.type_visual_encoder == 'TCN':
            self.visual_dim = 2 * visual_dim
        else:
            self.visual_dim = visual_dim
        
        # speech encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        # LayerNorm
        if causal:
            if config.mode_LN == 'cLN':
                self.LN = module.cLN(self.enc_dim, eps=1e-8)
            elif config.mode_LN == 'BN': # BN is worse for multi-speakers training
                self.LN = nn.BatchNorm1d(self.enc_dim, eps=1e-08)
            else:
                self.LN = nn.LayerNorm(self.enc_dim, eps=1e-08)
        else:
            self.LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)
        # linear layer: transform the dimension of features
        self.BN = nn.Conv1d(self.enc_dim, self.feature_dim, 1)
        if config.type_visual_encoder == 'TCN':
            # visual encoder based TCN, causal: visual_dim=64, non-causal: visual_dim=128
            self.fcv = nn.Linear(256, self.visual_dim)
            self.visual_encoder = module.TCN_visual(self.visual_dim, self.visual_dim*4, config.visual_layer, config.visual_stack,
                skip=self.skip, causal=self.causal, dilated=True)
        else:
            # visual encoder based LSTM, causal: visual_dim=64, non-causal: visual_dim=128
            self.visual_encoder = nn.LSTM(256, self.visual_dim, num_layers=3, batch_first=True, bidirectional=(not self.causal))
        # audio-visual separator
        self.separator = module.TCN_audio(self.feature_dim, self.feature_dim*4, self.layer, self.stack, skip=self.skip,
                                causal=self.causal, dilated=True)
        # post-processing layers
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(self.feature_dim, self.enc_dim, 1))
        # speech decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest
        
    def forward(self, input, visual, num_spk, CACHE_audio=None, CACHE_visual=None):
        # padding, do not need padding when use streaming inference strategy 
        if not config.inference:
            output, rest = self.pad_signal(input)
        else:
            output = input.unsqueeze(1)
        # speech encoder & layer norm & linear layer
        enc_output = self.encoder(output)  # shape = [B, N, L]
        shape = enc_output.shape
        enc_output = enc_output.unsqueeze(1).expand(shape[0], num_spk, shape[1], shape[2]).contiguous().view(-1, shape[1], shape[2])
        if config.mode_LN == 'LN':
            enc_output = enc_output.transpose(1, 2)
            enc_output = self.LN(enc_output)
            enc_output = enc_output.transpose(1, 2)
            mix = self.BN(enc_output)
        else:
            mix = self.BN(self.LN(enc_output))
        if config.type_visual_encoder == 'TCN':
            # TCN-based visual encoder
            visual = self.fcv(visual)
            visual = visual.transpose(1, 2)
            # In this version, streaming inference strategy only use in TCN-based visual encoder
            visual, CACHE_visual = self.visual_encoder(visual, CACHE_visual)
            visual = visual.transpose(1, 2)
        else:
            # LSTM-based visual encoder
            self.visual_encoder.flatten_parameters()
            visual, _ = self.visual_encoder(visual)
        query = visual
        query = F.interpolate(query.transpose(1, 2), mix.shape[2], mode='nearest')
        # audio-visual separator
        if config.finetune:
            # freeze the weights of visual encoder to train separator only
            output, CACHE_audio = self.separator(mix, query.detach(), num_spk, CACHE_audio)
        else:
            output, CACHE_audio = self.separator(mix, query, num_spk, CACHE_audio)
        # predict the speech features of target speaker
        masks = torch.sigmoid(self.output(output))
        masked_output = enc_output * masks  # B, C, N, L
        # speech decoder
        output_wav = self.decoder(masked_output).squeeze(1)  # B*C, 1, L
        if not config.inference:
            output_wav = output_wav[:, self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        return output_wav, CACHE_audio, CACHE_visual