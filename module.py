import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import config

"""
we use skip-connection in TCN block by default
"""

class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        batch_size, channel, time_step = input.size(0), input.size(1), input.size(2)
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
    
            
class DepthConv1d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()
        self.causal = causal
        self.skip = skip
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        if config.inference:
            self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
            groups=hidden_channel, padding=0) # the key point exist in padding method
        else:
            self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
            groups=hidden_channel, padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        # now, cLN, BN and LN all can meet streaming inference strategy, but BN and LN are simple
        print("normalization type: {},".format(config.mode_LN))
        if config.mode_LN == 'cLN':
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        elif config.mode_LN == 'BN':
            self.reg1 = nn.BatchNorm1d(hidden_channel, eps=1e-08)
            self.reg2 = nn.BatchNorm1d(hidden_channel, eps=1e-08)
        elif config.mode_LN == 'gLN':
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.LayerNorm(hidden_channel, eps=1e-08)
            self.reg2 = nn.LayerNorm(hidden_channel, eps=1e-08)
        # skip-path
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input, cache_last=None):
        if config.mode_LN == 'LN':
            output = self.nonlinearity1(self.conv1d(input))
            output = output.transpose(1, 2)
            output = self.reg1(output)
            output = output.transpose(1, 2)
        else:
            output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if config.inference:
            cache_next = output
            output = torch.cat((cache_last, output, cache_last), dim=2)
        if self.causal:
            if config.mode_LN == 'LN':
                output = self.nonlinearity2(self.dconv1d(output)[:,:,:-self.padding])
                output = output.transpose(1, 2)
                output = self.reg2(output)
                output = output.transpose(1, 2)
            else:
                output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:,:,:-self.padding]))
        else:
            if config.mode_LN == 'LN':
                output = self.nonlinearity2(self.dconv1d(output))
                output = output.transpose(1, 2)
                output = self.reg2(output)
                output = output.transpose(1, 2)
            else:
                output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            if config.inference:
                return residual, skip, cache_next
            else:
                return residual, skip
        else:
            if config.inference:
                return residual, cache_next
            else:
                return residual

        
class TCN_audio(nn.Module):
    def __init__(self, BN_dim, hidden_dim,
                 layer, stack, num_spk=2, kernel=3, skip=True, 
                 causal=True, dilated=True):
        super(TCN_audio, self).__init__()
        # the input is a sequence of features of shape (B, N, L)
        self.receptive_field = 0
        self.dilated = dilated
        self.layer = layer
        self.stack = stack
        self.skip = skip
        self.causal = causal
        if not self.causal:
            self.visual_dim = config.VISUAL_DIM * 2
        else:
            self.visual_dim = config.VISUAL_DIM
        # the method of deep concatenate fusion (DCF) use target and other speaker's visual information 
        if config.MODAL_FUSION == 'DCF':
            self.fc = nn.Linear(128 + num_spk * self.visual_dim, 128, bias=True)
        # the method of concatenate fusion (CF) use target speaker's visual information only
        if config.MODAL_FUSION == 'CF':
            self.fc = nn.Linear(BN_dim + self.visual_dim, BN_dim, bias=True) 
        self.TCN = nn.ModuleList([])
        # TCN module that be used to process audio
        base = 2
        for s in range(self.stack):
            for i in range(self.layer):
                if self.dilated:
                    if config.low_latency and (s == 0 and i in [0, 1, 2]):
                        print('use low_latency model in audio path')
                        self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=base**i, padding=base**i, skip=self.skip, causal=False))
                    else: 
                        self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=base**i, padding=base**i, skip=self.skip, causal=self.causal)) 
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=self.skip, causal=self.causal))
                if i == 0 and s == 0:
                    self.receptive_field = self.receptive_field + kernel
                else:
                    if self.dilated:
                        self.receptive_field = self.receptive_field + (kernel - 1) * base**i
                    else:
                        self.receptive_field = self.receptive_field + (kernel - 1)                  
        print("Receptive field in audio path: {:3d} frames.".format(self.receptive_field))

    def forward(self, input, query, num_spk, CACHE_audio=None):
        # input shape: (B, N, L)
        #  select the multi-modal fusion position
        fusion_position = [config.FUSION_POSITION]
        output = input
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                if str(i) in fusion_position:
                    if config.MODAL_FUSION == 'CF':
                        output = torch.cat((output, query), dim=1)
                        output = output.transpose(1, 2)
                        output = self.fc(output)
                        output = output.transpose(1, 2)
                    if config.MODAL_FUSION == 'DCF':
                        shape = output.shape
                        output = output.view(-1, num_spk, shape[1], shape[2])
                        qshape = query.shape
                        query = query.view(-1, num_spk, qshape[1], qshape[2]) # shape:[B,num_spk,N,L]
                        if num_spk == 1:
                            output = torch.cat((output, query), dim=2)
                            output = output.view(-1, shape[1] + qshape[1], shape[2])
                        if num_spk == 2:
                            output0 = torch.cat((output[:,0,:,:],query[:,0,:,:],query[:,1,:,:]), dim=1)
                            output1 = torch.cat((output[:,1,:,:],query[:,1,:,:],query[:,0,:,:]), dim=1)
                            output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1)), dim=1)
                            output = output.view(-1, shape[1] + 2*qshape[1], shape[2])
                        if num_spk == 3:
                            output0 = torch.cat((output[:,0,:,:],query[:,0,:,:],query[:,1,:,:]+query[:,2,:,:]), dim=1)
                            output1 = torch.cat((output[:,1,:,:],query[:,1,:,:],query[:,0,:,:]+query[:,2,:,:]), dim=1)
                            output2 = torch.cat((output[:,2,:,:],query[:,2,:,:],query[:,0,:,:]+query[:,1,:,:]), dim=1)
                            output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1), output2.unsqueeze(1)), dim=1)
                            output = output.view(-1, shape[1] + 2*qshape[1], shape[2])
                        output = output.transpose(1, 2)
                        output = self.fc(output)
                        output = output.transpose(1, 2)
                if self.causal and config.inference:
                    residual, skip, cache = self.TCN[i](output, CACHE_audio[str(i)])
                    CACHE_audio[str(i)] = torch.cat((CACHE_audio[str(i)], cache[:,:,-2*2**(i%self.layer):]), dim=2)[:,:,-2*2**(i%self.layer):]
                else:
                    residual, skip = self.TCN[i](output) # causal=False, causal=True and inference=False 
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                if str(i) in fusion_position:
                    if config.MODAL_FUSION == 'CF':
                        output = torch.cat((output, query), dim=1)
                        output = output.transpose(1, 2)
                        output = self.fc(output)
                        output = output.transpose(1, 2)
                    if config.MODAL_FUSION == 'DCF':
                        shape = output.shape
                        output = output.view(-1, num_spk, shape[1], shape[2])
                        qshape = query.shape
                        query = query.view(-1, num_spk, qshape[1], qshape[2]) # shape:[B,num_spk,N,L]
                        if num_spk == 1:
                            output = torch.cat((output, query), dim=2)
                            output = output.view(-1, shape[1] + qshape[1], shape[2])
                        if num_spk == 2:
                            output0 = torch.cat((output[:,0,:,:],query[:,0,:,:],query[:,1,:,:]), dim=1)
                            output1 = torch.cat((output[:,1,:,:],query[:,1,:,:],query[:,0,:,:]), dim=1)
                            output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1)), dim=1)
                            output = output.view(-1, shape[1] + 2*qshape[1], shape[2])
                        if num_spk == 3:
                            output0 = torch.cat((output[:,0,:,:],query[:,0,:,:],query[:,1,:,:]+query[:,2,:,:]), dim=1)
                            output1 = torch.cat((output[:,1,:,:],query[:,1,:,:],query[:,0,:,:]+query[:,2,:,:]), dim=1)
                            output2 = torch.cat((output[:,2,:,:],query[:,2,:,:],query[:,0,:,:]+query[:,1,:,:]), dim=1)
                            output = torch.cat((output0.unsqueeze(1), output1.unsqueeze(1), output2.unsqueeze(1)), dim=1)
                            output = output.view(-1, shape[1] + 2*qshape[1], shape[2])
                        output = output.transpose(1, 2)
                        output = self.fc(output)
                        output = output.transpose(1, 2)
                if self.causal and config.inference:
                    residual, cache = self.TCN[i](output, CACHE_audio[str(i)])
                    CACHE_audio[str(i)] = torch.cat((CACHE_audio[str(i)], cache[:,:,-2*2**(i%self.layer):]), dim=2)[:,:,-2*2**(i%self.layer):]
                else:
                    residual = self.TCN[i](output) # causal=False, causal=True and inference=False
                output = output + residual 
        # output layer
        if self.skip:
            output = skip_connection 
        else:
            output = output
        return output, CACHE_audio


class TCN_visual(nn.Module):
    def __init__(self, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, 
                 causal=False, dilated=True):
        super(TCN_visual, self).__init__()
        # TCN module for processing visual features
        self.receptive_field = 0
        self.dilated = dilated
        self.TCN = nn.ModuleList([])
        self.layer = layer
        self.stack = stack
        self.causal = causal
        self.skip = skip
        base = 2
        for s in range(self.stack):
            for i in range(self.layer):
                if self.dilated:
                    if config.low_latency and (s == 0 and i == 1):
                        print('use low_latency model in visual path')
                        self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=base**i, padding=base**i, skip=self.skip, causal=False))
                    else:
                        self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=base**i, padding=base**i, skip=self.skip, causal=causal))
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=self.skip, causal=causal))   
                if i == 0 and s == 0:
                    self.receptive_field = self.receptive_field + kernel
                else:
                    if self.dilated:
                        self.receptive_field = self.receptive_field + (kernel - 1) * base**i
                    else:
                        self.receptive_field = self.receptive_field + (kernel - 1)            
        print("Receptive field in visual path: {:3d} frames.".format(self.receptive_field))
        
    def forward(self, input, CACHE_visual=None):
        # input shape: (B, N, L)
        output = input
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                if self.causal and config.inference:
                    residual, skip, cache = self.TCN[i](output, CACHE_visual[str(i)])
                    CACHE_visual[str(i)] = torch.cat((CACHE_visual[str(i)], cache[:,:,-2*2**(i%self.layer):]), dim=2)[:,:,-2*2**(i%self.layer):]
                else:
                    residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                if self.causal and config.inference:
                    residual, cache = self.TCN[i](output, CACHE_visual[str(i)])
                    CACHE_visual[str(i)] = torch.cat((CACHE_visual[str(i)], cache[:,:,-2*2**(i%self.layer):]), dim=2)[:,:,-2*2**(i%self.layer):]
                else:
                    residual = self.TCN[i](output)
                output = output + residual
        # output layer
        if self.skip:
            output = skip_connection
        else:
            output = output
        return output, CACHE_visual