import torch
from itertools import permutations
from torch.autograd import Variable
import numpy as np

# this is Keunwoo Choi's implementation of istft.
# https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e#file-istft-torch-py
EPS = 1e-8
def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True, length=None):
    """stft_matrix = (batch, freq, time, complex) 
    
    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2. 
    """
    assert normalized == False
    assert onesided == True
    assert window == "hann"
    assert center == True

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    
    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))

        ytmp = istft_window *  iffted
        y[:, sample:(sample+n_fft)] += ytmp
    
    y = y[:, n_fft//2:]
    
    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = torch.cat(y[:, :length], torch.zeros(y.shape[0], length - y.shape[1], device=y.device))
    
    coeff = n_fft/float(hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return Variable(mask).cuda()


# reference https://github.com/jonlu0602/DeepDenoisingAutoencoder/blob/master/python/utils.py


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    # source_lengths = torch.Tensor(source_lengths)
    # source_lengths = source_lengths.unsqueeze(0).expand(source.shape[0], len(source_lengths))
    B, C, T = source.size()
    # mask padding position along T
    # mask = get_mask(source, source_lengths)
    # estimate_source *= mask

    # Step 1. Zero-mean norm
    # num_samples = source_lengths.contiguous.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.mean(source, dim=2, keepdim=True)
    mean_estimate = torch.mean(estimate_source, dim=2, keepdim=True)
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    # zero_mean_target *= mask
    # zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    # perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    perms_one_hot = source.new_zeros((perms.size()[0],perms.size()[1], C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def cal_si_snr_with_order(source, estimate_source, source_lengths):
    """Calculate SI-SNR with given order.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with order
    # reshape to use broadcast
    s_target = zero_mean_target  # [B, C, T]
    s_estimate = zero_mean_estimate  # [B, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]
    #print(pair_wise_si_snr)
    return torch.sum(pair_wise_si_snr,dim=1)/C


def cal_sisnr_pit_loss(source, estimate_source, source_lengths):
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(max_snr)
    return loss


def cal_sisnr_order_loss(source, estimate_source, source_lengths):
    max_snr = cal_si_snr_with_order(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(max_snr)
    return loss

def uPIT_complex_loss(target, predict, loss_func):
    # target and predict's shape is [batch, num_k, T, F]
    min_loss = torch.min(loss_func(target, predict), loss_func(target[:,0,:,:], predict[:,1,:,:]) + loss_func(target[:,1,:,:], predict[:,0,:,:]))
    return min_loss 


def cal_si_mag_snr_with_order(source, estimate_source, source_lengths):
    """Calculate SI-SNR with given order.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with order
    # reshape to use broadcast
    s_target = zero_mean_target  # [B, C, T]
    s_estimate = zero_mean_estimate  # [B, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]
    #print(pair_wise_si_snr)
    return torch.sum(pair_wise_si_snr,dim=1)/C


def cal_mag_sisnr_order_loss(source, estimate_source, source_lengths):
    shape = source.shape
    source_unfold = torch.reshape(source, (shape[0], shape[1]*shape[2]))
    source_unfold_ = source_unfold.unsqueeze(1)
    estimate_source_unfold = torch.reshape(estimate_source, (shape[0], shape[1]*shape[2]))
    estimate_source_unfold_ = estimate_source_unfold.unsqueeze(1)
    source_lengths = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1]*shape[2])).cuda()
    max_snr = cal_si_snr_with_order(source_unfold_, estimate_source_unfold_, source_lengths)
    loss = 0 - torch.mean(max_snr)
    return loss


def cal_mag_sisnr_pit_loss(source, estimate_source, source_lengths):
    shape = source.shape
    source_unfold = torch.reshape(source, (shape[0], shape[1]*shape[2]))
    source_unfold_ = source_unfold.unsqueeze(1)
    estimate_source_unfold = torch.reshape(estimate_source, (shape[0], shape[1]*shape[2]))
    estimate_source_unfold_ = estimate_source_unfold.unsqueeze(1)
    source_lengths = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1]*shape[2])).cuda()
    max_snr, _, _ = cal_si_snr_with_pit(source_unfold_, estimate_source_unfold_, source_lengths)
    loss = 0 - torch.mean(max_snr)
    return loss