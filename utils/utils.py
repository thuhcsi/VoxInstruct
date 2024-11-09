import yaml
import matplotlib
import torch
import numpy as np

matplotlib.use("Agg")
import matplotlib.pylab as plt

import torchaudio
from torch.utils.data import Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import random
import math
from typing import Optional
import torch.nn.functional as F
from tqdm import tqdm
from scipy.io.wavfile import write


# [B, T1, T2]
def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    return


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def cycle(dl):
    while True:
        for data in tqdm(dl):
            yield data


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
    return


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def export_tensor(data, name):
    print("export {} shape {}".format(name, data.shape))
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    np.save(name, data)
    WriteMatrixToBinary("{}.bin".format(name), data)


def set_random_seed(seed=123):
    """Set random seed manully to get deterministic results"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_config_from_file(file):
    with open(file, 'r') as f:
        hp = yaml.load(f,Loader=yaml.FullLoader)
    hp = HParams(**hp)
    return hp


def calculate_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {}, size of saving: {}M".format(
        pytorch_total_params,
        pytorch_total_params*4/1024/1024))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params: {}, size of saving: {}M".format(
        pytorch_total_params,
        pytorch_total_params*4/1024/1024))
    return


def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask


def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensors_to_device.append(tensor.to(device, non_blocking=True))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def compute_loss(logits, target, mask, compute_acc=False, topk=(1,)):

    logits = logits.to(torch.float32).contiguous()
    target = target.to(torch.long).contiguous()
    mask = mask.to(torch.float32).contiguous()

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)

    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size()) * mask
    # mask: (batch, max_len)
    loss = losses.sum() / mask.sum()

    if compute_acc:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            acc = []
            maxk = max(topk)
            _, pred = log_probs_flat.topk(maxk, 1, True, True)
            correct = pred.eq(target_flat.expand_as(pred))
            correct = correct * (mask.reshape(-1, 1).expand_as(pred))
            for k in topk:
                correct_k = correct[:, :k].sum() / mask.sum()
                acc.append(correct_k)
        return loss, acc
    
    return loss


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype('int16')
    write(output_file, sr, audio)
    return


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-1e5, temperature=1.0):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    logits = logits / temperature
    return logits



