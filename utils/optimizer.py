# from lion_pytorch import Lion
from torch.optim import AdamW, Adam
import torch
import numpy as np


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
        params,
        lr=1e-4,
        wd=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        filter_by_requires_grad=False,
        group_wd_params=True,
        use_lion=False,
        **kwargs
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    # if use_lion:
    #     return Lion(params, lr=lr, betas=betas, weight_decay=wd)
    #
    # else:
    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, start_anneal=0, anneal_steps=1000000, anneal_rate=2, warmup_steps=4000, num_gpu=1):
        self.n_warmup_steps = warmup_steps
        self.start_anneal = start_anneal
        self.anneal_steps = anneal_steps
        self.anneal_rate = anneal_rate
        self.num_gpu = num_gpu
        print('num gpu: ', num_gpu)

    def get_lr_scale(self, epoch):
        epoch = epoch // self.num_gpu
        if epoch == 0:
            epoch = 1
        lr = np.min(
            [
                np.power(epoch, -0.5),
                np.power(self.n_warmup_steps, -1.5) * epoch,
            ]
        )
        if (epoch - self.start_anneal) // self.anneal_steps > 0:
            lr = lr * np.power(self.anneal_rate, (epoch - self.start_anneal) // self.anneal_steps)
        return lr
