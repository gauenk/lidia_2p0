import numpy as np
from easydict import EasyDict as edict

def select_sigma(sigma):
    sigmas = np.array([15, 25, 50])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]


def get_default_opt(sigma):
    opt = edict()
    opt.sigma = sigma
    opt.seed = 123
    opt.max_chunk = 40000
    opt.block_w = 64
    opt.lr = 1e-2
    opt.epoch_num = 10
    opt.epochs_between_check = 5
    opt.dset_stride = 32
    opt.train_batch_size = 4
    opt.cuda_retrain = True
    return opt



