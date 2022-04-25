
"""
A non-local denosing loop using lidia as the denoiser
"""

# -- python deps --
from tqdm import tqdm
import copy,math
import torch
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

import torchvision.utils as tvu

# -- imports --
import lidia.agg as agg
import lidia.utils as utils
import lidia.alloc_dnls as alloc
# import lidia.alloc as alloc
import lidia.search_mask as search_mask
import lidia.search as search
import lidia.proc_nl as proc_nl
import lidia.deno as deno
import lidia.utils as utils
from lidia.params import get_params,get_args


# -- project imports --
# from lidia.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp
from lidia.utils import optional
from lidia.data import save_burst
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- create model --
from lidia.nl_model_io import get_lidia_patch_model,get_default_opt

# -- for testing [okay to delete me] --
from torchvision.transforms.functional import pad as pad_fxn
from torchvision.transforms.functional import center_crop


def denoise_nl(noisy, sigma, pm_vid=None, flows=None, gpuid=0, clean=None, verbose=True):
    """
    Lightweight Image Denoising with Instance Adaptation (LIDIA)

    """

    # -- get device --
    use_gpu = th.cuda.is_available() and gpuid >= 0
    device = 'cuda:%d' % gpuid if use_gpu else 'cpu'

    # -- model --

    # -- to tensor --
    if not th.is_tensor(noisy):
        noisy = th.from_numpy(noisy).to(device)

    # -- setup lidia inputs --
    t,c,h,w = noisy.shape
    params = get_params(sigma,verbose,"default")
    flows = alloc.allocate_flows(flows,noisy.shape,noisy.device)
    if not(clean is None):
        params.srch_img = ["clean","clean"]

    # -- args --
    t,c,h,w = noisy.shape
    args = get_args(params,t,c,0,noisy.device)

    # -- prepare image --
    pad = 2#args.ps//2
    # noisy = pad_fxn(noisy,(pad,pad,pad,pad),padding_mode="constant")
    # print("noisy.shape: ",noisy.shape)

    # -- allocs and args --
    images = alloc.allocate_images(noisy,None,clean)

    # -- get model --
    args.lidia_model = get_lidia_patch_model(device,noisy.shape,sigma)
    # args.deno = "bayes"
    args.deno = "lidia"
    args.bsize = 4624*5#4096*5
    args.rand_mask = False
    args.chnls = 1
    args.dilation = 1

    # -- exec non-local step --
    proc_nl.exec_nl_step(images,flows,args)
    deno = images['deno'].clone()

    # -- center crop --
    # deno = center_crop(deno,(h,w))

    return deno
