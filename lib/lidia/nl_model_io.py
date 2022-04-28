

# -- from neighbors --
from .nl_modules import *
from .utils import *
from .utils.model_info import select_sigma,get_default_opt

# -- misc --
import warnings
import os.path
import argparse

# -- data --
from easydict import EasyDict as edict

# -- nnf --
import torch.nn.functional as tnnf

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange


def get_lidia_model(device,im_shape,sigma):

    # -- select noise --
    lidia_sigma = select_sigma(sigma)

    # -- load arch --
    arch_opt = ArchitectureOptions(rgb=True, small_network=False)
    pad_offs, total_pad = calc_padding(arch_opt)
    nl_denoiser = NonLocalDenoiser(pad_offs, arch_opt).to(device)
    criterion = nn.MSELoss(reduction='mean')
    nl_denoiser.cuda()

    # -- select device --
    device = next(nl_denoiser.parameters()).device
    state_file_name0 = '/home/gauenk/Documents/packages/lidia/lidia-deno/models/model_state_sigma_{}_c.pt'.format(lidia_sigma)
    assert os.path.isfile(state_file_name0)
    model_state0 = torch.load(state_file_name0)
    nl_denoiser.patch_denoise_net.load_state_dict(model_state0['state_dict'])

    # -- kent's add --
    nl_denoiser.im_shape = list(im_shape)

    return nl_denoiser

def get_lidia_patch_model(device,im_shape,sigma):

    # -- load standard model --
    model = get_lidia_model(device,im_shape,sigma)

    # -- append with wrap function --
    im_params = edict()
    im_params.pixels_h = int(im_shape[-2])
    im_params.pixels_w = int(im_shape[-1])
    save_memory = False
    max_batch = 40000
    def patches_fwd(model,patches_0,dists_0,inds_0,patches_1,dists_1,inds_1):
        pdn = model.patch_denoise_net
        patches_0 = patches_0
        patches_1 = patches_1
        dists_0 = dists_0
        dists_1 = dists_1
        pdeno, pweights = pdn(patches_0, dists_0, inds_0,
                              patches_1, dists_1, inds_1,
                              im_params, im_params, save_memory, max_batch)
        print("pdeno.shape: ",pdeno.shape)
        pdeno = rearrange(pdeno,'t r (c h w) -> (t r) 1 1 c h w',h=5,w=5)
        deno = model.final_agg(pdeno,pweights,dists_0,inds_0,im_shape)
        return deno

    model.forward = patches_fwd

    return model

