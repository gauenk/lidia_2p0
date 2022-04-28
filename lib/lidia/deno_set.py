
from .modules import *
from .utils import *
import warnings
import os.path
import argparse
# import matplotlib.pyplot as plt
import torch as th
import numpy as np
import torch.nn.functional as tnnf
from easydict import EasyDict as edict
from einops import rearrange

# -- load model info --
# from .nl_model_io import select_sigma,get_lidia_model,get_default_opt
from .model_io import select_sigma,get_lidia_model,get_default_opt

def denoise_ntire2020(noisy,sigma,pm_vid=None,flows=None,train=False):

    # -- types --
    nosiy = noisy.type(th.float32)

    # -- input params --
    assert flows is None
    if pm_vid is None: pm_vid = noisy

    # -- get model & info --
    nl_denoiser = get_lidia_model(noisy.device,noisy.shape,sigma)
    opt = get_default_opt(sigma)

    # -- denoise --
    with th.no_grad():
        deno_n = nl_denoiser(noisy, train=train, save_memory=False,
                             max_chunk=opt.max_chunk, srch_img=pm_vid,
                             srch_flows=flows)
    return deno_n

def denoise_npc(noisy,sigma,pm_basic):

    # -- get deno --
    nl_denoiser = get_lidia_model(noisy.device,noisy.shape,sigma)

    # -- adaptation config --
    opt = get_default_opt(sigma)
    total_pad = 20

    # -- exec adaptation --
    print("Adaptation.")
    nadapts = 2
    nframes = int(noisy.shape[0])
    for astep in range(nadapts):
        # for t in range(nframes):
            # print("LIDIA Frame [%d/%d]" % (t+1,nframes))
        noisy_t = (noisy/255. - 0.5)/0.5
        pm_basic_t = (pm_basic/255. - 0.5)/0.5
        pm_basic_t = pm_basic_t.clone()
        pm_basic_t = pm_basic_t.clamp(-1,1).cpu()
        pm_basic_t = rearrange(pm_basic_t,'t c h w -> t c h w')
        print(pm_basic_t.shape)
        # image_dn = process_image(nl_denoiser, noisy_t, max_chunk, pm_basic_t)
        nl_denoiser = adapt_net(nl_denoiser, opt, total_pad, pm_basic_t)

    # -- iterate over frames --
    deno = []
    avg_psnr = 0
    max_chunk = 40000
    nframes = int(noisy.shape[0])
    for t in range(nframes):
        print("LIDIA Frame [%d/%d]" % (t+1,nframes))
        noisy_t = (noisy[[t]]/255. - 0.5)/0.5
        pm_basic_t = (pm_basic[[t]] - 0.5)/0.5
        image_dn = process_image(nl_denoiser, noisy_t, max_chunk, pm_basic_t)
        image_dn = (image_dn /2 + 0.5)*255.
        deno.append(image_dn)
    deno = th.cat(deno)

    return deno
