
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


def select_sigma(sigma):
    sigmas = np.array([15, 25, 50])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def get_nl_deno(device,sigma):
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
    return nl_denoiser

def get_default_opt(sigma):
    opt = edict()
    opt.sigma = sigma
    opt.seed = 123
    opt.max_chunk = 40000
    opt.block_w = 16
    opt.lr = 1e-3
    opt.epoch_num = 5
    opt.epochs_between_check = 5
    opt.dset_stride = 1
    opt.train_batch_size = 4
    opt.cuda_retrain = True
    return opt

def denoise_ntire2020(noisy,sigma,pm_vid=None,flows=None):

    # -- types --
    nosiy = noisy.type(th.float32)

    # -- input params --
    assert flows is None
    if pm_vid is None: pm_vid = noisy

    # -- get model & info --
    nl_denoiser = get_nl_deno(noisy.device,sigma)
    opt = get_default_opt(sigma)

    # -- denoise --
    with th.no_grad():
        deno_n = nl_denoiser(noisy, train=False, save_memory=True,
                             max_chunk=opt.max_chunk, srch_img=pm_vid, srch_flows=flows)
    return deno_n

def denoise_npc(noisy,sigma,pm_basic):

    # -- get deno --
    nl_denoiser = get_nl_deno(noisy.device,sigma)

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
