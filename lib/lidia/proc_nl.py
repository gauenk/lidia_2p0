

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

# -- package --
import vnlb
from vnlb.testing.file_io import save_images

# -- batching, scatter, search, and gather --
import dnls

# -- imports --
import lidia.agg as agg
import lidia.utils as utils
import lidia.alloc_dnls as alloc
import lidia.search_mask as search_mask
import lidia.search as search
import lidia.deno as deno
import lidia.utils as utils
from lidia.utils import update_flat_patch

# -- project imports --
# from lidia.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp
from lidia.data import save_burst
import pprint
pp = pprint.PrettyPrinter(indent=4)


def exec_nl_step(images,flows,args):

    # -- init --
    # pp.pprint(args)

    # -- create access mask --
    mask,ngroups = search_mask.init_mask(images.shape,args)
    mask[...] = 1.
    mask2 = mask.clone()
    masks = [mask,mask2]

    # -- param --
    search_dilations = [1,2]
    assert len(search_dilations) == args.nlevels

    # -- allocate memory --
    patches = alloc.allocate_patches(args.patch_shape,images.clean,
                                     args.device,args.nlevels)
    bufs = alloc.allocate_bufs(args.bufs_shape,args.device,args.nlevels)

    # -- batching params --
    nelems,nbatches = utils.batching.batch_params(mask,args.bsize,args.nstreams)
    cmasked_prev = nelems

    # -- color xform --
    # utils.color.rgb2yuv_images(images)

    # -- logging --
    if args.verbose: print(f"Processing LIDIA [step {args.step}]")

    # -- over batches --
    if args.verbose: pbar = tqdm(total=nelems)
    for batch in range(nbatches):

        # -- exec search --
        done = False
        for level in range(args.nlevels):
            key = patches.levels[level]
            args.dilation = search_dilations[level]
            mask_l = masks[level]
            done = search.exec_search(patches[key],images,flows,mask_l,bufs[key],args)

        # -- refinemenent the searching --
        # search.exec_refinement(patches,bufs,args.sigma)

        # -- flat patches --
        update_flat_patch(patches,args)

        # -- valid patches --
        vpatches = get_valid_patches(patches,bufs)
        vbufs = get_valid_bufs(bufs)
        if vpatches.shape[0] == 0:
            break

        # -- denoise patches --
        deno.denoise(vpatches,vbufs,args,args.deno,images.means)

        # -- fill valid --
        fill_valid_patches(vpatches,patches,bufs)

        # -- aggregate patches --
        bufs[bufs.levels[0]].vals[:,0] = 0. # no weights
        agg.agg_patches(patches,images,bufs,args)

        # -- misc --
        th.cuda.empty_cache()

        # -- loop update --
        cmasked = masks[0].sum().item()
        delta = cmasked_prev - cmasked
        cmasked_prev = cmasked
        nmasked  = nelems - cmasked
        msg = "[Pixels %d/%d]: %d" % (nmasked,nelems,delta)
        if args.verbose:
            tqdm.write(msg)
            # tqdm.write(("done: %d"%done))
            # tqdm.write(("batches [%d/%d]"% (batch,nbatches)))
            pbar.update(delta)

        # -- logging --
        # print("sum weights: ",torch.sum(images.weights).item())
        # print("sum deno: ",torch.sum(images.deno).item())
        # print("sum basic: ",torch.sum(images.basic).item())

        # - terminate --
        if done: break

    # -- reweight vals --
    # reweight_vals(images)
    # images.weights[th.where(images.weights<5)]=0
    # print("sum: ",th.sum(images.weights>0).item())

    # -- reweight deno --
    weights = images.weights
    index = torch.nonzero(weights,as_tuple=True)
    images.deno[index] /= weights[index]

    # -- fill zeros with basic --
    fill_img = images.basic if args.step==1 else images.noisy
    fill_img = (fill_img/255.-0.5)/0.5
    index = torch.nonzero(weights==0,as_tuple=True)
    images.deno[index] = fill_img[index]

    # -- inspect mask --
    save_mask = False
    if save_mask:
        mask = images.weights == 0
        print("mask.shape: ",mask.shape)
        mask = mask.type(th.float)[:,None]
        print("mask.shape: ",mask.shape,mask.dtype)
        tvu.save_image(mask,"output/mask.png")
        exit(0)

    # -- color xform --
    # utils.color.yuv2rgb_images(images)

    # -- rescale --
    # pnoisy = 255.*(pnoisy*0.5+0.5)
    tmp = images.deno*0.5 + 0.5
    dmin,dmax = tmp.min().item(),tmp.max().item()
    print("deno[min,max]: ",dmin,dmax)

    images.deno[...] = 255.*(images.deno*0.5 + 0.5)
    dmin,dmax = images.deno.min().item(),images.deno.max().item()
    print("deno[min,max]: ",dmin,dmax)


    # -- synch --
    torch.cuda.synchronize()

def reweight_vals(images):
    nmask_before = images.weights.sum().item()
    index = torch.nonzero(images.weights,as_tuple=True)
    images.vals[index] /= images.weights[index]
    irav = images.vals.ravel().cpu().numpy()
    print(np.quantile(irav,[0.1,0.2,0.5,0.8,0.9]))
    # thresh = 0.00014
    thresh = 1e-3
    nz = th.sum(images.vals < thresh).item()
    noupdate = th.nonzero(images.vals > thresh,as_tuple=True)
    images.weights[noupdate] = 0
    th.cuda.synchronize()
    nmask_after = images.weights.sum().item()
    delta_nmask = nmask_before - nmask_after
    print("tozero: [%d/%d]" % (nmask_after,nmask_before))

def fill_valid_patches(vpatches,patches,bufs):
    levels = list(patches.levels)
    for level in levels:
        fill_valid_patches_levels(vpatches[level],patches[level],bufs[level])

def fill_valid_patches_levels(vpatches,patches,bufs):
    tim = th.iinfo(th.int32).min
    valid = th.where(~th.any(th.any(bufs.inds==tim,2),1))
    for key in patches:
        if (key in patches.tensors) and not(patches[key] is None):
            patches[key][valid] = vpatches[key]

def get_valid_bufs(bufs):
    levels = list(bufs.levels)
    vbufs = edict()
    for level in levels:
        vbufs[level] = get_valid_bufs_level(bufs[level])
    vbufs.shape = vbufs[levels[0]].shape
    vbufs.levels = bufs.levels
    return vbufs

def get_valid_bufs_level(bufs):
    tim = th.iinfo(th.int32).min
    valid = th.where(~th.any(th.any(bufs.inds==tim,2),1))
    nv = len(valid[0])
    vbufs = edict()
    for key in bufs:
        if (key in bufs.tensors) and not(bufs[key] is None):
            vbufs[key] = bufs[key][valid]
        else:
            vbufs[key] = bufs[key]
    vbufs.shape[0] = nv
    return vbufs

def get_valid_patches(patches,bufs):
    levels = list(patches.levels)
    vpatches = edict()
    for level in levels:
        vpatches[level] = get_valid_patches_level(patches[level],bufs[level])
    vpatches.shape = vpatches[levels[0]].shape
    vpatches.levels = patches.levels
    return vpatches

def get_valid_patches_level(patches,bufs):
    tim = th.iinfo(th.int32).min
    valid = th.where(~th.any(th.any(bufs.inds==tim,2),1))
    nv = len(valid[0])
    print("total,nv: ",len(bufs.inds),nv)
    vpatches = edict()
    for key in patches:
        if (key in patches.tensors) and not(patches[key] is None):
            vpatches[key] = patches[key][valid]
        else:
            vpatches[key] = patches[key]
    vpatches.shape[0] = nv
    return vpatches

def proc_nl_cache(vid_set,vid_name,sigma):
    return read_nl_sequence(vid_set,vid_name,sigma)


