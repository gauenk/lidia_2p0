# -- python deps --
import torch
import scipy
import numpy as np
from einops import rearrange

# -- numba --
from numba import njit,cuda

# -- package --
import npc.search_mask as imask
from npc.utils import groups2patches
from npc.utils import Timer


def agg_patches_midpix(patches,images,bufs,args,cs_ptr=None,denom="hw"):
    # -- default stream --
    if cs_ptr is None:
        cs_ptr = torch.cuda.default_stream().cuda_stream

    # -- filter by valid --
    valid = torch.nonzero(torch.all(bufs.inds!=-1,1))[:,0]
    vnoisy = patches.noisy[valid]
    vinds = bufs.inds[valid]
    vvals = bufs.vals[valid]

    # -- iterate over "nkeep" --
    if args.nkeep != -1:
        vinds = vinds[:,:args.nkeep]

    compute_agg_batch_midpix(images.deno,vnoisy,vinds,images.weights,
                             vvals,images.vals,args.ps,args.ps_t,cs_ptr,denom=denom)

def compute_agg_batch_midpix(deno,patches,inds,weights,vals,ivals,
                             ps,ps_t,cs_ptr,denom="hw"):

    # -- numbify the torch tensors --
    deno_nba = cuda.as_cuda_array(deno)
    patches_nba = cuda.as_cuda_array(patches)
    inds_nba = cuda.as_cuda_array(inds)
    weights_nba = cuda.as_cuda_array(weights)
    vals_nba = cuda.as_cuda_array(vals)
    ivals_nba = cuda.as_cuda_array(ivals)
    cs_nba = cuda.external_stream(cs_ptr)

    # -- launch params --
    bsize,num = inds.shape
    c,ph,pw = patches.shape[-3:]
    threads = (c,ph,pw)
    blocks = (bsize,num)

    # -- launch kernel --
    # exec_agg_cuda[blocks,threads,cs_nba](deno_nba,patches_nba,inds_nba,weights_nba,
    #                                      vals_nba_,ivals_nba,ps,ps_t)
    exec_agg_simple_midpix(deno,patches,inds,weights,vals,ivals,ps,ps_t,denom=denom)

def exec_agg_simple_midpix(deno,patches,inds,weights,vals,ivals,ps,ps_t,denom="hw"):

    # -- numbify --
    device = deno.device
    deno_nba = deno.cpu().numpy()
    patches_nba = patches.cpu().numpy()
    inds_nba = inds.cpu().numpy()
    weights_nba = weights.cpu().numpy()
    vals_nba = vals.cpu().numpy()
    ivals_nba = ivals.cpu().numpy()

    # -- exec numba --
    exec_agg_simple_midpix_numba(deno_nba,patches_nba,inds_nba,
                                 weights_nba,vals_nba,ivals_nba,ps,ps_t,
                                 denom=denom)

    # -- back pack --
    deno_nba = torch.FloatTensor(deno_nba).to(device)
    deno[...] = deno_nba
    weights_nba = torch.FloatTensor(weights_nba).to(device)
    weights[...] = weights_nba
    ivals_nba = torch.FloatTensor(ivals_nba).to(device)
    ivals[...] = ivals_nba


@njit
def exec_agg_simple_midpix_numba(deno,patches,inds,weights,vals,
                                 ivals,ps,ps_t,denom="hw"):

    # -- shape --
    nframes,color,height,width = deno.shape
    chw = color*height*width
    hw = height*width
    bsize,npatches = inds.shape # "npatches" _must_ be from "inds"
    Z = hw if denom == "hw" else chw
    psHalf = ps//2

    for bi in range(bsize):
        for ni in range(npatches):
            ind = inds[bi,ni]
            if ind == -1: continue
            t0 = ind // Z
            h0 = (ind % hw) // width
            w0 = ind % width

            # print(t0,h0,w0)
            for pt in range(ps_t):
                for pi in range(ps):
                    for pj in range(ps):
                        t1 = (t0+pt)# % nframes
                        h1 = (h0+pi-psHalf)# % height
                        w1 = (w0+pj-psHalf)# % width

                        if t1 < 0 or t1 >= nframes: continue
                        if h1 < 0 or h1 >= height: continue
                        if w1 < 0 or w1 >= width: continue

                        for ci in range(color):
                            gval = patches[bi,ni,pt,ci,pi,pj]
                            deno[t1,ci,h1,w1] += gval
                        weights[t1,h1,w1] += 1.
                        # if ni > 0:
                        #     ivals[t0+pt,h0+pi,w0+pj] += vals[bi,ni]


