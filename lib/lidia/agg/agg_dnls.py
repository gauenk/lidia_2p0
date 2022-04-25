


import dnls
import torch as th

def agg_patches_dnls(patches,images,bufs,args):

    # -- filter by valid --
    l0 = patches.levels[0]
    tim = th.iinfo(th.int32).min
    valid = th.where(~th.any(th.any(bufs[l0].inds==tim,2),1))
    vnoisy = patches[l0].noisy[valid]
    vvals = bufs[l0].vals[valid]
    vinds = bufs[l0].inds[valid]

    # -- iterate over "nkeep" --
    if args.nkeep != -1:
        vvals = vvals[:,:args.nkeep]
        vinds = vinds[:,:args.nkeep]
        vnoisy = vnoisy[:,:args.nkeep]

    # -- exec --
    dnls.simple.gather.run(vnoisy,vvals,vinds,vid=images.deno,
                           wvid=images.weights)
