


import dnls
import torch as th

def agg_patches_dnls(patches,images,bufs,args):

    # -- filter by valid --
    valid = th.where(~th.any(th.any(bufs.inds==-1,2),1))[0]
    vnoisy = patches.noisy[valid]
    vvals = bufs.vals[valid]
    vinds = bufs.inds[valid]

    # -- iterate over "nkeep" --
    if args.nkeep != -1:
        vvals = vvals[:,:args.nkeep]
        vinds = vinds[:,:args.nkeep]
        vnoisy = vnoisy[:,:args.nkeep]

    # -- exec --
    dnls.simple.gather.run(vnoisy,vvals,vinds,vid=images.deno,
                           wvid=images.weights)
