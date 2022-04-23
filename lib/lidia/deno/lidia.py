

from einops import rearrange,repeat
from lidia.model_io import get_default_opt

def denoise(patches,bufs,args):
    model = args.lidia_model
    opt = get_default_opt(args.sigma)


    # -- shape patches --
    pnoisy = patches.noisy
    pnoisy = rearrange(pnoisy,'r n t c h w -> r n (t c h w)')
    pnoisy = rearrange(pnoisy,'(t p) n d -> t p n d',t=args.t)

    # -- shape values --
    nlDists = rearrange(bufs.vals,'(t p) k -> t p k',t=args.t)
    nlInds = rearrange(bufs.inds,'(t p) k thr -> t p k thr',t=args.t)

    # -- exec --
    dnoisy = model(model,pnoisy,nlDists,nlInds,pnoisy,nlDists,nlInds)

    # -- fill --
    patches.noisy[...] = dnoisy
