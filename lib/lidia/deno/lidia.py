

from einops import rearrange,repeat
from lidia.model_io import get_default_opt

def denoise(patches,bufs,args):
    model = args.lidia_model
    opt = get_default_opt(args.sigma)
    pnoisy = patches.noisy
    pnoisy = rearrange(pnoisy,'r n t c h w -> r n (t c h w)')
    print("HI")
    print("pnoisy.shape: ",pnoisy.shape)
    dnoisy = model(model,pnoisy,bufs.vals,pnoisy,bufs.vals)
    print("dnoisy.shape: ",dnoisy.shape)
    patches.noisy[...] = dnoisy
