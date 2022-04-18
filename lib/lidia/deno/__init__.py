
from .bayes import denoise as bayes_denoise
from .lidia import denoise as lidia_denoise

def denoise(patches,args,dtype):
    if dtype == "bayes":
        return bayes_denoise(patches,args)
    elif dtype == "lidia":
        return lidia_denoise(patches,args)
    else:
        raise ValueError(f"Unknown denoiser [{dtype}]")
