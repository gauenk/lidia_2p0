
from .bayes import denoise as bayes_denoise
from .lidia import denoise as lidia_denoise

def denoise(patches,bufs,args,dtype,noisy_means):
    if dtype == "bayes":
        return bayes_denoise(patches,args)
    elif dtype == "lidia":
        return lidia_denoise(patches,bufs,args,noisy_means)
    else:
        raise ValueError(f"Unknown denoiser [{dtype}]")
