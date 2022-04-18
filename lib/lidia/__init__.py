from . import data
from . import data_modules
from .deno_set import denoise_npc,denoise_ntire2020
from .impl import denoise_nl


def denoise(noisy, sigma, pm_vid=None,
            flows=None, gpuid=0, clean=None, verbose=True, ftype="nl"):
    if ftype == "nl":
        return denoise_nl(noisy, sigma, pm_vid, flows, gpuid, clean, verbose)
    elif ftype == "default":
        return denoise_ntire2020(noisy, sigma, pm_vid, flows, gpuid, clean, verbose)
    else:
        raise ValueError(f"Unknown lidia function type [{ftype}]")



