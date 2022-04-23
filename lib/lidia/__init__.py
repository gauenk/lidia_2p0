
# -- Remove Numba Warnings --
import warnings
from numba import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#
# -- API --
#

from . import modules
from . import nl_modules

from . import data
from . import data_modules
from .deno_set import denoise_npc,denoise_ntire2020
from .impl import denoise_nl


def denoise(noisy, sigma, pm_vid=None, flows=None,
            gpuid=0, clean=None, verbose=True, ftype="nl"):
    if ftype == "nl":
        return denoise_nl(noisy, sigma, pm_vid, flows, gpuid, clean, verbose)
    elif ftype in ["default","ntire2020"]:
        noisy_01 = noisy/255.
        noisy_n1p1 = (noisy_01 - 0.5)/0.5
        deno_n1p1 = denoise_ntire2020(noisy_n1p1, sigma, pm_vid, flows)
        print("n1p1: ",deno_n1p1.min(),deno_n1p1.max())
        deno_01 = deno_n1p1 * 0.5 + 0.5
        print("01: ",deno_01.min(),deno_01.max())
        deno = 255. * deno_01
        return deno
    else:
        raise ValueError(f"Unknown lidia function type [{ftype}]")



