
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
from .nl_search import run_search


def denoise(noisy, sigma, pm_vid=None, flows=None,
            gpuid=0, clean=None, verbose=True, ftype="nl", train=False):
    if ftype == "nl":
        return denoise_nl(noisy, sigma, pm_vid, flows, gpuid, clean, verbose)
    elif ftype in ["default","ntire2020"]:
        noisy_n1p1 = (noisy/255. - 0.5)/0.5
        deno_n1p1 = denoise_ntire2020(noisy_n1p1, sigma, pm_vid, flows, train)
        deno = 255*(deno_n1p1 * 0.5 + 0.5)
        return deno
    else:
        raise ValueError(f"Unknown lidia function type [{ftype}]")



