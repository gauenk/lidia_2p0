
# -- misc --
import sys,tqdm
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- package imports [to test] --
from lidia import denoise_nl
from lidia import denoise_ntire2020
from lidia.data import save_burst

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/")

#
#
# -- Primary Testing Class --
#
#

class TestLidiaDenoiseRgb(unittest.TestCase):

    #
    # -- Load Data --
    #

    def load_burst(self,name,ext="jpg"):
        path = DATA_DIR / name
        assert path.exists()
        burst = []
        for t in range(MAX_NFRAMES):
            fn = path / ("%05d.%s" % (t,ext))
            if not fn.exists(): break
            img_t = Image.open(str(fn)).convert("RGB")
            img_t = np.array(img_t)
            img_t = rearrange(img_t,'h w c -> c h w')
            burst.append(img_t)
        if len(burst) == 0:
            print(f"WARNING: no images loaded. Check ext [{ext}]")
        burst = 1.*np.stack(burst)
        burst = th.from_numpy(burst).type(th.float32)
        return burst

    def exec_lidia_denoise(self,name,sigma,device="cuda:0"):

        # -- get data --
        clean = self.load_burst(name).to(device)
        noisy = clean + sigma * th.randn_like(clean)

        # -- exec denos --
        deno_nl = denoise_nl(noisy,sigma)
        deno_def = denoise_ntire2020(noisy,sigma)

        # -- compare --
        error_vals = th.sum((deno_nl - deno_def)**2).item()
        assert error_vals < 1e-10

    def test_lidia_denoise(self):

        # -- test 1 --
        name,sigma = "davis_baseball_64x64",50.
        self.exec_lidia_denoise(name,sigma)
