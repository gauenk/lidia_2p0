
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
import lidia
from lidia.model_io import get_lidia_model as get_lidia_model_ntire
from lidia.nl_model_io import get_lidia_model as get_lidia_model_nl
from lidia.data import save_burst

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)


def run_rgb2gray(tensor):
    kernel = th.tensor([0.2989, 0.5870, 0.1140], dtype=th.float32)
    kernel = kernel.view(1, 3, 1, 1)
    rgb2gray = th.nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(1, 1),bias=False)
    rgb2gray.weight.data = kernel
    rgb2gray.weight.requires_grad = False
    rgb2gray = rgb2gray.to(tensor.device)
    tensor = rgb2gray(tensor)
    return tensor


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

    @unittest.skip
    def test_nonlocal1(self):
        # -- params --
        name = "davis_baseball_64x64"
        sigma = 50.
        device = "cuda:0"

        # -- exec --
        self.run_nonlocal1_test(name,sigma,device)


    def run_nonlocal1_test(self,name,sigma,device):

        # -- get data --
        clean = self.load_burst(name).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -- exec ntire search  --
        ntire_output = model_ntire.run_nn1(noisy)
        ntire_patches = ntire_output[0]
        ntire_dists = ntire_output[1]
        ntire_inds = ntire_output[2]

        # -- exec nl search  --
        nl_output = model_nl.run_nn1(noisy)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        #
        # -- Viz --
        #

        #
        # -- Comparisons --
        #

        # -- patches  --
        error = (ntire_patches[:,:,:] - nl_patches[:,:,:])**2
        error = error.sum().item()
        assert error < 1e-10

        # -- dists  --
        error = (ntire_dists - nl_dists)**2
        error = error.sum().item()
        assert error < 1e-10

        # -- inds  --
        error = (ntire_inds - nl_inds)**2
        error = error.sum().item()
        assert error < 1e-10
