
# -- misc --
import sys,tqdm
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image
from torchvision.transforms.functional import center_crop

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- package imports [to test] --
import lidia
from lidia import denoise_nl
from lidia import denoise_ntire2020
from lidia.data import save_burst
from lidia.model_io import get_lidia_model as get_lidia_model_ntire
from lidia.nl_model_io import get_lidia_model as get_lidia_model_nl

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

def print_stats(name,tensor):
    imin,imax = tensor.min().item(),tensor.max().item()
    imean = tensor.mean().item()
    label = "%s[min,max,mean]: " % name
    print(label,imin,imax,imean)

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
        # burst = th.randn((1,3,256,256))
        return burst

    def exec_lidia_denoise(self,name,sigma,device="cuda:0"):

        # -- get data --
        clean = self.load_burst(name).to(device)[:3,:,:96,:128]
        clean = clean.contiguous()
        print("clean.shape: ",clean.shape)
        noisy = clean + sigma * th.randn_like(clean)

        # -- some stats --
        noisy_s = (noisy/255. - 0.5)/0.5
        noisy_s -= noisy_s.mean((-1,-2),True)
        print_stats("noisy",noisy_s)

        # -- exec denos --
        deno_def = lidia.denoise(noisy.clone(),sigma,ftype="ntire2020")
        deno_nl = lidia.denoise(noisy.clone(),sigma,ftype="nl")

        # -- save for viz --
        save_burst(deno_def,SAVE_DIR,"deno_default")
        save_burst(deno_nl,SAVE_DIR,"deno_nl")

        # -- compare --
        error_vals = th.sum((deno_nl - deno_def)**2).item()
        assert error_vals < 1e-10

    def exec_ntire_stepwise_check(self,name,sigma,train,device="cuda:0"):
        """
        Verify that running lidia using a set of
        function calls will give the same output
        as using the standard function call.
        """

        # -- get data --
        clean = self.load_burst(name).to(device)[:3,:,:96,:128]
        clean = clean.contiguous()
        print("clean.shape: ",clean.shape)
        noisy = clean + sigma * th.randn_like(clean)

        # -- some stats --
        noisy_s = (noisy/255. - 0.5)/0.5
        noisy_s -= noisy_s.mean((-1,-2),True)
        print_stats("noisy",noisy_s)

        # -- load model --
        im_shape = noisy.shape
        model = get_lidia_model_ntire(device,im_shape,sigma)

        # -- exec denos --
        deno_def = lidia.denoise(noisy.clone(),sigma,ftype="ntire2020",train=train)
        deno_def = deno_def.detach()
        print("deno_def.shape: ",deno_def.shape)
        deno_steps = model.run_parts(noisy.clone(),sigma,train=train)
        deno_steps = deno_steps.detach()
        print("deno_steps.shape: ",deno_steps.shape)

        # -- save for viz --
        save_burst(deno_def,SAVE_DIR,"deno_default")
        save_burst(deno_steps,SAVE_DIR,"deno_steps")

        # -- compare --
        error_vals = th.sum((deno_def - deno_steps)**2).item()
        assert error_vals < 1e-10

    def exec_nl_stepwise_check(self,name,sigma,train,device="cuda:0"):
        """
        Verify that running lidia using a set of
        function calls will give the same output
        as using the standard function call.
        """

        # -- get data --
        ps = 5
        clean = self.load_burst(name).to(device)[:3,:,:96,:128]
        clean = clean.contiguous()
        print("clean.shape: ",clean.shape)
        noisy = clean + sigma * th.randn_like(clean)

        # -- load model --
        im_shape = noisy.shape
        model = get_lidia_model_nl(device,im_shape,sigma)

        # -- exec denos --
        deno_def = lidia.denoise(noisy.clone(),sigma,ftype="ntire2020",train=train)
        deno_def = deno_def.detach()
        deno_steps = model.run_parts(noisy.clone(),sigma,train=train)
        deno_steps = deno_steps.detach()

        # -- save for viz --
        save_burst(deno_def,SAVE_DIR,"deno_default")
        save_burst(deno_steps,SAVE_DIR,"deno_steps")

        # -- save delta --
        delta = th.abs(deno_def - deno_steps)
        delta /= delta.max()
        save_burst(delta,SAVE_DIR,"deno_delta")

        # -- compare all [boarders have error] --
        error_vals = th.sum((deno_def - deno_steps)**2).item()
        assert error_vals < 110.

        # -- max pixel diff --
        error_vals = ((deno_def - deno_steps)**2).max().item()
        assert error_vals < 1.


    def test_lidia_denoise(self):

        # -- set seed --
        seed = 123
        th.manual_seed(seed)
        np.random.seed(seed)

        # -- test 1 --
        name,sigma = "davis_baseball_64x64",15.
        name = "davis_salsa"
        # self.exec_lidia_denoise(name,sigma)
        # self.exec_ntire_stepwise_check(name,sigma,False)
        # self.exec_nl_stepwise_check(name,sigma,False)

        # -- train true --
        # self.exec_ntire_stepwise_check(name,sigma,True)
        self.exec_nl_stepwise_check(name,sigma,True)



