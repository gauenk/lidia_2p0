
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

def run_rgb2gray_patches(patches,ps):
    t,h,w,k,d = patches.shape
    patches = rearrange(patches,'t h w k (c ph pw) -> (t h w k) c ph pw',ph=ps,pw=ps)
    patches = run_rgb2gray(patches)
    patches = rearrange(patches,'(t h w k) 1 ph pw -> t h w k (ph pw)',t=t,h=h,w=w)
    return patches


#
#
# -- Primary Testing Class --
#
#

class TestAgg0(unittest.TestCase):

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

    def test_agg0_ntire(self):
        name = "davis_baseball_64x64"
        sigma = 50.
        device = "cuda:0"
        self.run_agg0_ntire(name,sigma,device)

    def test_agg0_dnls(self):
        name = "davis_baseball_64x64"
        sigma = 50.
        device = "cuda:0"
        self.run_agg0_dnls(name,sigma,device)


    def run_agg0_ntire(self,name,sigma,device):

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #        Prepare
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        clean = self.load_burst(name).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #        Execute
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        #
        # -- ground-truth --
        #

        # -- patches --
        noisy_mod = (noisy.clone()/255.-0.5)/0.5
        noisy_mod -= noisy_mod.mean(dim=(-2,-1),keepdim=True)
        ntire_output = model_ntire.run_nn0(noisy)#_mod)
        ntire_patches = ntire_output[0]
        ntire_dists = ntire_output[1]
        ntire_inds = ntire_output[2]
        t,hp,wp,k,d = ntire_patches.shape

        # -- agg --
        ha,wa = 72,72
        ipatches = rearrange(ntire_patches,'t h w k d -> t (h w) k d')
        idists = rearrange(ntire_dists,'t h w k -> t (h w) k')
        ntire_agg0,ntire_s0,ntire_fold = model_ntire.run_agg0(ipatches,idists,ha,wa)
        ntire_fold = ntire_fold.detach()
        ntire_agg0 = rearrange(ntire_agg0,'t (h w) k d -> t h w k d',h=hp).detach()/30.

        #
        # -- comparison --
        #

        # -- patches --
        noisy_mod = (noisy.clone()/255.-0.5)/0.5
        noisy_mod -= noisy_mod.mean(dim=(-2,-1),keepdim=True)
        nl_output = model_nl.run_nn0_lidia_search(noisy)#_mod)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        # -- agg --
        ha,wa = 72,72
        ipatches = rearrange(nl_patches,'t h w k d -> t (h w) k d')
        iinds = rearrange(nl_inds,'t h w k tr -> t (h w) k tr')
        idists = rearrange(nl_dists,'t h w k -> t (h w) k')
        nl_agg0,nl_s0,nl_fold = model_nl.run_agg0(ipatches,idists,iinds,64,64)
        nl_fold = nl_fold.detach()
        nl_agg0 = rearrange(nl_agg0,'t (h w) k d -> t h w k d',h=hp).detach()/30.

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #     Vizualization
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        # # -- fold --
        # error = th.abs(ntire_fold - nl_fold)/255.
        # save_burst(error,"output/tests/agg0/","error_fold")
        # error = error.sum()
        # assert error < 1e-10

        # # -- agg --
        # nta = rearrange(ntire_agg0[:,:,:,0,13::25],'t h w c -> t c h w')
        # nla = rearrange(nl_agg0[:,:,:,0,13::25],'t h w c -> t c h w')
        # save_burst(nta,"output/tests/agg0/","ntire_agg")
        # save_burst(nla,"output/tests/agg0/","nl_agg")

        # # -- agg delta --
        # error = th.mean((ntire_agg0 - nl_agg0)**2,-1)
        # error = repeat(error[...,0],'t h w -> t c h w',c=3)
        # save_burst(error,"output/tests/agg0/","error_agg")

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #      Compare
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        # -- sep --
        error = (nl_s0 - ntire_s0)**2
        error = error.sum().item()
        assert error < 1e-10

        # -- fold --
        error = (nl_fold - ntire_fold)**2
        error = error.sum().item()
        assert error < 1e-10

        # -- patches  --
        error = (nl_agg0 - ntire_agg0)**2
        error = error.sum().item()
        assert error < 1e-10


    def run_agg0_dnls(self,name,sigma,device):

        #
        # -- Prepare --
        #

        clean = self.load_burst(name).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #        Execute
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        #
        # -- ground-truth --
        #

        # -- patches --
        noisy_mod = (noisy.clone()/255.-0.5)/0.5
        noisy_mod -= noisy_mod.mean(dim=(-2,-1),keepdim=True)
        ntire_output = model_ntire.run_nn0(noisy)
        ntire_patches = ntire_output[0]
        ntire_dists = ntire_output[1]
        ntire_inds = ntire_output[2]
        t,hp,wp,k,d = ntire_patches.shape

        # -- agg --
        ha,wa = 72,72
        ipatches = rearrange(ntire_patches,'t h w k d -> t (h w) k d')
        idists = rearrange(ntire_dists,'t h w k -> t (h w) k')
        ntire_agg0,ntire_s0,ntire_fold = model_ntire.run_agg0(ipatches,idists,ha,wa)
        ntire_fold = ntire_fold.detach()
        ntire_agg0 = rearrange(ntire_agg0,'t (h w) k d -> t h w k d',h=hp).detach()/30.

        #
        # -- comparison --
        #

        # -- patches --
        noisy_mod = (noisy.clone()/255.-0.5)/0.5
        noisy_mod -= noisy_mod.mean(dim=(-2,-1),keepdim=True)
        nl_output = model_nl.run_nn0_dnls_search(noisy)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        # -- agg --
        ha,wa = 72,72
        ipatches = rearrange(nl_patches,'t h w k d -> t (h w) k d')
        iinds = rearrange(nl_inds,'t h w k tr -> t (h w) k tr')
        idists = rearrange(nl_dists,'t h w k -> t (h w) k')
        nl_agg0,nl_s0,nl_fold = model_nl.run_agg0(ipatches,idists,iinds,64,64)
        nl_fold = nl_fold.detach()
        nl_agg0 = rearrange(nl_agg0,'t (h w) k d -> t h w k d',h=hp).detach()/30.

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #     Vizualization
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        print("nl_fold.shape: ",nl_fold.shape)
        print("ntire_fold.shape: ",ntire_fold.shape)
        delta = th.abs(nl_fold - ntire_fold)
        delta /= delta.max()
        save_burst(delta,"output/tests/agg0","error_fold")

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #      Compare
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        # -- fold  --
        error = (nl_fold - ntire_fold)**2
        print(error.max())
        error = error.sum().item()
        assert error < 1e-10

        # -- patches  --
        error = (nl_agg0 - ntire_agg0)**2
        error = error.sum().item()
        assert error < 1e-10

