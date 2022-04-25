
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

    # @unittest.skip("testing nl1")
    def test_nonlocal0(self):
        # -- params --
        name = "davis_baseball_64x64"
        sigma = 50.
        device = "cuda:0"

        # -- set seed --
        seed = 123
        th.manual_seed(seed)
        np.random.seed(seed)

        # -- exec --
        # self.run_nonlocal0_lidia_search(name,sigma,device)
        self.run_nonlocal0_dnls_search(name,sigma,device)

    def run_nonlocal0_lidia_search(self,name,sigma,device):

        # -- get data --
        clean = self.load_burst(name).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -- exec ntire search  --
        ntire_output = model_ntire.run_nn0(noisy)
        ntire_patches = ntire_output[0]
        ntire_dists = ntire_output[1]
        ntire_inds = ntire_output[2]

        # -- exec nl search  --
        nl_output = model_nl.run_nn0_lidia_search(noisy)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        #
        # -- Viz and Prints --
        #

        print("nl_dists.shape: ",nl_dists.shape)
        print("ntire_dists.shape: ",ntire_dists.shape)
        print("nl_inds.shape: ",nl_inds.shape)
        print("ntire_inds.shape: ",ntire_inds.shape)
        print(nl_inds[0,16,16])
        print(ntire_inds[0,16,16])
        print(nl_dists[0,16,16])
        print(ntire_dists[0,16,16])


        print("-"*20)
        print("-"*20)
        print(ntire_patches[0,16,16,0])
        print(nl_patches[0,16,16,0])
        print("-"*20)
        print("-"*20)

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

    def run_nonlocal0_dnls_search(self,name,sigma,device):

        # -- get data --
        clean = self.load_burst(name).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape
        ps = 5

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -- exec ntire search  --
        ntire_output = model_ntire.run_nn0(noisy)
        ntire_patches = ntire_output[0]
        ntire_dists = ntire_output[1]
        ntire_inds = ntire_output[2]

        # -- exec nl search  --
        nl_output = model_nl.run_nn0_dnls_search(noisy)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        #
        # -- Viz Comparison --
        #

        # print("nl_inds.shape: ",nl_inds.shape)
        # print("dists [nl,ntire]")
        # print("[nl]: ",nl_dists[0,31,31])
        # print("[ntire]: ",ntire_dists[0,31,31])

        # print("inds [nl,ntire]")
        # print("[nl]: ",nl_inds[0,31,31])
        # print("[ntire]: ",ntire_inds[0,31,31])


        # pinds = th.stack([nl_inds[0,35:45,-2:-1,1],ntire_inds[0,35:45,-2:-1,1]],-1)
        # pdists = th.stack([nl_dists[0,30:50,-2:-1,1],ntire_dists[0,30:50,-2:-1,1]],-1)
        # # print("pinds.shape: ",pinds.shape)
        # # print(pinds)
        # print(pdists)
        # # print(nl_dists[0,25:35,-2:,:2])
        # # print(nl_inds[0,25:35,-2:,:2])
        # # print(ntire_dists[0,25:35,-2:,:2])

        # # print("nl_dists.shape: ",nl_dists.shape)
        # # print("ntire_dists.shape: ",ntire_dists.shape)
        # # print("nl_inds.shape: ",nl_inds.shape)
        # # print("ntire_inds.shape: ",ntire_inds.shape)
        # # print(nl_inds[0,0,0,0])
        # # print(ntire_inds[0,0,0,0])
        # # print(nl_inds[0,-1,-1,0])
        # # print(ntire_inds[0,-1,-1,0])


        # p0 = ntire_patches[0,31,31,0].view(1,3,5,5)
        # p1 = ntire_patches[0,31,31,1].view(1,3,5,5)
        # print("p0.shape: ",p0.shape)
        # print("p1.shape: ",p1.shape)
        # print(p0)
        # print(p1)
        # p0_gray = run_rgb2gray(p0)/255.
        # p1_gray = run_rgb2gray(p1)/255.
        # dist = th.sum((p0_gray - p1_gray)**2).item()
        # print("[ntire] Dist: ",dist)

        # print("nl_patches.shape: ",nl_patches.shape)
        # p0 = nl_patches[0,31,31,0].view(1,3,5,5)
        # p1 = nl_patches[0,31,31,1].view(1,3,5,5)
        # print(p0)
        # print(p1)
        # print("p0.shape: ",p0.shape)
        # print("p1.shape: ",p1.shape)
        # p0_gray = run_rgb2gray(p0)/255.
        # p1_gray = run_rgb2gray(p1)/255.
        # dist = th.sum((p0_gray - p1_gray)**2).item()
        # print("[nl] Dist: ",dist)


        # ti = 0

        # delta = th.sum((ntire_patches[ti]/255. - nl_patches[ti]/255.)**2,dim=-1)
        # print("delta.shape: ",delta.shape)
        # args = th.where(delta>1e-10)
        # print(args)
        # delta = repeat(delta,'h w k -> k c h w',c=3)
        # print("delta.shape: ",delta.shape)
        # save_burst(delta,"./output/tests/agg","delta_p")
        # print(args[0][0],args[1][0])
        # print("-"*10)
        # print(ntire_patches[ti][args[0][0],args[1][0],args[2][0]])
        # print(nl_patches[ti][args[0][0],args[1][0],args[2][0]])
        # print("-"*10)
        # print("ntire_inds.shape: ",ntire_inds.shape)
        # print(ntire_inds[ti][args[0][0],args[1][0],:])
        # print(nl_inds[ti][args[0][0],args[1][0],:])
        # a = ntire_inds[ti][args[0][0],args[1][0],:]
        # b = nl_inds[ti][args[0][0],args[1][0],:]
        # print(a - b)
        # print("-"*10)
        # print("ntire_dists.shape: ",ntire_dists.shape)
        # print(ntire_dists[ti][args[0][0],args[1][0],:])
        # print(nl_dists[ti][args[0][0],args[1][0],:])
        # print("-"*10)


        # delta = (ntire_inds[ti] - nl_inds[ti])**2
        # delta = th.mean(delta.type(th.float),dim=-1)
        # print("delta.shape: ",delta.shape)
        # delta = repeat(delta,'h w k -> k c h w',c=3)
        # print("delta.shape: ",delta.shape)
        # save_burst(delta,"./output/tests/agg","delta_i")


        # delta = (ntire_dists[ti] - nl_dists[ti])**2
        # # delta = th.mean(delta.type(th.float),dim=-1)
        # # print("delta.max(): ", delta.max().item())
        # # delta /= delta.max().item()
        # # print("delta.shape: ",delta.shape)
        # delta = repeat(delta,'h w k -> k c h w',c=3)
        # # print("delta.shape: ",delta.shape)
        # save_burst(delta,"./output/tests/agg","delta_d")

        # print("-"*20)
        # print("-"*20)

        # print(ntire_patches[0,0,0,0])
        # print(nl_patches[0,0,0,0])

        # print("-"*20)
        # print("-"*20)

        # print(ntire_patches[0,0,0,1])
        # print(nl_patches[0,0,0,1])

        # print("-"*20)
        # print("-"*20)


        # print("inds [nl,ntire]")
        # print(nl_inds[0,31,31])
        # print(ntire_inds[0,31,31])

        # print("-- spatial dists --")
        # print(nl_dists[0,8:15,8:15,1])
        # print(ntire_dists[0,8:15,8:15,1])
        # print(nl_dists[0,15:22,15:22,1])
        # print(ntire_dists[0,15:22,15:22,1])
        # print("-"*20)

        # print(ntire_patches[0,31,31,0])
        # print(nl_patches[0,31,31,0])

        # print(ntire_patches[0,10,10,0])
        # print(nl_patches[0,10,10,0])

        # print("nl_patches.shape: ",nl_patches.shape)

        #
        # -- Comparisons --
        #

        """
        We can't do direct comparisons because equal dist
        locations may be swapped.
        """

        # -- dists  --
        error = (ntire_dists - nl_dists)**2
        error = error.sum().item()
        assert error < 1e-8

        # -- inds  --
        # error = (ntire_inds - nl_inds)**2
        # error = error.sum().item()
        # assert error < 1e-10

        # -- [nl] patch-based dists == dist --
        nl_patches = run_rgb2gray_patches(nl_patches,ps)/255.
        dists = (nl_patches - nl_patches[...,[0],:])**2
        nl_pdists = th.sum(dists,-1)
        error = (nl_pdists - nl_dists)**2
        error = error.sum().item()
        assert error < 1e-8

        # -- [ntire] patch-based dists == dist --
        ntire_patches = run_rgb2gray_patches(ntire_patches,ps)/255.
        dists = (ntire_patches - ntire_patches[...,[0],:])**2
        ntire_pdists = th.sum(dists,-1)
        error = (ntire_pdists - ntire_dists)**2
        error = error.sum().item()
        assert error < 1e-8


