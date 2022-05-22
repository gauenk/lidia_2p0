
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

class TestNn1(unittest.TestCase):

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

    def test_nonlocal1(self):
        # -- params --
        # name = "davis_baseball_64x64"
        name = "davis_salsa"
        sigma = 50.
        device = "cuda:0"

        # -- no-train --
        self.run_nonlocal1_lidia_search(name,sigma,False,device)
        self.run_nonlocal1_dnls_search(name,sigma,False,device)
        self.run_proc_search(name,sigma,False,device)

        # -- train --
        self.run_nonlocal1_lidia_search(name,sigma,True,device)
        self.run_nonlocal1_dnls_search(name,sigma,True,device)
        self.run_proc_search(name,sigma,True,device)

    def run_nonlocal1_lidia_search(self,name,sigma,train,device):

        # -- get data --
        clean = self.load_burst(name).to(device)
        clean = clean[:5,:,:96,:128].contiguous()
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -- exec ntire search  --
        ntire_output = model_ntire.run_nn1(noisy,train=train)
        ntire_patches = ntire_output[0]
        ntire_dists = ntire_output[1]
        ntire_inds = ntire_output[2]

        # -- exec nl search  --
        nl_output = model_nl.run_nn1_lidia_search(noisy,train=train)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        #
        # -- Viz and Prints --
        #

        # print("nl_dists.shape: ",nl_dists.shape)
        # print("ntire_dists.shape: ",ntire_dists.shape)
        # print("nl_inds.shape: ",nl_inds.shape)
        # print("ntire_inds.shape: ",ntire_inds.shape)
        # print(nl_inds[0,16,16])
        # print(ntire_inds[0,16,16])
        # print(nl_dists[0,16,16])
        # print(ntire_dists[0,16,16])

        # print("-"*20)
        # print("-"*20)
        # print(ntire_patches[0,16,16,0])
        # print(nl_patches[0,16,16,0])
        # print("-"*20)
        # print("-"*20)

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


    def run_nonlocal1_dnls_search(self,name,sigma,train,device):

        # -- get data --
        ps = 5
        clean = self.load_burst(name).to(device)
        clean = clean[:5,:,:96,:128].contiguous()
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -- exec ntire search  --
        ntire_output = model_ntire.run_nn1(noisy/255.,train=train)
        ntire_patches = ntire_output[0]
        ntire_dists = ntire_output[1]
        ntire_inds = ntire_output[2]

        # -- exec nl search  --
        nl_output = model_nl.run_nn1_dnls_search(noisy/255.,train=train)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        #
        # -- Viz and Prints --
        #

        hi,wi = 0,0
        # hi,wi = 16,16
        # print("nl_dists.shape: ",nl_dists.shape)
        # print("ntire_dists.shape: ",ntire_dists.shape)
        # print("nl_inds.shape: ",nl_inds.shape)
        # print("ntire_inds.shape: ",ntire_inds.shape)
        # print("-"*20)
        # print("-"*20)
        # print(nl_inds[0,hi,wi])
        # print(ntire_inds[0,hi,wi])
        # print("-"*20)
        # print("-"*20)
        # print(nl_dists[0,hi,wi])
        # print(ntire_dists[0,hi,wi])
        # print("-"*20)
        # print("-"*20)
        # print(ntire_patches[0,hi,wi,0].view(3,5,5)[0])
        # print(nl_patches[0,hi,wi,0].view(3,5,5)[0])
        # print("-"*20)
        # print("-"*20)

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

        # -- 1st patch content  --
        error = (nl_patches[...,0,:] - ntire_patches[...,0,:])**2
        error = error.sum().item()
        assert error < 1e-8

        # -- ave patch content  --
        # allow for some error b/c k^th rank may have multi. equiv dists
        nl_mp = nl_patches[...,:,:].mean(-2)
        ntire_mp = ntire_patches[...,:,:].mean(-2)
        error = (nl_mp - ntire_mp)**2
        error = error.sum().item()
        assert error < 1.5

        # -- [nl] patch-based dists == dist --
        nl_patches = run_rgb2gray_patches(nl_patches,ps)
        dists = (nl_patches - nl_patches[...,[0],:])**2
        nl_pdists = th.sum(dists,-1)
        error = ((nl_pdists[...,1:] - nl_dists[...,:-1]))**2
        error = error.sum().item()
        assert error < 1e-8

        # -- [ntire] patch-based dists == dist --
        ntire_patches = run_rgb2gray_patches(ntire_patches,ps)
        dists = (ntire_patches - ntire_patches[...,[0],:])**2
        ntire_pdists = th.sum(dists,-1)
        error = ((ntire_pdists[...,1:] - ntire_dists[...,:-1]))**2
        error = error.sum().item()
        assert error < 1e-8

    def run_proc_search(self,name,sigma,train,device):

        # -- get data --
        clean = self.load_burst(name).to(device)
        clean = clean[:5,:,:96,:128].contiguous()
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape
        ps = 5

        # -- load model --
        model_ntire = get_lidia_model_ntire(device,im_shape,sigma)
        model_nl = get_lidia_model_nl(device,im_shape,sigma)

        # -- prepare image --
        nl_noisy = (noisy.clone()/255.-0.5)/0.5
        nl_noisy -= nl_noisy.mean(dim=(-2,-1),keepdim=True)

        # -- exec nl search  --
        nl_output = model_nl.run_nn1_dnls_search(nl_noisy.clone(),train=train)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        # -- exec proc nl search loop  --
        nl_res = lidia.run_search(nl_noisy.clone(),sigma,train=train)
        patches,inds,dists = nl_res.p1,nl_res.i1,nl_res.d1

        #
        # -- Viz --
        #

        # print("nl_dists.shape: ",nl_dists.shape)
        # print("dists.shape: ",dists.shape)
        # print("nl_inds.shape: ",nl_inds.shape)
        # print("inds.shape: ",inds.shape)
        # print(nl_inds[0,16,16])
        # print(inds[0,16,16])
        # print(nl_dists[0,16,16])
        # print(dists[0,16,16])

        #
        # -- Comparison --
        #

        # -- eq dists --
        error = th.sum((nl_dists - dists)**2/nl_dists).item()
        assert error < 1e-8

        # -- 0th neigh --
        error = (nl_patches[...,0,:] - patches[...,0,:])**2
        error = error.sum().item()
        assert error < 1e-8

        # -- ave neigh --
        nl_mpatch = nl_patches.mean(-2)
        mpatch = patches.mean(-2)
        error = (nl_mpatch - mpatch)**2
        error = error.sum().item()
        assert error < 1.


