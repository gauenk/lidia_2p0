
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
from lidia import denoise_nl
from lidia import denoise_ntire2020
from lidia.data import save_burst

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)


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

    def format_patches(self,clean,noisy,npatches,color,ps):

        # -- init --
        device = clean.device
        patches = th.randn((npatches,ps,ps),device=device)

        # -- fill center with name --
        psMid = ps//2
        patches[...] = 0.
        patches[:,psMid,psMid] = th.arange(npatches,device=device)#/npatches

        # -- repeat --
        patches = repeat(patches,'n h w -> n c h w',c=color)

        return patches

    def exec_agg_module_test(self,name,sigma,device="cuda:0"):

        # -- get data --
        clean = self.load_burst(name).to(device)
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape

        # -- get patches --
        npatches,color,ps = 4624,3,5
        patches = self.format_patches(clean,noisy,npatches,color,ps)

        # -- create layers --
        agg0_layer = lidia.modules.Aggregation0(ps)
        agg0inds_layer = lidia.nl_modules.AggregationInds(ps)

        # -- exec fwds --
        s,e = 72*30,72*32
        psMid = ps//2
        print(patches[s:e,0,psMid,psMid])
        patches = rearrange(patches,'n c h w -> 1 n 1 (c h w)')
        agg0_out = agg0_layer.forward2fold(patches,h+8,w+8)
        print(agg0_out[0,0,:5,:5])#*100.)
        print(agg0_out[0,0,30:32])#*100.)
        print(agg0_out[0,0,30:32,30:32])
        print(agg0_out.max(),agg0_out.min())
        agg0_out /= agg0_out.max()
        print("agg0_out.shape: ",agg0_out.shape)
        save_burst(agg0_out,SAVE_DIR,"agg0_out")

        agg0inds_out = agg0inds_layer(patches,h+8,w+8)
        # print("agg0inds_out.shape: ",agg0inds_out.shape)
        # save_burst(agg0inds_out,SAVE_DIR,"agg0inds_out")


        # -- compare --
        error_vals = th.sum((agg0_out - agg0inds_out)**2).item()
        assert error_vals < 1e-10

    def test_agg_module(self):

        # -- test 1 --
        name,sigma = "davis_baseball_64x64",50.
        self.exec_agg_module_test(name,sigma)
