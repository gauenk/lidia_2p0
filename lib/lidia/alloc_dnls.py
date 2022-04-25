

"""
Allocate memory once for many subroutines
"""

import torch as th
from easydict import EasyDict as edict

#
# -- Get Patches --
#

def allocate_patches(shape,clean,device,nlevels):
    tsize,npa,pt,c,ps,ps = shape
    patches = edict()
    levels = []
    for level in range(nlevels):
        key = "s%d" % level
        levels.append(key)
        patches[key] = allocate_patches_level(shape,clean,device)
    patches.shape = shape
    patches.levels = levels
    return patches

def allocate_patches_level(shape,clean,device):

    # -- init --
    tf32 = th.float32
    tsize = shape[0]

    # -- alloc mem --
    patches = edict()
    patches.noisy = th.zeros(shape,device=device,dtype=tf32)
    patches.basic = th.zeros(shape,device=device,dtype=tf32)
    patches.clean = None
    if not(clean is None):
        patches.clean = th.zeros(shape,device=device,dtype=tf32)
    patches.flat = th.zeros(tsize,device=device,dtype=th.bool)
    patches.shape = shape

    # -- names --
    patches.images = ["noisy","basic","clean"]
    patches.tensors = ["noisy","basic","clean","flat"]

    return patches

#
# -- Optical Flow --
#

def allocate_flows(flows,shape,device):
    t,c,h,w = shape
    if flows is None:
        flows = edict()
        zflow = th.zeros((t,2,h,w)).to(device)
        flows.fflow = zflow
        flows.bflow = zflow.clone()
    else:
        flows = edict({k:v.to(device) for k,v in flows.items()})
    return flows

#
# -- Allocate Image Memory --
#

def allocate_images(noisy,basic,clean,search=None):

    # -- create images --
    imgs = edict()
    imgs.noisy = noisy
    imgs.shape = noisy.shape
    imgs.device = noisy.device

    # -- unpack params --
    dtype = noisy.dtype
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- basic --
    imgs.basic = basic
    if basic is None:
        imgs.basic = th.zeros((t,c,h,w),dtype=dtype).to(device)

    # -- clean --
    imgs.clean = clean
    if not(clean is None) and not(th.is_tensor(clean)):
        imgs.clean = th.from_numpy(imgs.clean).to(device)

    # -- search --
    imgs.search = search

    # -- deno & agg weights --
    imgs.deno = th.zeros((t,c,h,w),dtype=dtype).to(device)
    imgs.weights = th.zeros((t,c,h,w),dtype=dtype).to(device)
    imgs.vals = th.zeros((t,h,w),dtype=dtype).to(device)

    # -- means --
    imgs.means = th.zeros((t,c,1,1),dtype=th.float32,device=device)

    # -- names --
    imgs.patch_images = ["noisy","basic","clean"]
    imgs.ikeys = ["noisy","basic","clean","deno","search"]

    return imgs

#
# -- Allocate Non-Local Search Buffers --
#

def allocate_bufs(shape,device,nlevels):
    bufs = edict()
    levels = []
    for level in range(nlevels):
        key = "s%d" % level
        levels.append(key)
        bufs[key] = allocate_bufs_level(shape,device)
    bufs.shape = shape
    bufs.levels = levels
    return bufs

def allocate_bufs_level(shape,device):

    # -- unpack shapes --
    tsize,npa = shape
    tf32 = th.float32
    ti32 = th.int32

    # -- alloc mem --
    l2bufs = edict()
    l2bufs.vals = th.zeros((tsize,npa),dtype=tf32,device=device)
    l2bufs.inds = -th.ones((tsize,npa,3),dtype=ti32,device=device)
    l2bufs.shape = shape
    l2bufs.tensors = ["vals","inds"]

    return l2bufs

