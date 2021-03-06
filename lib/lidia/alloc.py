
"""
Allocate memory once for many subroutines
"""

import torch as th
from easydict import EasyDict as edict

def allocate_patches(shape,clean,device):

    # -- unpack shapes --
    tsize,npa,ps_t,c,ps,ps = shape
    tf32 = th.float32

    # -- alloc mem --
    patches = edict()
    patches.noisy = th.zeros((tsize,npa,ps_t,c,ps,ps)).type(tf32).to(device)
    patches.basic = th.zeros((tsize,npa,ps_t,c,ps,ps)).type(tf32).to(device)
    patches.clean = None
    if not(clean is None):
        patches.clean = th.zeros((tsize,npa,ps_t,c,ps,ps)).type(tf32).to(device)
    patches.flat = th.zeros((tsize)).type(th.bool).to(device)
    patches.shape = shape

    # -- names --
    patches.images = ["noisy","basic","clean"]
    patches.tensors = ["noisy","basic","clean","flat"]

    return patches

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
        imgs.basic = th.zeros((t,c,h,w),dtype=dtype,device=device)

    # -- clean --
    imgs.clean = clean
    if not(clean is None) and not(th.is_tensor(clean)):
        imgs.clean = th.from_numpy(imgs.clean).to(device)

    # -- search --
    imgs.search = search

    # -- deno & agg weights --
    imgs.deno = th.zeros((t,c,h,w),dtype=dtype,device=device)
    imgs.weights = th.zeros((t,h,w),dtype=dtype,device=device)
    imgs.vals = th.zeros((t,h,w),dtype=dtype,device=device)

    # -- means --
    imgs.means = th.zeros((t,c,1,1),dtype=th.float32,device=device)

    # -- names --
    imgs.patch_images = ["noisy","basic","clean"]
    imgs.ikeys = ["noisy","basic","clean","deno","search"]

    return imgs

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
#
# -- Allocate Non-Local Search Buffers --
#
#

def allocate_bufs(shape,device,atype):
    if atype == "eccv2022":
        return allocate_bufs_eccv2022(shape,device)
    elif atype == "faiss":
        return allocate_bufs_faiss(shape,device)
    elif atype == "dnls":
        return allocate_bufs_dnls(shape,device)
    else:
        raise ValueError("Uknown bufs allocate type [{atype}]")

def allocate_bufs_eccv2022(shape,device):

    # -- unpack shapes --
    tsize,npa = shape
    tf32 = th.float32
    tl = th.long

    # -- alloc mem --
    l2bufs = edict()
    l2bufs.vals = th.zeros((tsize,npa),dtype=tf32,device=device)
    l2bufs.inds = -th.ones((tsize,npa),dtype=tl,device=device)
    l2bufs.shape = shape
    l2bufs.tensors = ["vals","inds"]

    return l2bufs

def allocate_bufs_faiss(shape,device):

    # -- unpack shapes --
    tsize,npa = shape
    tf32 = th.float32
    ti32 = th.int32

    # -- alloc mem --
    l2bufs = edict()
    l2bufs.vals = th.zeros((tsize,npa),dtype=tf32,device=device)
    l2bufs.inds = -th.ones((tsize,npa),dtype=ti32,device=device)
    l2bufs.shape = shape
    l2bufs.tensors = ["vals","inds"]

    return l2bufs

def allocate_bufs_dnls(shape,device):

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



