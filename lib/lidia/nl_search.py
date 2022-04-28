"""

Run Exhaustive Non-Local Search

"""

# -- linalg --
import torch as th

# -- data mngmnt --
from easydict import EasyDict as edict

# -- vision -
from torch.nn.functional import pad as pad_fxn
# from torchvision.transforms.functional import pad as pad_fxn

# -- lidia --
import lidia
import lidia.alloc_dnls as alloc
import lidia.search_mask as search_mask
import lidia.search as search
import lidia.utils as utils
from lidia.utils.lidia_utils import calc_padding_rgb,get_image_params
from lidia.params import get_params,get_args
from .nl_modules import NonLocalDenoiser as NLD

def run_search(noisy, sigma,
               pm_vid=None, flows=None, gpuid=0, clean=None, train=False, verbose=True):
    """
    Lightweight Image Denoising with Instance Adaptation (LIDIA)

    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Search Setup
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- get device --
    use_gpu = th.cuda.is_available() and gpuid >= 0
    device = 'cuda:%d' % gpuid if use_gpu else 'cpu'

    # -- to tensor --
    if not th.is_tensor(noisy):
        noisy = th.from_numpy(noisy).to(device)

    # -- setup lidia inputs --
    t,c,h,w = noisy.shape
    params = get_params(sigma,verbose,"default")
    flows = alloc.allocate_flows(flows,noisy.shape,noisy.device)
    params.srch_img = ["noisy","noisy"]

    # -- args --
    t,c,h,w = noisy.shape
    args = get_args(params,t,c,0,noisy.device)
    means = noisy.mean(dim=(-2,-1),keepdim=True)
    noisy -= means

    # -- prepare image --
    noisy_nn0,clean_nn0,params0 = get_nn0(noisy,clean,train,args)
    noisy_nn1,clean_nn1,params1 = get_nn1(noisy,clean,train,args)

    # -- allocs and args --
    images = alloc.allocate_images(noisy,means,None,clean)
    images.noisy_nn0 = noisy_nn0
    images.clean_nn0 = clean_nn0
    images.noisy_nn1 = noisy_nn1
    images.clean_nn1 = clean_nn1

    # -- number of elems --
    hb,wb = params0['patches_h'],params0['patches_w']
    print(params0)
    print(params1)


    # -- get model --
    args.lidia_model = None
    args.deno = "lidia"
    args.bsize = t*hb*wb # Exhaustive
    args.rand_mask = False
    args.chnls = 1
    args.dilation = 1

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #     Create Masks for Patches
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- mask slices --

    # -- prepare masks [dil=1] --
    mask0,hs0,ws0 = get_search_mask(noisy,noisy_nn0,params0,args)
    mask1,hs1,ws1 = get_search_mask(noisy,noisy_nn1,params1,args)
    masks = [mask0,mask1]

    # -- param --
    search_dilations = [1,2]
    assert len(search_dilations) == args.nlevels

    # -- allocate memory --
    patches = alloc.allocate_patches(args.patch_shape,images.clean,
                                     args.device,args.nlevels)
    bufs = alloc.allocate_bufs(args.bufs_shape,args.device,args.nlevels)

    # -- batching params --
    nelems,nbatches = utils.batching.batch_params(mask1,args.bsize,args.nstreams)
    cmasked_prev = nelems
    print(mask1[0].sum())

    # -- exec search --
    for level in range(args.nlevels):
        print("LEVEL: ",level)
        key = patches.levels[level]
        args.dilation = search_dilations[level]
        mask_l = masks[level]
        if level == 0: args.srch_img = "noisy_nn0"
        elif level == 1: args.srch_img = "noisy_nn1"
        else: raise ValueError("Uknown level.")
        search.exec_search(patches[key],images,flows,mask_l,bufs[key],args)

    # -- rescale inds --
    i0 = bufs["s0"].inds.view(t,hb,wb,args.k,3)
    i0[...,1] -= hs0
    i0[...,2] -= ws0
    i1 = bufs["s1"].inds.view(t,hb,wb,args.k,3)
    i1[...,1] -= hs1
    i1[...,2] -= ws1
    print(i0[0,:3,:3,0])
    print(i1[0,:3,:3,0])

    # -- packup results --
    res = edict()
    res.p0 = patches["s0"].noisy.view(t,hb,wb,args.k,-1)
    res.i0 = i0
    res.d0 = bufs["s0"].vals.view(t,hb,wb,args.k)
    res.p1 = patches["s1"].noisy.view(t,hb,wb,args.k,-1)
    res.i1 = i1
    res.d1 = bufs["s1"].vals.view(t,hb,wb,args.k)

    return res

def get_search_mask(noisy,noisy_pad,iparams,args):

    # -- get base shape --
    t,c,h,w = noisy.shape
    hb,wb = iparams['patches_h'],iparams['patches_w']
    ppad = 2*(args.ps//2) # patches used to pad interior image
    # hb,wb = h+ppad,w+ppad

    # -- get excess pix --
    _,_,hp,wp = noisy_pad.shape
    hs,ws = (hp - hb)//2,(wp - wb)//2
    print("-"*30)
    print("noisy.shape: ",noisy.shape)
    print("noisy_pad.shape: ",noisy_pad.shape)
    print(hs,ws)
    print(hb,wb)
    print(iparams)
    print("-"*30)

    # -- compute center of base mask --
    hslice = slice(hs,hb+hs)
    wslice = slice(ws,wb+ws)

    # -- create mask --
    t,device = noisy.shape[0],noisy.device
    mask = th.zeros((t,hp,wp),device=device,dtype=th.bool)
    mask[:,hslice,wslice] = 1

    return mask,hs,ws

def get_nn0(noisy,clean,train,args):

    # -- pad-crop noisy --
    neigh_pad = 14
    pad_offs,_ = calc_padding_rgb(args.ps)
    noisy = NLD._pad_crop0(noisy,pad_offs,train,args.ps)
    if not(clean is None):
        clean = NLD._pad_crop0(clean,pad_offs,train,args.ps)

    # -- indices correction --
    params = get_image_params(noisy,args.ps,neigh_pad)

    return noisy,clean,params

def get_nn1(noisy,clean,train,args):

    # -- pad --
    neigh_pad = 14

    # -- noisy -
    noisy = NLD._pad_crop1(noisy,train,'reflect',args.ps)
    if train: noisy = noisy.contiguous()
    noisy = exec_bilinear_conv(noisy)
    noisy = NLD._pad_crop1(noisy,train,'constant',args.ps)

    # -- noisy -
    if not(clean is None):
        clean = NLD._pad_crop1(clean,train,'reflect',args.ps)
        if train: clean = clean.contiguous()
        clean = exec_bilinear_conv(clean)
        clean = NLD._pad_crop1(clean,train,'constant',args.ps)

    # -- indices correction --
    params = get_image_params(noisy,2*args.ps-1,2*neigh_pad)

    return noisy,clean,params


def exec_bilinear_conv(vid):

    # -- create --
    kernel_1d = th.tensor((1 / 4, 1 / 2, 1 / 4), dtype=th.float32)
    kernel_2d = (kernel_1d.view(-1, 1) * kernel_1d).view(1, 1, 3, 3)
    nn_bilinear_conv = th.nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=(3, 3), bias=False)
    nn_bilinear_conv.weight.data = kernel_2d
    nn_bilinear_conv.weight.requires_grad = False
    nn_bilinear_conv = nn_bilinear_conv.to(vid.device)

    # -- exec --
    t,c,h,w = vid.shape
    vid = vid.view(t*c,1,h,w)
    vid = nn_bilinear_conv(vid)
    vid = vid.view(t,c,h-2,w-2)

    return vid



