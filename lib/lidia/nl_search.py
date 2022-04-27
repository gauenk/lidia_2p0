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
import lidia.alloc_dnls as alloc
import lidia.search_mask as search_mask
import lidia.search as search
import lidia.utils as utils
from lidia.params import get_params,get_args


def run_search(noisy, sigma,
               pm_vid=None, flows=None, gpuid=0, clean=None, verbose=True):
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
    noisy_nn0,clean_nn0,indCor0 = get_lidia_padded_img_nn0(noisy,clean,args)
    noisy_nn1,clean_nn1,indCor1 = get_lidia_padded_img_nn1(noisy,clean,args)

    # -- allocs and args --
    images = alloc.allocate_images(noisy,means,None,clean)
    images.noisy_nn0 = noisy_nn0
    images.clean_nn0 = clean_nn0
    images.noisy_nn1 = noisy_nn1
    images.clean_nn1 = clean_nn1
    print("noisy.shape: ",noisy.shape)
    print("images.noisy_nn0.shape: ",images.noisy_nn0.shape)

    # -- number of elems --
    hb = h + 2*(args.ps//2)
    wb = w + 2*(args.ps//2)

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
    mask0 = get_search_mask(noisy,noisy_nn0,args)
    mask1 = get_search_mask(noisy,noisy_nn1,args)
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

    # -- exec search --
    for level in range(args.nlevels):
        key = patches.levels[level]
        args.dilation = search_dilations[level]
        mask_l = masks[level]
        if level == 0: args.srch_img = "noisy_nn0"
        elif level == 1: args.srch_img = "noisy_nn1"
        else: raise ValueError("Uknown level.")
        search.exec_search(patches[key],images,flows,mask_l,bufs[key],args)

    # -- rescale inds --
    i0 = bufs["s0"].inds.view(t,hb,wb,args.k,3)
    i0[...,1] -= indCor0[0]
    i0[...,2] -= indCor0[1]
    i1 = bufs["s1"].inds.view(t,hb,wb,args.k,3)
    i1[...,1] -= indCor1[0]
    i1[...,2] -= indCor1[1]

    # -- packup results --
    res = edict()
    res.p0 = patches["s0"].noisy
    res.i0 = i0
    res.d0 = bufs["s0"].vals.view(t,hb,wb,args.k)
    res.p1 = patches["s1"].noisy
    res.i1 = i1
    res.d1 = bufs["s1"].vals.view(t,hb,wb,args.k)

    return res


def get_lidia_padded_img_nn0(noisy,clean,args):
    pad_r = 2*(args.ps//2)
    pad_c = 2*(args.ps//2)
    noisy = pad_fxn(noisy,[pad_r,]*4,"reflect")
    noisy = pad_fxn(noisy,[pad_c,]*4,"constant",-1)
    if not(clean is None):
        clean = pad_fxn(clean,[pad_r,]*4,"reflect")
        clean = pad_fxn(clean,[pad_c,]*4,"constant",-1)
    print("[padding nn0] noisy.shape: ",noisy.shape)

    # -- indices correction --
    indCor = [6,6]

    return noisy,clean,indCor

def get_search_mask(noisy,noisy_pad,args):

    # -- get base shape --
    t,c,h,w = noisy.shape
    hb = h + 2*(args.ps//2)
    wb = w + 2*(args.ps//2)

    # -- get excess pix --
    _,_,hp,wp = noisy_pad.shape
    hs = (hp - hb)//2
    ws = (wp - wb)//2

    # -- compute center of base mask --
    hslice = slice(hs,hb+hs)
    wslice = slice(hs,wb+hs)

    # -- create mask --
    t,device = noisy.shape[0],noisy.device
    mask = th.zeros((t,hp,wp),device=device,dtype=th.bool)
    mask[:,hslice,wslice] = 1

    return mask

def pad_crop1_reflect(noisy,args):
    bilinear_pad = 1
    averaging_pad = (args.ps - 1) // 2
    patch_w_scale_1 = 2 * args.ps - 1
    find_nn_pad = (patch_w_scale_1 - 1) // 2
    reflect_pad = [averaging_pad + bilinear_pad + find_nn_pad] * 4
    noisy = pad_fxn(noisy, reflect_pad, 'reflect')
    return noisy

def pad_crop1_constant(noisy,args):
    constant_pad = [28] * 4
    noisy = pad_fxn(noisy, constant_pad, 'constant', -1)
    return noisy

def get_lidia_padded_img_nn1(noisy,clean,args):

    # -- pad --
    noisy = pad_crop1_reflect(noisy,args)
    print("[reflect] noisy.shape: ",noisy.shape)

    # -- filter --
    t,c,h,w = noisy.shape
    noisy = noisy.view(t*c,1,h,w)
    noisy = exec_bilinear_conv(noisy)
    noisy = noisy.view(t,c,h-2,w-2)
    print("[reflect] noisy.shape: ",noisy.shape)

    # -- bilinear conv & crop --
    noisy = pad_crop1_constant(noisy, args)

    # -- indices correction --
    indCor = [32,32]

    return noisy,clean,indCor


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



