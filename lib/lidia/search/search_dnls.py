# -- python imports --
import torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- [a required package] --
# import vpss

# -- local package --
import lidia.search_mask as search_mask
from lidia.utils.batching import view_batch
from lidia.utils.logger import vprint

# -- filtering --
from torchvision.transforms.functional import pad as pad_fxn

# -- search packages --
import dnls

def exec_search_dnls(patches,imgs,flows,mask,bufs,args):

    # -- setup --
    bsize = args.bsize
    cs = th.cuda.default_stream()
    cs_ptr = th.cuda.default_stream().cuda_stream
    done = False

    # --reset values --
    bufs.inds[...] = -1
    bufs.vals[...] = float("inf")

    # -- smaller batch sizes impact quality --
    for index in range(args.nstreams):

        # -- grab access --
        print("total: ",mask.sum().item(),args.bsize)
        srch_inds = search_mask.mask2inds(mask,bsize,args.rand_mask)
        if srch_inds.shape[0] == 0:
            done = True
            break

        # -- grab batch --
        vbufs = edict()
        for key in bufs.keys():
            vbufs[key] = view_batch(bufs[key],index,bsize)

        vpatches = edict()
        for key in patches.keys():
            vpatches[key] = view_batch(patches[key],index,bsize)

        # -- exec search --
        search_and_fill(imgs,vpatches,vbufs,srch_inds,flows,args)

        # -- update mask naccess --
        before = mask.sum().item()
        search_mask.update_mask_inds(mask,vbufs.inds,args.c,args.nkeep,
                                     boost=args.aggreBoost)
        after = mask.sum().item()

        # -- wait for all streams --
        th.cuda.synchronize()

    # -- update term. condition --
    done = done or (mask.sum().item() == 0)

    return done

def search_and_fill(imgs,patches,bufs,srch_inds,flows,args):

    # -- select search image --
    if args.srch_img == "noisy":
        srch_img = imgs.noisy
    elif args.srch_img == "basic":
        srch_img = imgs.basic
    elif args.srch_img == "clean":
        srch_img = imgs.clean
    elif args.srch_img == "search":
        srch_img = imgs.search
    elif args.srch_img == "noisy_nn0":
        srch_img = imgs.noisy_nn0
    elif args.srch_img == "noisy_nn1":
        srch_img = imgs.noisy_nn1
    else:
        raise ValueError(f"uknown search image [{args.srch_img}]")

    # -- color correct before search --
    srch_img = exec_rgb2gray(srch_img)

    # -- sim search block --
    bufs.inds[...] = th.iinfo(th.int32).min
    bufs.vals[...] = float("inf")
    vals,inds = dnls.simple.search.run(srch_img,srch_inds,flows,args.k,
                                       args.ps,args.pt,args.ws,args.wt,1,
                                       stride=args.dilation,
                                       dilation=args.dilation)
    nq = vals.shape[0]
    bufs.vals[:nq,...] = vals[...]
    bufs.inds[:nq,...] = inds[...]
    th.cuda.synchronize()

    # -- get invalid --
    # tim = th.iinfo(th.int32).min
    # invalid = th.where(th.any(th.any(bufs.inds==tim,2),1))

    # # -- ensure 1st location is self [no matter what] --
    # t,c,h,w = srch_img.shape
    # flat_inds = get_flat_inds(srch_inds,c,h,w)

    # # -- fill partial with first inds --
    # bsize = flat_inds.shape[0]
    # bufs.inds[:bsize,0] = flat_inds
    # bufs.vals[:,0] = 0.

    # -- fill patches --
    for key in imgs.patch_images:

        # -- skip --
        pass_key = (imgs[key] is None) or (patches[key] is None)
        if pass_key: continue

        # -- prepare image --
        imgs_k = imgs[key]
        if key == "noisy":
            if args.dilation == 1:
                imgs_k = imgs["noisy_nn0"]
            elif args.dilation == 2:
                imgs_k = imgs["noisy_nn1"]
            else:
                raise ValueError("Uknown dilation.")

        # -- fill --
        pkey = dnls.simple.scatter.run(imgs_k,bufs.inds,
                                       args.ps,args.pt,
                                       dilation=args.dilation)
        if key == "noisy":
            print("imgs_k.shape: ",imgs_k.shape)
            print("imgs[key].shape: ",imgs[key].shape)
            # print(imgs_k[:,10:12,10:12])
            print("pkey.shape: ",pkey.shape)
            # print(pkey)
        patches[key][...] = pkey[...]

def exec_rgb2gray(image_rgb):
    rgb2gray = th.nn.Conv2d(in_channels=3, out_channels=1,
                            kernel_size=(1, 1), bias=False)
    rgb2gray.weight.data = torch.tensor([0.2989, 0.5870, 0.1140],
                                        dtype=torch.float32).view(1, 3, 1, 1)
    rgb2gray.weight.requires_grad = False
    rgb2gray = rgb2gray.to(image_rgb.device)
    image_g = rgb2gray(image_rgb)
    return image_g

def rescale_img(img):
    return (img/255. - 0.5)/0.5

def print_stats(name,tensor):
    imin,imax = tensor.min().item(),tensor.max().item()
    imean = tensor.mean().item()
    label = "%s[min,max,mean]: " % name
    print(label,imin,imax,imean)

def exec_filtering(vid,ps):
    pad = 1
    t,c,h,w = vid.shape
    vid = pad_fxn(vid, [pad]*4, padding_mode='reflect')
    vid = bilinear_conv(vid)
    return vid

def bilinear_conv(vid):

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

