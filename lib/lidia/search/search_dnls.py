# -- python imports --
import torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- [a required package] --
import vpss

# -- local package --
import lidia.search_mask as search_mask
from lidia.utils.batching import view_batch
from lidia.utils.logger import vprint

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
    else:
        raise ValueError(f"uknown search image [{args.srch_img}]")
    # srch_img = imgs.noisy if args.step == 0 else imgs.basic
    # srch_img = srch_img if (imgs.clean is None) else imgs.clean

    # -- color correct before search --
    srch_img = exec_rgb2gray(srch_img)

    # -- sim search block --
    bufs.inds[...] = th.iinfo(th.int32).min
    bufs.vals[...] = float("inf")
    vals,inds = dnls.simple.search.run(srch_img,srch_inds,flows,args.k,
                                       args.ps,args.pt,args.ws,args.wt,1,
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
        imgs_k = rescale_img(imgs[key])
        if key == "noisy":
            imgs.means[...] = imgs_k.mean((-1,-2),keepdim=True)
            print("imgs_k.shape: ",imgs_k.shape)
            imgs_k -= imgs.means
            print_stats("[search_dnls] noisy",imgs_k)

        # -- fill --
        pkey = dnls.simple.scatter.run(imgs_k,bufs.inds,
                                       args.ps,args.pt,dilation=args.dilation)
        patches[key][...] = pkey[...]

def exec_rgb2gray(image_rgb):
    rgb2gray = th.nn.Conv2d(in_channels=3, out_channels=1,
                            kernel_size=(1, 1), bias=False)
    rgb2gray.weight.data = torch.tensor([0.2989, 0.5870, 0.1140],
                                        dtype=torch.float32).view(1, 3, 1, 1)
    rgb2gray.weight.requires_grad = False
    rgb2gray = rgb2gray.to(image_rgb.device)
    image_g = rgb2gray(image_rgb)
    print("image_rgb.shape: ",image_rgb.shape)
    print("image_g.shape: ",image_g.shape)
    return image_g

def rescale_img(img):
    return (img/255. - 0.5)/0.5

def print_stats(name,tensor):
    imin,imax = tensor.min().item(),tensor.max().item()
    imean = tensor.mean().item()
    label = "%s[min,max,mean]: " % name
    print(label,imin,imax,imean)

