

import torch as th
from einops import rearrange,repeat
from lidia.model_io import get_default_opt
import dnls
from lidia.nl_modules import save_image,save_burst

def denoise(patches,bufs,args):

    # -- get model --
    model = args.lidia_model
    opt = get_default_opt(args.sigma)
    t = args.t
    ps = args.ps
    pt = args.pt

    # -- patches --
    assert args.nlevels == 2
    levels = patches.levels
    pnoisy0 = format_patches(patches[levels[0]].noisy,args.t)
    pnoisy1 = format_patches(patches[levels[1]].noisy,args.t)

    # -- non-local (values,inds) --
    nlDists0,nlInds0 = format_nl(bufs[levels[0]],args.t)
    nlDists1,nlInds1 = format_nl(bufs[levels[1]],args.t)

    # -- update dist vals --
    update_dists(nlDists0,patches[levels[0]].noisy)
    update_dists(nlDists1,patches[levels[1]].noisy)

    # -- save ex --
    shape = (t,3,64,64)
    _pnoisy = patches[levels[0]].noisy
    _nlInds = nlInds0[:,:,0].view(-1,1,3)
    _nlDists = th.ones_like(nlDists0)[:,:,0].view(-1,1)
    dnoisy,wd = dnls.simple.gather.run(_pnoisy,_nlDists,_nlInds,shape=shape)
    dnoisy /= wd
    save_burst(dnoisy,"pre_dnoisy")

    delta = th.sum((pnoisy0 - pnoisy1)**2).item()
    print("delta: ",delta)
    assert delta > 0.

    # -- exec --
    dnoisy = model(model,pnoisy0,nlDists0,nlInds0,pnoisy1,nlDists1,nlInds1)
    print("dnoisy[min,max]: ",dnoisy.min().item(),dnoisy.max().item())
    # print("weight[min,max]: ",weights.min().item(),weights.max().item())
    # print(weights.shape)
    print("dnoisy.shape: ",dnoisy.shape)
    save_burst(dnoisy,"dnoisy")

    # -- re-scatter --
    print("nlInds0.shape: ",nlInds0.shape)
    _nlInds = rearrange(nlInds0[:,:,0],'n p thr -> (n p) 1 thr')
    dnoisy = dnls.simple.scatter.run(dnoisy,_nlInds,ps,pt)
    print("dnoisy.shape: ",dnoisy.shape)

    # -- reshape --
    pnoisy = dnoisy
    # pnoisy = rearrange(dnoisy,'t p d -> (t p) 1 d',t=args.t)
    # pnoisy = rearrange(pnoisy,'r 1 (c h w) -> r 1 1 c h w',h=ps,w=ps)
    # weights = rearrange(weights,'t p 1 -> (t p) 1')

    # -- rescale --
    # pnoisy = 255.*(pnoisy*0.5+0.5)
    print("pnoisy[min,max]: ",pnoisy.min().item(),pnoisy.max().item())

    # -- fill --
    print("patches[levels[0]].noisy.shape: ",patches[levels[0]].noisy.shape)
    print("pnoisy.shape: ",pnoisy.shape)
    patches[levels[0]].noisy[:,:1,...] = pnoisy

    print("bufs[levels[0]].vals[:,:1].shape: ",bufs[levels[0]].vals.shape)
    print("levels[0]: ",levels[0])
    bufs[levels[0]].vals[:,:1] = 1.
    print(bufs[levels[0]].vals[:,0])
    # bufs[levels[0]].vals[:,:1] = weights

    th.cuda.synchronize()

def update_dists(nlDists,pnoisy):

    # -- unpack/rescape --
    t,npt,k = nlDists.shape
    pnoisy = rescale_patches(pnoisy)
    pnoisy = pnoisy.view(t,npt,k,-1)

    # -- reompute dist --
    nlDists[:,:,:-1] = th.mean((pnoisy[:,:,1:,:] - pnoisy[:,:,[0],:])**2,dim=-1)

    # -- fill var --
    dim = pnoisy.shape[-1]
    patch_var = th.std(pnoisy[:,:,0],-1)**2 * dim
    nlDists[:,:,-1] = patch_var

def format_patches(pnoisy,t):
    pnoisy = rearrange(pnoisy,'r k t c h w -> r k (t c h w)')
    pnoisy = rearrange(pnoisy,'(t p) k d -> t p k d',t=t)
    pnoisy = rescale_patches(pnoisy)
    return pnoisy

def rescale_patches(patches):
    return (patches/255. - 0.5)/0.5

def format_nl(bufs,t):
    nlDists = rearrange(bufs.vals,'(t p) k -> t p k',t=t)
    nlInds = rearrange(bufs.inds,'(t p) k thr -> t p k thr',t=t)
    return nlDists,nlInds

