
from torch.nn.functional import unfold
from torch.nn.functional import fold
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn.functional import conv2d
from .utils.lidia_utils import *
from torchvision.transforms.functional import center_crop
import torch as th
from pathlib import Path
from einops import repeat,rearrange

from .utils.io import save_burst,save_image
from .utils.logging import print_extrema
from .utils.inds import get_3d_inds

import dnls

import torch as th
from pathlib import Path
from einops import repeat

# -- separate class and logic --
from .utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

@register_method
def run_agg0(self,patches,dist0,inds0,h,w):

    # -- compute weights --
    pdn = self.patch_denoise_net
    weights0 = pdn.weights_net0(th.exp(-pdn.alpha0.abs() * dist0)).unsqueeze(-1)
    weighted_patches = patches * weights0

    # -- compute sep+agg --
    sep_net = self.patch_denoise_net.separable_fc_net
    x0 = sep_net.sep_part1_s0(weighted_patches)
    y_out0 = sep_net.ver_hor_agg0_pre(x0)
    y_out0,fold_out0 = sep_net.agg0(y_out0, dist0, inds0, h, w,both=True)
    y_out0 = sep_net.ver_hor_bn_re_agg0_post(y_out0)

    return y_out0,x0,fold_out0

@register_method
def run_agg1(self,patches,dist1,inds1,h,w):

    # -- compute weights --
    pdn = self.patch_denoise_net
    weights1 = pdn.weights_net1(th.exp(-pdn.alpha1.abs() * dist1)).unsqueeze(-1)
    weighted_patches = patches * weights1
    weights1 = weights1[:, :, 0:1, :]
    wpatches = weighted_patches

    # -- compute patches output --
    sep_net = self.patch_denoise_net.separable_fc_net
    x1 = sep_net.sep_part1_s1(wpatches)
    y_out1 = sep_net.ver_hor_agg1_pre(x1)
    y_out1,fold_out1 = sep_net.agg1(y_out1 / weights1,dist1, inds1,h,w,both=True)
    y_out1 = weights1 * y_out1
    y_out1 = sep_net.ver_hor_bn_re_agg1_post(y_out1)

    return y_out1,x1,fold_out1,wpatches

@register_method
def run_nn0_dnls_search(self,image_n,train=False):

    #
    # -- Our Search --
    #

    # -- pad & unpack --
    device = image_n.device
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)
    ps = self.patch_w
    pad = ps//2

    # -- get search image --
    if self.arch_opt.rgb: img_nn0 = self.rgb2gray(image_n0)
    else: img_nn0 = image_n0

    # -- get search inds --
    t,c,h,w = image_n.shape
    hp,wp = h+2*pad,w+2*pad
    queryInds = th.arange(t*hp*wp,device=device).reshape(-1,1,1,1)
    queryInds = get_3d_inds(queryInds,hp,wp)[:,0]
    t,c,h0,w0 = image_n0.shape
    sh,sw = (h0 - hp)//2,(w0 - wp)//2
    queryInds[...,1] += sh
    queryInds[...,2] += sw

    # -- search --
    ps = self.patch_w
    k,pt,ws,wt,chnls = 14,1,29,0,1
    nlDists,nlInds = dnls.simple.search.run(img_nn0,queryInds,None,
                                            k,ps,pt,ws,wt,chnls)

    # -- rename dists,inds --
    top_dist0 = nlDists
    top_ind0 = nlInds

    #
    # -- Scatter Section --
    #

    # -- indexing patches --
    t,c,h,w = image_n0.shape
    patches = dnls.simple.scatter.run(image_n0,top_ind0,self.patch_w)
    ishape = '(t p) k 1 c h w -> t p k (c h w)'
    patches = rearrange(patches,ishape,t=t)

    # -- rehape --
    sp = int(np.sqrt(patches.shape[1]))
    patches = rearrange(patches,'t (h w) k d -> t h w k d',h=sp)
    top_ind0 = rearrange(top_ind0,'(t h w) k tr -> t h w k tr',t=t,h=sp)
    top_dist0 = rearrange(top_dist0,'(t h w) k -> t h w k',t=t,h=sp)

    # -- append anchor patch spatial variance --
    d = patches.shape[-1]
    patch_dist0 = top_dist0[...,1:]
    patch_var0 = patches[..., [0], :].std(dim=-1).pow(2)*d
    patch_dist0 = th.cat((patch_dist0, patch_var0), dim=-1)

    # -- rescale --
    top_ind0[...,1] -= sw
    top_ind0[...,2] -= sw

    return patches,patch_dist0,top_ind0

@register_method
def run_nn0_lidia_search(self,image_n,train=False):

    #
    # -- Lidia Search --
    #

    # -- pad & unpack --
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    device = image_n.device
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)

    # -- get search image --
    if self.arch_opt.rgb: img_nn0 = self.rgb2gray(image_n0)
    else: img_nn0 = image_n0

    # -- get image-based parameters --
    im_params0 = get_image_params(image_n0, self.patch_w, 14)
    im_params0['pad_patches_w_full'] = im_params0['pad_patches_w']

    # -- run knn search --
    top_dist0, top_ind0 = self.find_nn(img_nn0, im_params0, self.patch_w)

    # -- prepare [dists,inds] --
    ip = im_params0['pad_patches']
    patch_dist0 = top_dist0.view(top_dist0.shape[0], -1, 14)[:, :, 1:]
    top_ind0 += ip * th.arange(top_ind0.shape[0],device=device).view(-1, 1, 1, 1)

    #
    # -- Our Section --
    #

    # -- get new inds --
    ps = self.patch_w
    t,c,h,w = image_n0.shape
    ch,cw = h-(ps-1),w-(ps-1)
    image_n0 = center_crop(image_n0,(ch,cw))

    # -- get new inds
    t,c,h,w = image_n0.shape
    dnls_inds = get_3d_inds(top_ind0,h,w)

    # -- indexing patches --
    im_patches_n0 = dnls.simple.scatter.run(image_n0,dnls_inds,self.patch_w)
    ishape = '(t p) k 1 c h w -> t p k (c h w)'
    im_patches_n0 = rearrange(im_patches_n0,ishape,t=t)

    # -- append anchor patch spatial variance --
    patch_var0 = im_patches_n0[:, :, 0, :].std(dim=-1).\
        unsqueeze(-1).pow(2) * patch_numel
    patch_dist0 = th.cat((patch_dist0, patch_var0), dim=-1)

    # -- rescale inds --
    dnls_inds[...,1] -= 14
    dnls_inds[...,2] -= 14

    # -- format [dists,inds] --
    t,h,w,k = top_dist0.shape
    dnls_inds = rearrange(dnls_inds,'(t h w) k tr -> t h w k tr',t=t,h=h)
    ip0 = im_patches_n0
    ip0 = rearrange(ip0,'t (h w) k d -> t h w k d',h=h)

    return ip0,top_dist0,dnls_inds

@register_method
def run_nn1_dnls_search(self,image_n,train=False):

    # -- unpack --
    ps = self.patch_w
    pad = 2*(ps-1) # dilation "= 2"

    # -- pad & unpack --
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    device = image_n.device
    image_n1 = self.pad_crop1(image_n, train, 'reflect')
    im_n1_b, im_n1_c, im_n1_h, im_n1_w = image_n1.shape

    # -- bilinear conv & crop --
    image_n1 = image_n1.view(im_n1_b * im_n1_c, 1,im_n1_h, im_n1_w)
    image_n1 = self.bilinear_conv(image_n1)
    image_n1 = image_n1.view(im_n1_b, im_n1_c, im_n1_h - 2, im_n1_w - 2)
    image_n1 = self.pad_crop1(image_n1, train, 'constant')

    #
    #  -- DNLS Search --
    #

    # -- get search image --
    if self.arch_opt.rgb: img_nn1 = self.rgb2gray(image_n1)
    else: img_nn1 = image_n1

    # -- get search inds --
    pad_s = ps//2
    t,c,h,w = image_n.shape
    hp,wp = h+2*pad_s,w+2*pad_s # determines # of pix centers
    queryInds = th.arange(t*hp*wp,device=device).reshape(-1,1,1,1)
    # hp,wp = h+2*pad,w+2*pad # determines what pixel inds mean
    queryInds = get_3d_inds(queryInds,hp,wp)[:,0]
    t,c,h0,w0 = image_n1.shape
    sh,sw = (h0 - hp)//2,(w0 - wp)//2
    queryInds[...,1] += sh
    queryInds[...,2] += sw

    # -- exec search --
    k,pt,ws,wt,chnls = 14,1,29,0,1
    nlDists,nlInds = dnls.simple.search.run(img_nn1,queryInds,None,
                                            k,ps,pt,ws,wt,chnls,
                                            stride=2,dilation=2)

    #
    # -- Scatter Section --
    #

    # -- dnls --
    pad = ps//2
    _t,_c,_h,_w = image_n.shape
    hp,wp = _h+2*pad,_w+2*pad
    patches = dnls.simple.scatter.run(image_n1,nlInds,ps,dilation=2)


    #
    # -- Final Formatting --
    #

    # - reshape --
    t,c,h,w = image_n.shape
    hp = int(np.sqrt(patches.shape[0]/t))
    nlDists = rearrange(nlDists,'(t h w) k -> t h w k',t=t,h=hp)
    nlInds = rearrange(nlInds,'(t h w) k tr -> t h w k tr',t=t,h=hp)
    ishape = '(t ih iw) k 1 c h w -> t ih iw k (c h w)'
    patches = rearrange(patches,ishape,ih=hp,iw=wp)

    # -- patch variance --
    d = patches.shape[-1]
    patch_var = patches[...,0,:].std(-1)**2*d
    nlDists[...,:-1] = nlDists[...,1:]
    nlDists[...,-1] = patch_var

    # -- centering inds --
    nlInds[...,1] -= sh
    nlInds[...,2] -= sw

    return patches,nlDists,nlInds

@register_method
def run_nn1_lidia_search(self,image_n,train=False):

    # -- unpack --
    ps = self.patch_w

    # -- pad & unpack --
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    device = image_n.device
    image_n1 = self.pad_crop1(image_n, train, 'reflect')
    im_n1_b, im_n1_c, im_n1_h, im_n1_w = image_n1.shape

    # -- bilinear conv & crop --
    image_n1 = image_n1.view(im_n1_b * im_n1_c, 1,im_n1_h, im_n1_w)
    image_n1 = self.bilinear_conv(image_n1)
    image_n1 = image_n1.view(im_n1_b, im_n1_c, im_n1_h - 2, im_n1_w - 2)
    image_n1 = self.pad_crop1(image_n1, train, 'constant')

    #
    #  -- LIDIA Search --
    #

    # -- img-based parameters --
    im_params1 = get_image_params(image_n1, 2 * self.patch_w - 1, 28)
    im_params1['pad_patches_w_full'] = im_params1['pad_patches_w']

    # -- get search image  --
    if self.arch_opt.rgb: img_nn1 = self.rgb2gray(image_n1)
    else: img_nn1 = image_n1

    # -- spatially split image --
    img_nn1_00 = img_nn1[:, :, 0::2, 0::2].clone()
    img_nn1_10 = img_nn1[:, :, 1::2, 0::2].clone()
    img_nn1_01 = img_nn1[:, :, 0::2, 1::2].clone()
    img_nn1_11 = img_nn1[:, :, 1::2, 1::2].clone()

    # -- get split image parameters --
    im_params1_00 = get_image_params(img_nn1_00, self.patch_w, 14)
    im_params1_10 = get_image_params(img_nn1_10, self.patch_w, 14)
    im_params1_01 = get_image_params(img_nn1_01, self.patch_w, 14)
    im_params1_11 = get_image_params(img_nn1_11, self.patch_w, 14)
    im_params1_00['pad_patches_w_full'] = im_params1['pad_patches_w']
    im_params1_10['pad_patches_w_full'] = im_params1['pad_patches_w']
    im_params1_01['pad_patches_w_full'] = im_params1['pad_patches_w']
    im_params1_11['pad_patches_w_full'] = im_params1['pad_patches_w']

    # -- run knn search! --
    top_dist1_00, top_ind1_00 = self.find_nn(img_nn1_00, im_params1_00,
                                             self.patch_w, scale=1, case='00')
    top_dist1_10, top_ind1_10 = self.find_nn(img_nn1_10, im_params1_10,
                                             self.patch_w, scale=1, case='10')
    top_dist1_01, top_ind1_01 = self.find_nn(img_nn1_01, im_params1_01,
                                             self.patch_w, scale=1, case='01')
    top_dist1_11, top_ind1_11 = self.find_nn(img_nn1_11, im_params1_11,
                                             self.patch_w, scale=1, case='11')

    # -- aggregate results [dists] --
    top_dist1 = th.zeros(im_params1['batches'], im_params1['patches_h'],
                            im_params1['patches_w'], 14, device=device)
    top_dist1 = top_dist1.fill_(float('nan'))
    top_dist1[:, 0::2, 0::2, :] = top_dist1_00
    top_dist1[:, 1::2, 0::2, :] = top_dist1_10
    top_dist1[:, 0::2, 1::2, :] = top_dist1_01
    top_dist1[:, 1::2, 1::2, :] = top_dist1_11

    # -- aggregate results [inds] --
    ipp = im_params1['pad_patches']
    top_ind1 = ipp * th.ones(top_dist1.shape,dtype=th.int64,device=device)
    top_ind1[:, 0::2, 0::2, :] = top_ind1_00
    top_ind1[:, 1::2, 0::2, :] = top_ind1_10
    top_ind1[:, 0::2, 1::2, :] = top_ind1_01
    top_ind1[:, 1::2, 1::2, :] = top_ind1_11
    top_ind1 += ipp * th.arange(top_ind1.shape[0],device=device).view(-1, 1, 1, 1)

    # -- inds rename --
    top_ind1_og = top_ind1

    #
    # -- Scatter Section --
    #

    # -- inds -> 3d_inds --
    pad = 2*(ps//2) # dilation "= 2"
    _t,_c,_h,_w = image_n1.shape
    hp,wp = _h-2*pad,_w-2*pad
    top_ind1 = get_3d_inds(top_ind1,hp,wp)

    # -- prepare inds for scatter
    sc_inds = top_ind1.clone()
    sc_inds[...,1] += pad
    sc_inds[...,2] += pad

    # -- dnls --
    pad = ps//2
    _t,_c,_h,_w = image_n.shape
    hp,wp = _h+2*pad,_w+2*pad
    im_patches_n1 = dnls.simple.scatter.run(image_n1,sc_inds,ps,dilation=2)
    ishape = '(t ih iw) k 1 c h w -> t (ih iw) k (c h w)'
    im_patches_n1 = rearrange(im_patches_n1,ishape,ih=hp,iw=wp)

    #
    # -- Final Formatting --
    #

    # -- re-shaping --
    t,h,w,k = top_dist1.shape
    top_ind1 = rearrange(top_ind1,'(t h w) k tr -> t h w k tr',t=t,h=h)
    ip1 = im_patches_n1
    ip1 = rearrange(ip1,'t (h w) k d -> t h w k d',h=h)

    return ip1,top_dist1,top_ind1

