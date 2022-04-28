
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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    The NonLocalDenoiser in Parts
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_parts(self,noisy,sigma,srch_img=None,flows=None,train=False,rescale=True):

    #
    # -- Prepare --
    #

    # -- normalize for input ---
    if rescale: noisy = (noisy/255. - 0.5)/0.5
    means = noisy.mean((-2,-1),True)
    noisy -= means
    if srch_img is None:
        srch_img = noisy

    #
    # -- Non-Local Search --
    #

    # -- [nn0 search]  --
    output0 = self.run_nn0(noisy.clone(),srch_img.clone(),flows,train)
    patches0 = output0[0]
    dists0 = output0[1]
    inds0 = output0[2]

    # -- [nn1 search]  --
    output1 = self.run_nn1(noisy.clone(),srch_img.clone(),flows,train)
    patches1 = output1[0]
    dists1 = output1[1]
    inds1 = output1[2]

    #
    # -- Separable ConvNet --
    #

    # -- reshape --
    patches0 = rearrange(patches0,'t h w k d -> t (h w) k d')
    dists0 = rearrange(dists0,'t h w k -> t (h w) k')
    inds0 = rearrange(inds0,'t h w k tr -> t (h w) k tr')
    patches1 = rearrange(patches1,'t h w k d -> t (h w) k d')
    dists1 = rearrange(dists1,'t h w k -> t (h w) k')
    inds1 = rearrange(inds1,'t h w k tr -> t (h w) k tr')

    # -- exec --
    image_dn,patches_w = self.run_pdn(patches0,dists0,inds0,
                                      patches1,dists1,inds1)
    #
    # -- Final Weight Aggregation --
    #

    h,w = 64,64
    image_dn = self.run_parts_final(image_dn,patches_w,inds0,h,w)

    # -- normalize for output ---
    image_dn += means
    if rescale: image_dn = 255*(image_dn * 0.5 + 0.5)

    return image_dn


@register_method
def run_parts_final(self,image_dn,patch_weights,inds,h,w):

    # -- prepare --
    nump = 68 # ??
    c = 3
    ps = self.patch_w
    pdim = image_dn.shape[-1]
    image_dn = image_dn * patch_weights
    ones_tmp = th.ones(1, 1, pdim, device=image_dn.device)
    patch_weights = (patch_weights * ones_tmp).transpose(2, 1)
    image_dn = image_dn.transpose(2, 1)

    # -- prepare gather --
    t,hw,k,tr = inds.shape
    inds = rearrange(inds[...,0,:],'t p tr -> (t p) 1 tr').clone()
    zeros = th.zeros_like(inds[...,0])
    hp = int(np.sqrt(hw))
    wp = hp
    # image_dn = rearrange(image_dn,'t (c h w) p -> (t p) 1 1 c h w',h=ps,w=ps)
    # wpatch = rearrange(patch_weights,'t (c h w) p -> (t p) 1 1 c h w',h=ps,w=ps)

    # -- inds --
    inds[:,:,1] += (ps//2)
    inds[:,:,2] += (ps//2)

    # -- fold --
    h,w = hp+2*(ps//2),wp+2*(ps//2)
    shape = (t,c,h,w)
    image_dn = fold(image_dn,(h,w),(ps,ps))
    patch_cnt = fold(patch_weights,(h,w),(ps,ps))
    # image_dn,_ = dnls.simple.gather.run(image_dn, zeros, inds, shape=shape)
    # patch_cnt,_ = dnls.simple.gather.run(wpatch, zeros, inds, shape=shape)

    # -- crop --
    row_offs = min(ps - 1, nump - 1)
    col_offs = min(ps - 1, nump - 1)
    image_dn = crop_offset(image_dn, (row_offs,), (col_offs,))
    image_dn /= crop_offset(patch_cnt, (row_offs,), (col_offs,))

    return image_dn

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Patch-based Denoising Network
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def print_stats(name,tensor):
    print(name,tensor.min().item(),
          tensor.max().item(),tensor.mean().item())

@register_method
def run_pdn(self,patches_n0,dist0,inds0,patches_n1,dist1,inds1):
    """
    Run patch denoiser network
    """
    # -- run sep-net --
    h,w = 64,64
    agg0,s0,_ = self.run_agg0(patches_n0,dist0,inds0,h,w)
    h,w = 64,64
    agg1,s1,_,_ = self.run_agg1(patches_n1,dist1,inds1,h,w)
    assert th.any(th.isnan(agg1)).item() is False

    # -- final output --
    inputs = th.cat((s0, s1, agg0, agg1), dim=-2)
    sep_net = self.patch_denoise_net.separable_fc_net
    noise = sep_net.sep_part2(inputs)

    # -- compute denoised patches --
    image_dn,patches_w = self.run_pdn_final(patches_n0,noise)

    return image_dn,patches_w

@register_method
def run_pdn_final(self,patches_n0,noise):
    pdn = self.patch_denoise_net
    patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
    patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
    patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
    patch_weights = th.exp(-pdn.beta.abs() * patch_exp_weights)
    return patches_dn,patch_weights

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Aggregation Steps
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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
    assert th.any(th.isnan(y_out1)).item() is False
    assert th.any(th.isnan(y_out1/weights1)).item() is False
    y_out1,fold_out1 = sep_net.agg1(y_out1 / weights1, dist1, inds1,h,w,both=True)
    assert th.any(th.isnan(y_out1)).item() is False
    y_out1 = weights1 * y_out1
    assert th.any(th.isnan(y_out1)).item() is False
    y_out1 = sep_net.ver_hor_bn_re_agg1_post(y_out1)
    assert th.any(th.isnan(y_out1)).item() is False

    return y_out1,x1,fold_out1,wpatches


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Run Nearest Neighbors Search
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_nn0(self,image_n,srch_img=None,flows=None,train=False):
    return self.run_nn0_dnls_search(image_n,srch_img,flows,train)
    # return self.run_nn0_lidia_search(image_n,train)

@register_method
def run_nn0_dnls_search(self,image_n,srch_img=None,flows=None,train=False):

    #
    # -- Our Search --
    #

    # -- pad & unpack --
    ps = self.patch_w
    neigh_pad = 14
    device = image_n.device
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]

    # -- prepeare image --
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)

    # -- params --
    params = get_image_params(image_n0, self.patch_w, neigh_pad)

    # -- get search image --
    if not(srch_img is None):
        img_nn0 = self.pad_crop0(image_n, self.pad_offs, train)
        img_nn0 = self.rgb2gray(img_nn0)
    elif self.arch_opt.rgb:
        img_nn0 = self.rgb2gray(image_n0)
    else:
        img_nn0 = image_n0
    print("[dnls] img_nn0.shape: ",img_nn0.shape)
    print(params)

    # -- get search inds --
    pad = ps//2
    t,c,h,w = image_n.shape
    # hp,wp = h+2*pad,w+2*pad
    hp,wp = params['patches_h'],params['patches_w']
    print("hp,wp: ",hp,wp)
    queryInds = th.arange(t*hp*wp,device=device).reshape(-1,1,1,1)
    queryInds = get_3d_inds(queryInds,hp,wp)[:,0]
    t,c,h0,w0 = image_n0.shape
    sh,sw = (h0 - hp)//2,(w0 - wp)//2
    queryInds[...,1] += sh
    queryInds[...,2] += sw

    # -- search --
    ps = self.patch_w
    k,pt,ws,wt,chnls = 14,1,29,0,1
    nlDists,nlInds = dnls.simple.search.run(img_nn0,queryInds,flows,
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
    pad_s = 2*(ps//2)
    patches = rearrange(patches,'t (h w) k d -> t h w k d',h=hp)
    top_ind0 = rearrange(top_ind0,'(t h w) k tr -> t h w k tr',t=t,h=hp)
    top_dist0 = rearrange(top_dist0,'(t h w) k -> t h w k',t=t,h=hp)

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
    k = 14
    neigh_pad = 14
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    device = image_n.device
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)
    ps = self.patch_w
    t,c,h,w = image_n.shape

    # -- get search image --
    if self.arch_opt.rgb: img_nn0 = self.rgb2gray(image_n0)
    else: img_nn0 = image_n0

    # -- get image-based parameters --
    params = get_image_params(image_n0, self.patch_w, neigh_pad)
    params['pad_patches_w_full'] = params['pad_patches_w']
    print("[lidia] img_nn0.shape: ",img_nn0.shape)
    print(params)

    # -- run knn search --
    top_dist0, top_ind0 = self.find_nn(img_nn0, params, self.patch_w)

    # -- prepare [dists,inds] --
    ip = params['pad_patches']
    patch_dist0 = top_dist0.view(top_dist0.shape[0], -1, k)[:, :, 1:]
    top_ind0 += ip * th.arange(top_ind0.shape[0],device=device).view(-1, 1, 1, 1)

    #
    # -- Our Section --
    #

    # -- get new inds
    ch,cw = params['pad_patches_h'],params['pad_patches_w']
    dnls_inds = get_3d_inds(top_ind0,ch,cw)

    # -- rescale offset --
    inds_pad = (ps-1)//2
    dnls_inds[...,1] += inds_pad
    dnls_inds[...,2] += inds_pad

    # -- indexing patches --
    im_patches_n0 = dnls.simple.scatter.run(image_n0,dnls_inds,self.patch_w)
    ishape = '(t p) k 1 c h w -> t p k (c h w)'
    im_patches_n0 = rearrange(im_patches_n0,ishape,t=t)

    # -- append anchor patch spatial variance --
    patch_var0 = im_patches_n0[:, :, 0, :].std(dim=-1).\
        unsqueeze(-1).pow(2) * patch_numel
    patch_dist0 = th.cat((patch_dist0, patch_var0), dim=-1)

    # -- rescale inds --
    dnls_inds[...,1] -= neigh_pad+inds_pad
    dnls_inds[...,2] -= neigh_pad+inds_pad

    # -- format [dists,inds] --
    t,h,w,k = top_dist0.shape
    dnls_inds = rearrange(dnls_inds,'(t h w) k tr -> t h w k tr',t=t,h=h)
    ip0 = im_patches_n0
    ip0 = rearrange(ip0,'t (h w) k d -> t h w k d',h=h)
    patch_dist0 = rearrange(patch_dist0,'t (h w) k -> t h w k',h=h)

    return ip0,patch_dist0,dnls_inds

@register_method
def run_nn1(self,image_n,srch_img=None,flows=None,train=False):
    return self.run_nn1_dnls_search(image_n,srch_img,flows,train)
    # return self.run_nn1_lidia_search(image_n,train)


@register_method
def prepare_image_n1(self,image_n,train):

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
    return image_n1

@register_method
def run_nn1_dnls_search(self,image_n,srch_img=None,flows=None,train=False):

    # -- unpack --
    t = image_n.shape[0]
    neigh_pad = 14
    ps = self.patch_w
    device = image_n.device
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]

    # -- pad & unpack --
    image_n1 = self.prepare_image_n1(image_n,train)
    print("image_n1.shape: ",image_n1.shape)
    params = get_image_params(image_n1, 2*self.patch_w-1, 2*neigh_pad)

    #
    #  -- DNLS Search --
    #

    # -- get search image --
    if not(srch_img is None):
        img_nn1 = self.prepare_image_n1(srch_img,train)
        img_nn1 = self.rgb2gray(img_nn1)
    elif self.arch_opt.rgb:
        img_nn1 = self.rgb2gray(image_n1)
    else:
        img_nn1 = image_n1

    # -- get search inds --
    hp,wp = params['patches_h'],params['patches_w']
    queryInds = th.arange(t*hp*wp,device=device).reshape(-1,1,1,1)
    queryInds = get_3d_inds(queryInds,hp,wp)[:,0]
    t,c,h0,w0 = image_n1.shape

    # -- inds offsets --
    sh,sw = (h0 - hp)//2,(w0 - wp)//2
    print(params)
    print(sh,sw)
    queryInds[...,1] += sh
    queryInds[...,2] += sw

    # -- exec search --
    k,pt,ws,wt,chnls = 14,1,29,0,1
    nlDists,nlInds = dnls.simple.search.run(img_nn1,queryInds,flows,
                                            k,ps,pt,ws,wt,chnls,
                                            stride=2,dilation=2)

    #
    # -- Scatter Section --
    #

    # -- dnls --
    patches = dnls.simple.scatter.run(image_n1,nlInds,ps,dilation=2)

    #
    # -- Final Formatting --
    #

    # - reshape --
    hp,wp = params['patches_h'],params['patches_w']
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
    t,c,h1,w1 = image_n1.shape
    sh,sw = (h1 - hp)//2,(w1 - wp)//2
    nlInds[...,1] -= sh
    nlInds[...,2] -= sw
    print("patches.shape: ",patches.shape)

    return patches,nlDists,nlInds

@register_method
def run_nn1_lidia_search(self,image_n,train=False):

    # -- unpack --
    neigh_pad = 14
    ps = self.patch_w
    k = 14
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    device = image_n.device

    # -- pad & unpack --
    image_n1 = self.prepare_image_n1(image_n,train)
    print("image_n1.shape: ",image_n1.shape)

    #
    #  -- LIDIA Search --
    #

    # -- img-based parameters --
    params = get_image_params(image_n1, 2 * self.patch_w - 1, 2*neigh_pad)
    params['pad_patches_w_full'] = params['pad_patches_w']

    # -- get search image  --
    if self.arch_opt.rgb: img_nn1 = self.rgb2gray(image_n1)
    else: img_nn1 = image_n1

    # -- spatially split image --
    img_nn1_00 = img_nn1[:, :, 0::2, 0::2].clone()
    img_nn1_10 = img_nn1[:, :, 1::2, 0::2].clone()
    img_nn1_01 = img_nn1[:, :, 0::2, 1::2].clone()
    img_nn1_11 = img_nn1[:, :, 1::2, 1::2].clone()

    # -- get split image parameters --
    params_00 = get_image_params(img_nn1_00, self.patch_w, neigh_pad)
    params_10 = get_image_params(img_nn1_10, self.patch_w, neigh_pad)
    params_01 = get_image_params(img_nn1_01, self.patch_w, neigh_pad)
    params_11 = get_image_params(img_nn1_11, self.patch_w, neigh_pad)
    params_00['pad_patches_w_full'] = params['pad_patches_w']
    params_10['pad_patches_w_full'] = params['pad_patches_w']
    params_01['pad_patches_w_full'] = params['pad_patches_w']
    params_11['pad_patches_w_full'] = params['pad_patches_w']

    # -- run knn search! --
    top_dist1_00, top_ind1_00 = self.find_nn(img_nn1_00, params_00,
                                             self.patch_w, scale=1, case='00')
    top_dist1_10, top_ind1_10 = self.find_nn(img_nn1_10, params_10,
                                             self.patch_w, scale=1, case='10')
    top_dist1_01, top_ind1_01 = self.find_nn(img_nn1_01, params_01,
                                             self.patch_w, scale=1, case='01')
    top_dist1_11, top_ind1_11 = self.find_nn(img_nn1_11, params_11,
                                             self.patch_w, scale=1, case='11')

    # -- aggregate results [dists] --
    top_dist1 = th.zeros(params['batches'], params['patches_h'],
                            params['patches_w'], k, device=device)
    top_dist1 = top_dist1.fill_(float('nan'))
    top_dist1[:, 0::2, 0::2, :] = top_dist1_00
    top_dist1[:, 1::2, 0::2, :] = top_dist1_10
    top_dist1[:, 0::2, 1::2, :] = top_dist1_01
    top_dist1[:, 1::2, 1::2, :] = top_dist1_11

    # -- aggregate results [inds] --
    ipp = params['pad_patches']
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
    hp,wp = params['pad_patches_h'],params['pad_patches_w']
    inds = get_3d_inds(top_ind1,hp,wp)

    # -- prepare inds for scatter
    _,_,h1,w1 = image_n1.shape
    pad = (h1-hp)//2
    inds[...,1] += pad
    inds[...,2] += pad

    # -- dnls --
    pad = ps//2
    hp,wp = params['patches_h'],params['patches_w']
    im_patches_n1 = dnls.simple.scatter.run(image_n1,inds,ps,dilation=2)
    ishape = '(t ih iw) k 1 c h w -> t (ih iw) k (c h w)'
    im_patches_n1 = rearrange(im_patches_n1,ishape,ih=hp,iw=wp)

    #
    # -- Final Formatting --
    #

    # -- re-shaping --
    t,h,w,k = top_dist1.shape
    inds = rearrange(inds,'(t h w) k tr -> t h w k tr',t=t,h=h)
    ip1 = im_patches_n1
    ip1 = rearrange(ip1,'t (h w) k d -> t h w k d',h=h)

    # -- patch variance --
    d = ip1.shape[-1]
    patch_var = ip1[...,0,:].std(-1)**2*d
    top_dist1[...,:-1] = top_dist1[...,1:]
    top_dist1[...,-1] = patch_var

    # -- zero-out inds --
    inds[...,1] -= (2*neigh_pad + 2*pad)
    inds[...,2] -= (2*neigh_pad + 2*pad)

    return ip1,top_dist1,inds

