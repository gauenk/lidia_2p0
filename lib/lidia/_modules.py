
from torch.nn.functional import unfold
from torch.nn.functional import fold
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn.functional import conv2d
from .utils.lidia_utils import *
from .utils.inds import get_3d_inds


import torch as th
from pathlib import Path
from einops import repeat

# -- lidia utils --
# from .utils.lidia_utils import get_image_params

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
def run_parts(self,noisy,sigma,train=False):

    #
    # -- Prepare --
    #

    # -- normalize for input ---
    noisy = (noisy/255. - 0.5)/0.5
    means = noisy.mean((-2,-1),True)
    noisy -= means

    #
    # -- Non-Local Search --
    #

    # -- [nn0 search]  --
    output0 = self.run_nn0(noisy,train=train)
    patches0 = output0[0]
    dists0 = output0[1]
    inds0 = output0[2]
    params0 = output0[3]

    # -- [nn1 search]  --
    output1 = self.run_nn1(noisy,train=train)
    patches1 = output1[0]
    dists1 = output1[1]
    inds1 = output1[2]
    params1 = output1[3]

    #
    # -- Separable ConvNet --
    #

    # -- reshape --
    shape = patches0.shape
    patches0 = rearrange(patches0,'t h w k d -> t (h w) k d')
    dists0 = rearrange(dists0,'t h w k -> t (h w) k')
    inds0 = rearrange(inds0,'t h w k tr -> t (h w) k tr')
    patches1 = rearrange(patches1,'t h w k d -> t (h w) k d')
    dists1 = rearrange(dists1,'t h w k -> t (h w) k')
    inds1 = rearrange(inds1,'t h w k tr -> t (h w) k tr')

    # -- exec --
    image_dn,patches_w = self.run_pdn(patches0,dists0,inds0,params0,
                                      patches1,dists1,inds1,params1)

    #
    # -- Final Weight Aggregation --
    #

    h,w = 64,64
    image_dn = self.run_parts_final(image_dn,patches_w,params0)

    # -- normalize for output ---
    image_dn += means
    image_dn = 255*(image_dn * 0.5 + 0.5)

    return image_dn


@register_method
def run_parts_final(self,image_dn,patch_weights,params):

    # -- prepare --
    ps = self.patch_w
    pdim = image_dn.shape[-1]
    image_dn = image_dn * patch_weights
    ones_tmp = th.ones(1, 1, pdim, device=image_dn.device)
    patch_weights = (patch_weights * ones_tmp).transpose(2, 1)
    image_dn = image_dn.transpose(2, 1)

    # -- fold --
    h,w = params['pixels_h'],params['pixels_w']
    image_dn = fold(image_dn, (h,w),(ps,ps))
    patch_cnt = fold(patch_weights, (h,w),(ps,ps))

    # -- crop --
    row_offs = min(ps - 1, params['patches_h'] - 1)
    col_offs = min(ps - 1, params['patches_w'] - 1)
    image_dn = crop_offset(image_dn, (row_offs,), (col_offs,))
    image_dn /= crop_offset(patch_cnt, (row_offs,), (col_offs,))

    return image_dn

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Patch-based Denoising Network
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_pdn(self,patches_n0,dist0,inds0,params0,patches_n1,dist1,inds1,params1):
    """
    Run patch denoiser network
    """
    # -- run sep-net --
    h,w = params0['pixels_h'],params0['pixels_w']
    agg0,s0,_ = self.run_agg0(patches_n0,dist0,h,w)
    h,w = params1['pixels_h'],params1['pixels_w']
    agg1,s1,_,_ = self.run_agg1(patches_n1,dist1,h,w)

    # -- final output --
    inputs = th.cat((s0, s1, agg0, agg1), dim=-2)
    sep_net = self.patch_denoise_net.separable_fc_net
    print(inputs.min(),inputs.mean(),inputs.max())
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
def run_agg0(self,patches,dist0,h,w):

    # -- compute weights --
    pdn = self.patch_denoise_net
    weights0 = pdn.weights_net0(th.exp(-pdn.alpha0.abs() * dist0)).unsqueeze(-1)
    weighted_patches = patches * weights0

    # -- compute sep layers --
    sep_net = self.patch_denoise_net.separable_fc_net
    x0 = sep_net.sep_part1_s0(weighted_patches)
    y_out0 = sep_net.ver_hor_agg0_pre(x0)
    y_out0,fold_out0 = sep_net.agg0(y_out0, h, w,both=True)
    y_out0 = sep_net.ver_hor_bn_re_agg0_post(y_out0)

    return y_out0,x0,fold_out0

@register_method
def run_agg1(self,patches,dist1,h,w):

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
    y_out1,fold_out1 = sep_net.agg1(y_out1 / weights1,h,w,both=True)
    y_out1 = weights1 * y_out1
    y_out1 = sep_net.ver_hor_bn_re_agg1_post(y_out1)

    return y_out1,x1,fold_out1,wpatches

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Run Nearest Neighbors Search
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_nn0(self,image_n,train=False):

    # -- pad & unpack --
    neigh_pad = 14
    t_n,c_n,h_n,w_n = image_n.shape
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    device = image_n.device
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)
    print("image_n0.shape: ",image_n0.shape)

    # -- get search image --
    if self.arch_opt.rgb: img_nn0 = self.rgb2gray(image_n0)
    else: img_nn0 = image_n0

    # -- get image-based parameters --
    params = get_image_params(image_n0, self.patch_w, neigh_pad)
    params['pad_patches_w_full'] = params['pad_patches_w']
    print("image_n.shape: ",image_n.shape)
    print(params)

    # -- run knn search --
    print("img_nn0.shape: ",img_nn0.shape)
    top_dist0, top_ind0 = self.find_nn(img_nn0, params, self.patch_w)

    # -- prepare [dists,inds] --
    ip = params['pad_patches']
    patch_dist0 = top_dist0.view(top_dist0.shape[0], -1, neigh_pad)[:, :, 1:]
    top_ind0 += ip * th.arange(top_ind0.shape[0],device=device).view(-1, 1, 1, 1)

    # -- get all patches -
    patches = unfold(image_n0, (self.patch_w, self.patch_w)).\
        transpose(1, 0).contiguous().view(patch_numel, -1).t()

    # -- organize patches by knn --
    patches = patches[top_ind0.view(-1), :].\
        view(top_ind0.shape[0], -1, neigh_pad, patch_numel)

    # -- append anchor patch spatial variance --
    patch_var0 = patches[:, :, 0, :].std(dim=-1).\
        unsqueeze(-1).pow(2) * patch_numel
    patch_dist0 = th.cat((patch_dist0, patch_var0), dim=-1)

    # -- convert to 3d inds --
    t,c,h,w = image_n0.shape
    ps = self.patch_w
    ch,cw = h-(ps-1),w-(ps-1)
    k = top_ind0.shape[-1]
    inds3d = get_3d_inds(top_ind0.view(-1,1,1,k),ch,cw)

    # -- rescale inds --
    inds3d[...,1] -= neigh_pad
    inds3d[...,2] -= neigh_pad

    # -- format [dists,inds] --
    h,w = params['patches_h'],params['patches_w']
    patches = rearrange(patches,'t (h w) k d -> t h w k d',h=h)
    patch_dist0 = rearrange(patch_dist0,'t (h w) k -> t h w k',h=h)
    inds3d = rearrange(inds3d,'(t h w) k tr -> t h w k tr',t=t,h=h)

    return patches,patch_dist0,inds3d,params

@register_method
def run_nn1(self,image_n,train=False):

    # -- misc unpack --
    neigh_pad = 14
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

    # -- img-based parameters --
    params = get_image_params(image_n1, 2 * self.patch_w - 1, 28)
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
                            params['patches_w'], neigh_pad, device=device)
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

    # -- get all patches --
    im_patches_n1 = unfold(image_n1, (self.patch_w, self.patch_w),
                           dilation=(2, 2)).transpose(1, 0).contiguous().\
                           view(patch_numel, -1).t()
    print("[modules] im_patches_n1.shape: ",im_patches_n1.shape)

    # -- organize by knn --
    np = top_ind1.shape[0]
    pn = patch_numel
    im_patches_n1 = im_patches_n1[top_ind1.view(-1), :].view(np, -1, neigh_pad, pn)
    print("[modules] im_patches_n1.shape: ",im_patches_n1.shape)

    # -- append anchor patch spatial variance --
    patch_dist1 = top_dist1.view(top_dist1.shape[0], -1, neigh_pad)[:, :, 1:]
    patch_var1 = im_patches_n1[:, :, 0, :].std(dim=-1).unsqueeze(-1).pow(2) * pn
    patch_dist1 = th.cat((patch_dist1, patch_var1), dim=-1)

    #
    # -- Final Formatting --
    #

    # -- inds -> 3d_inds --
    pad = 2*(ps//2) # dilation "= 2"
    _t,_c,_h,_w = image_n1.shape
    hp,wp = _h-2*pad,_w-2*pad
    top_ind1 = get_3d_inds(top_ind1,hp,wp)

    # -- rescale inds --
    top_ind1[...,1] -= 28
    top_ind1[...,2] -= 28

    # -- re-shaping --
    t,h,w,k = top_dist1.shape
    pdist = rearrange(patch_dist1,'t (h w) k -> t h w k',h=h)
    top_ind1 = rearrange(top_ind1,'(t h w) k tr -> t h w k tr',t=t,h=h)
    ip1 = im_patches_n1
    ip1 = rearrange(ip1,'t (h w) k d -> t h w k d',h=h)
    print("[ip1] ip1.shape: ",ip1.shape)

    return ip1,pdist,top_ind1,params



