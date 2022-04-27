from torch.nn.functional import unfold
from torch.nn.functional import fold
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn.functional import conv2d
from .utils.lidia_utils import *


import torch as th
from pathlib import Path
from einops import repeat

# -- import module logic --
from .utils import clean_code
from . import _modules
from .utils.io import save_burst,save_image
from .utils.logging import print_extrema


class ArchitectureOptions:
    def __init__(self, rgb, small_network):
        assert not (rgb and small_network), "LIDIA-S doesn't support color images"
        self.rgb = rgb
        self.small_network = small_network


class VerHorMat(nn.Module):
    def __init__(self, ver_in, ver_out, hor_in, hor_out):
        super(VerHorMat, self).__init__()
        self.ver = nn.Linear(in_features=ver_in, out_features=ver_out, bias=False)
        self.hor = nn.Linear(in_features=hor_in, out_features=hor_out, bias=False)
        self.b = nn.Parameter(th.empty(hor_out, ver_out, dtype=th.float32))
        nn.init.xavier_normal_(self.b, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.ver(x)
        x = self.hor(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + self.b

        return x

    def extra_repr(self):
        return 'b.shape=' + str(tuple(self.b.shape))

    def get_ver_out(self):
        return self.ver.out_features

    def get_hor_out(self):
        return self.hor.out_features


class Aggregation0(nn.Module):
    def __init__(self, patch_w):
        super(Aggregation0, self).__init__()
        self.patch_w = patch_w

    def forward2fold(self, x, pixels_h, pixels_w):
        images, patches, hor_f, ver_f = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(images * hor_f, ver_f, patches)
        # print("[agg0:2] x.shape: ",x.shape)
        patch_cnt = th.ones(x[0:1, ...].shape, device=x.device)
        patch_cnt = fold(patch_cnt, (pixels_h, pixels_w), (self.patch_w, self.patch_w))
        # print("[agg0] patch_cnt.shape: ",patch_cnt.shape)
        x = fold(x, (pixels_h, pixels_w), (self.patch_w, self.patch_w)) / patch_cnt
        return patch_cnt

    def forward(self, x, pixels_h, pixels_w,both=False):
        # print("[agg0] x.shape: ",x.shape)
        images, patches, hor_f, ver_f = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(images * hor_f, ver_f, patches)
        # print("[agg0:2] x.shape: ",x.shape)
        patch_cnt = th.ones(x[0:1, ...].shape, device=x.device)
        patch_cnt = fold(patch_cnt, (pixels_h, pixels_w), (self.patch_w, self.patch_w))
        # print("[agg0] patch_cnt.shape: ",patch_cnt.shape)
        # print_extrema("prefold-agg0.x",x)
        x = fold(x, (pixels_h, pixels_w), (self.patch_w, self.patch_w)) / patch_cnt
        xf = x
        # print("[module] gather shape: ",x.shape)
        # print_extrema("postfold-agg0.x",x)
        # print("[agg0:fold]: x.shape ",x.shape)
        x = unfold(x, (self.patch_w, self.patch_w))
        # print_extrema("unfold-agg0.x",x)
        # print("[agg0:unfold]: x.shape ",x.shape)
        x = x.view(images, hor_f, ver_f, patches).permute(0, 3, 1, 2)
        # print("[agg0:view]: x.shape ",x.shape)

        if both:
            return x,xf
        else:
            return x

class Aggregation1(nn.Module):
    def __init__(self, patch_w):
        super(Aggregation1, self).__init__()
        self.patch_w = patch_w

        kernel_1d = th.tensor((1 / 4, 1 / 2, 1 / 4), dtype=th.float32)
        kernel_2d = (kernel_1d.view(-1, 1) * kernel_1d).view(1, 1, 3, 3)
        self.bilinear_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False)
        self.bilinear_conv.weight.data = kernel_2d
        self.bilinear_conv.weight.requires_grad = False

    def forward(self, x, pixels_h, pixels_w, both=False):
        # tag-agg1
        # print("[agg1:1] x.shape: ",x.shape)
        # print("[agg1.input] x.shape: ",x.shape)
        images, patches, hor_f, ver_f = x.shape
        x = x.permute(0, 2, 3, 1).view(images * hor_f, ver_f, patches)
        # print("[agg1:2] x.shape: ",x.shape)


        # isize = (68,68) # (pixels_h, pixels_w)
        # ksize = (self.patch_w, self.patch_w)
        # dil = (1,1)
        # pads = (2,2)
        # patch_cnt = th.ones(x[0:1, ...].shape, device=x.device)
        # patch_cnt = fold(patch_cnt, isize, ksize, dilation=dil,padding=pads)
        # print_extrema("patch_cnt",patch_cnt)
        # x_v1 = fold(x, isize, ksize, dilation=dil,padding=pads) / patch_cnt
        # print("[folded] x.shape: ",x_v1.shape)
        # print_extrema("post-agg1.x",x_v1)
        # # # print_extrema("post-agg1.wx",wx)
        # x_v1 -= x_v1.min()
        # x_v1 /= x_v1.max()
        # save_burst(x_v1,"agg1_x_v1")


        isize = (pixels_h, pixels_w)
        ksize = (self.patch_w, self.patch_w)
        dil = (2,2)
        pads = (0,0)
        patch_cnt = th.ones(x[0:1, ...].shape, device=x.device)
        patch_cnt = fold(patch_cnt, isize, ksize, dilation=dil,padding=pads)
        x_v2 = fold(x, isize, ksize, dilation=dil,padding=pads) / patch_cnt
        x = x_v2

        # -- view --
        x_b, x_c, x_h, x_w = x.shape
        x = self.bilinear_conv(nn_func.pad(x, [1] * 4, 'reflect').view(x_b * x_c, 1, x_h + 2, x_w + 2)) \
            .view(x_b, x_c, x_h, x_w)
        x = unfold(x, (self.patch_w, self.patch_w), dilation=(2, 2))
        x = x.view(images, hor_f, ver_f, patches).permute(0, 3, 1, 2)

        if both:
            return x,x_v2
        else:
            return x

class VerHorBnRe(nn.Module):
    def __init__(self, ver_in, ver_out, hor_in, hor_out, bn):
        super(VerHorBnRe, self).__init__()
        self.ver_hor = VerHorMat(ver_in=ver_in, ver_out=ver_out, hor_in=hor_in, hor_out=hor_out)
        if bn:
            self.bn = nn.BatchNorm2d(hor_out)
        self.thr = nn.ReLU()

    def forward(self, x):
        x = self.ver_hor(x)
        if hasattr(self, 'bn'):
            x = self.bn(x.transpose(-2, -3)).transpose(-2, -3)
        x = self.thr(x)

        return x


class SeparablePart1(nn.Module):
    def __init__(self, arch_opt, hor_size, patch_numel, ver_size):
        super(SeparablePart1, self).__init__()

        self.ver_hor_bn_re0 = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size, hor_in=14, hor_out=hor_size, bn=False)
        if not arch_opt.small_network:
            self.a = nn.Parameter(th.tensor((0,), dtype=th.float32))
            self.ver_hor_bn_re1 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size, hor_in=14, hor_out=hor_size, bn=True)

    def forward(self, x):
        x = self.ver_hor_bn_re0(x)
        if hasattr(self, 'ver_hor_bn_re1'):
            x = self.a * x + (1 - self.a) * self.ver_hor_bn_re1(x)

        return x


class SeparablePart2(nn.Module):
    def __init__(self, arch_opt, hor_size_in, patch_numel, ver_size):
        super(SeparablePart2, self).__init__()

        self.ver_hor_bn_re2 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size, hor_in=hor_size_in, hor_out=56, bn=True)
        if not arch_opt.small_network:
            self.a = nn.Parameter(th.tensor((0,), dtype=th.float32))
            self.ver_hor_bn_re3 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size, hor_in=56, hor_out=56, bn=True)
        self.ver_hor_out = VerHorMat(ver_in=ver_size, ver_out=patch_numel, hor_in=56, hor_out=1)

    def forward(self, x):
        x = self.ver_hor_bn_re2(x)
        if hasattr(self, 'ver_hor_bn_re3'):
            x = self.a * x + (1 - self.a) * self.ver_hor_bn_re3(x)
        x = self.ver_hor_out(x)

        return x


class SeparableFcNet(nn.Module):
    def __init__(self, arch_opt, patch_w, ver_size):
        super(SeparableFcNet, self).__init__()
        patch_numel = (patch_w ** 2) * 3 if arch_opt.rgb else patch_w ** 2

        self.sep_part1_s0 = SeparablePart1(arch_opt=arch_opt, hor_size=14, patch_numel=patch_numel, ver_size=ver_size)
        self.sep_part1_s1 = SeparablePart1(arch_opt=arch_opt, hor_size=14, patch_numel=patch_numel, ver_size=ver_size)
        self.ver_hor_agg0_pre = VerHorMat(ver_in=ver_size, ver_out=patch_numel, hor_in=14, hor_out=1)
        self.agg0 = Aggregation0(patch_w)
        self.ver_hor_bn_re_agg0_post = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size, hor_in=1, hor_out=14, bn=False)
        self.ver_hor_agg1_pre = VerHorMat(ver_in=ver_size, ver_out=patch_numel, hor_in=14, hor_out=1)
        self.agg1 = Aggregation1(patch_w)
        self.ver_hor_bn_re_agg1_post = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size, hor_in=1, hor_out=14, bn=False)

        self.sep_part2 = SeparablePart2(arch_opt=arch_opt, hor_size_in=56, patch_numel=patch_numel, ver_size=ver_size)


    def forward(self, x0, x1, weights1, im_params0, im_params1, save_memory, max_chunk):
        # print("x0.shape: ",x0.shape)
        # print("x1.shape: ",x1.shape)
        # print("weights1.shape: ",weights1.shape)
        if save_memory:
            out = th.zeros(x0[:, :, 0:1, :].shape, device=x0.device).fill_(float('nan'))
            out_s0_shape = th.Size((x0.shape[0:2]) +
                                      (self.sep_part1_s0.ver_hor_bn_re0.ver_hor.get_hor_out(),) +
                                      (self.sep_part1_s0.ver_hor_bn_re0.ver_hor.get_ver_out(),))
            out_part1_s0 = th.zeros(out_s0_shape, device=x0.device).fill_(float('nan'))
            apply_on_chuncks(self.sep_part1_s0, x0, max_chunk, out_part1_s0)

            # sep_part1_s1
            out_s1_shape = th.Size((x1.shape[0:2]) +
                                      (self.sep_part1_s1.ver_hor_bn_re0.ver_hor.get_hor_out(),) +
                                      (self.sep_part1_s1.ver_hor_bn_re0.ver_hor.get_ver_out(),))
            out_part1_s1 = th.zeros(out_s1_shape, device=x1.device).fill_(float('nan'))
            apply_on_chuncks(self.sep_part1_s1, x1, max_chunk, out_part1_s1)

            # agg0
            y0_shape = th.Size((out_part1_s0[:, :, 0:1, :].shape[0:3]) +
                                  (self.ver_hor_agg0_pre.get_ver_out(),))
            y_tmp0 = th.zeros(y0_shape, device=out_part1_s0.device).fill_(float('nan'))
            # print("y_tmp0.shape: ",y_tmp0.shape)
            apply_on_chuncks(self.ver_hor_agg0_pre, out_part1_s0, max_chunk, y_tmp0)
            # print("y_tmp0.shape: ",y_tmp0.shape)
            y_tmp0 = self.agg0(y_tmp0, im_params0['pixels_h'], im_params0['pixels_w'])
            # print("y_tmp0.shape: ",y_tmp0.shape)

            # agg1
            y1_shape = th.Size((out_part1_s1[:, :, 0:1, :].shape[0:3]) +
                                  (self.ver_hor_agg1_pre.get_ver_out(),))
            y_tmp1 = th.zeros(y1_shape, device=out_part1_s1.device).fill_(float('nan'))
            apply_on_chuncks(self.ver_hor_agg1_pre, out_part1_s1, max_chunk, y_tmp1)
            y_tmp1 = weights1 * self.agg1(y_tmp1 / weights1, im_params1['pixels_h'], im_params1['pixels_w'])

            # sep_part2
            for si in range(0, out_part1_s0.shape[1], max_chunk):
                ei = min(si + max_chunk, out_part1_s0.shape[1])
                y_out0 = self.ver_hor_bn_re_agg0_post(y_tmp0[:, si:ei, :, :])
                y_out1 = self.ver_hor_bn_re_agg1_post(y_tmp1[:, si:ei, :, :])
                out[:, si:ei, :, :] = self.sep_part2(th.cat((out_part1_s0[:, si:ei, :, :],
                                                                out_part1_s1[:, si:ei, :, :],
                                                                y_out0, y_out1), dim=-2))

        else:
            # -- sep comps --
            print_extrema("[pre-sep] x0",x0)
            print_extrema("[pre-sep] x1",x1)
            print("x0.shape: ",x0.shape)
            print("x1.shape: ",x1.shape)
            x0 = self.sep_part1_s0(x0)
            x1 = self.sep_part1_s1(x1)

            # -- agg0 --
            y_out0 = self.ver_hor_agg0_pre(x0)
            y_out0 = self.agg0(y_out0, im_params0['pixels_h'], im_params0['pixels_w'])
            y_out0 = self.ver_hor_bn_re_agg0_post(y_out0)

            # -- agg1 --
            y_out1 = self.ver_hor_agg1_pre(x1)
            print("y_out1.shape: ",y_out1.shape)
            y_out1 = weights1 * self.agg1(y_out1 / weights1,
                                          im_params1['pixels_h'],
                                          im_params1['pixels_w'])
            y_out1 = self.ver_hor_bn_re_agg1_post(y_out1)

            # -- combine --
            out = self.sep_part2(th.cat((x0, x1, y_out0, y_out1), dim=-2))

        return out


class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        for layer in range(6):
            self.add_module('fc{}'.format(layer), nn.Linear(in_features=14,
                                                            out_features=14,
                                                            bias=False))
            self.add_module('bn{}'.format(layer), nn.BatchNorm1d(14))
            self.add_module('relu{}'.format(layer), nn.ReLU())

        self.add_module('fc_out', nn.Linear(in_features=14, out_features=14, bias=True))

    def forward(self, x):
        for name, module in self._modules.items():
            if 'bn' in name:
                images, patches, values = x.shape
                x = module(x.view(images * patches, values)).view(images, patches, values)
            else:
                x = module(x)
        return x


class PatchDenoiseNet(nn.Module):
    def __init__(self, arch_opt, patch_w, ver_size):
        super(PatchDenoiseNet, self).__init__()
        self.arch_opt = arch_opt
        self.separable_fc_net = SeparableFcNet(arch_opt=arch_opt, patch_w=patch_w, ver_size=ver_size)
        self.weights_net0 = FcNet()
        self.alpha0 = nn.Parameter(th.tensor((0.5,), dtype=th.float32))
        self.beta = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

        self.weights_net1 = FcNet()
        self.alpha1 = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

    def forward(self, patches_n0, dist0, patches_n1, dist1, im_params0, im_params1,
                save_memory, max_chunk):

        # print("dist0.shape: ",dist0.shape)
        weights0 = self.weights_net0(th.exp(-self.alpha0.abs() * dist0)).unsqueeze(-1)
        weighted_patches_n0 = patches_n0 * weights0

        weights1 = self.weights_net1(th.exp(-self.alpha1.abs() * dist1)).unsqueeze(-1)
        weighted_patches_n1 = patches_n1 * weights1
        weights1_first = weights1[:, :, 0:1, :]

        noise = self.separable_fc_net(weighted_patches_n0, weighted_patches_n1,
                                      weights1_first,
                                      im_params0, im_params1, save_memory, max_chunk)

        # print("patches_n0.shape: ",patches_n0.shape)
        patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
        patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
        patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
        patch_weights = th.exp(-self.beta.abs() * patch_exp_weights)
        # print_extrema("patch_weights",patch_weights)

        return patches_dn, patch_weights


@clean_code.add_methods_from(_modules)
class NonLocalDenoiser(nn.Module):
    def __init__(self, pad_offs, arch_opt):
        super(NonLocalDenoiser, self).__init__()
        self.arch_opt = arch_opt
        self.pad_offs = pad_offs

        self.patch_w = 5 if arch_opt.rgb else 7
        self.ver_size = 80 if arch_opt.rgb else 64

        self.rgb2gray = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), bias=False)
        self.rgb2gray.weight.data = th.tensor([0.2989, 0.5870, 0.1140], dtype=th.float32).view(1, 3, 1, 1)
        self.rgb2gray.weight.requires_grad = False

        kernel_1d = th.tensor((1 / 4, 1 / 2, 1 / 4), dtype=th.float32)
        kernel_2d = (kernel_1d.view(-1, 1) * kernel_1d).view(1, 1, 3, 3)
        self.bilinear_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False)
        self.bilinear_conv.weight.data = kernel_2d
        self.bilinear_conv.weight.requires_grad = False

        self.patch_denoise_net = PatchDenoiseNet(arch_opt=arch_opt, patch_w=self.patch_w, ver_size=self.ver_size)

    def find_nn(self, image_pad, im_params, patch_w, scale=0,
                case=None, img_search=None):
        neigh_patches_w = 2 * 14 + 1
        top_dist = th.zeros(im_params['batches'], im_params['patches_h'],
                               im_params['patches_w'], neigh_patches_w ** 2,
                               device=image_pad.device).fill_(float('nan'))
        dist_filter = th.ones(1, 1, patch_w, patch_w, device=image_pad.device)
        top_dist[...] = float("inf")

        if img_search is None: img_search = image_pad
        y = image_pad[:, :, 14:14 + im_params['pixels_h'],
                      14:14 + im_params['pixels_w']]
        for row in range(neigh_patches_w):
            for col in range(neigh_patches_w):
                lin_ind = row * neigh_patches_w + col
                y_shift = image_pad[:, :,row:row + im_params['pixels_h'],
                                    col:col + im_params['pixels_w']]
                top_dist[:, :, :, lin_ind] = conv2d(((y_shift - y)) ** 2,\
                                                    dist_filter).squeeze(1)

        top_dist, top_ind = th.topk(top_dist, 14, dim=3, largest=False, sorted=True)
        # print(top_dist[0,16,16,:])
        top_ind_rows = top_ind // neigh_patches_w
        top_ind_cols = top_ind % neigh_patches_w
        col_arange = th.arange(im_params['patches_w'], device=image_pad.device).view(1, 1, -1, 1)
        row_arange = th.arange(im_params['patches_h'], device=image_pad.device).view(1, -1, 1, 1)
        if scale == 1:
            if case == '00':
                top_ind_rows = top_ind_rows * 2 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + col_arange * 2
            elif case == '10':
                top_ind_rows = top_ind_rows * 2 + 1 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + col_arange * 2
            elif case == '01':
                top_ind_rows = top_ind_rows * 2 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + 1 + col_arange * 2
            elif case == '11':
                top_ind_rows = top_ind_rows * 2 + 1 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + 1 + col_arange * 2
            else:
                assert False
        elif scale == 0:
            top_ind_rows += row_arange
            top_ind_cols += col_arange
        else:
            assert False
        top_ind = top_ind_rows * im_params['pad_patches_w_full'] + top_ind_cols

        return top_dist, top_ind

    # def run_pdn(self,patches_n0,dist0,inds0,patches_n1,dist1,inds1):

    #     # -- run sep-net --
    #     h,w = 72,72
    #     agg0,s0,_ = self.run_agg0(patches_n0,dist0,h,w)
    #     h,w = 76,76
    #     agg1,s1,_,_ = self.run_agg1(patches_n1,dist1,h,w)

    #     # -- final output --
    #     print("s0.shape: ",s0.shape)
    #     print("s1.shape: ",s1.shape)
    #     print("agg0.shape: ",agg0.shape)
    #     print("agg1.shape: ",agg1.shape)
    #     inputs = th.cat((s0, s1, agg0, agg1), dim=-2)
    #     sep_net = self.patch_denoise_net.separable_fc_net
    #     noise = sep_net.sep_part2(inputs)

    #     # -- compute denoised patches --
    #     image_dn,patches_w = self.run_pdn_final(patches_n0,noise)

    #     return image_dn,patches_w

    # def run_parts(self,noisy,sigma):

    #     #
    #     # -- Prepare --
    #     #

    #     # -- normalize for input ---
    #     noisy = (noisy/255. - 0.5)/0.5
    #     means = noisy.mean((-2,-1),True)
    #     noisy -= means

    #     #
    #     # -- Non-Local Search --
    #     #

    #     # -- [nn0 search]  --
    #     output0 = self.run_nn0(noisy)
    #     patches0 = output0[0]
    #     dists0 = output0[1]
    #     inds0 = output0[2]

    #     # -- [nn1 search]  --
    #     output1 = self.run_nn1(noisy)
    #     patches1 = output1[0]
    #     dists1 = output1[1]
    #     inds1 = output1[2]

    #     #
    #     # -- Separable ConvNet --
    #     #

    #     # -- reshape --
    #     patches0 = rearrange(patches0,'t h w k d -> t (h w) k d')
    #     dists0 = rearrange(dists0,'t h w k -> t (h w) k')
    #     inds0 = rearrange(inds0,'t h w k tr -> t (h w) k tr')
    #     patches1 = rearrange(patches1,'t h w k d -> t (h w) k d')
    #     dists1 = rearrange(dists1,'t h w k -> t (h w) k')
    #     inds1 = rearrange(inds1,'t h w k tr -> t (h w) k tr')

    #     # -- exec --
    #     image_dn,patches_w = self.run_pdn(patches0,dists0,inds0,
    #                                       patches1,dists1,inds1)

    #     #
    #     # -- Final Weight Aggregation --
    #     #

    #     h,w = 64,64
    #     image_dn = self.run_nld_final(image_dn,patches_w,h,w)

    #     # -- normalize for output ---
    #     image_dn += means
    #     image_dn = 255*(image_dn * 0.5 + 0.5)

    #     return image_dn

    # def run_nld_final(self,image_dn,patch_weights,h,w):

    #     # -- prepare --
    #     np = 68 # ??
    #     ps = self.patch_w
    #     pdim = image_dn.shape[-1]
    #     image_dn = image_dn * patch_weights
    #     ones_tmp = th.ones(1, 1, pdim, device=image_dn.device)
    #     patch_weights = (patch_weights * ones_tmp).transpose(2, 1)
    #     image_dn = image_dn.transpose(2, 1)

    #     # -- fold --
    #     print("image_dn.shape: ",image_dn.shape)
    #     print("patch_weights.shape: ",patch_weights.shape)
    #     h,w = 72,72
    #     image_dn = fold(image_dn, (h,w),(ps,ps))
    #     patch_cnt = fold(patch_weights, (h,w),(ps,ps))

    #     # -- crop --
    #     row_offs = min(ps - 1, np - 1)
    #     col_offs = min(ps - 1, np - 1)
    #     image_dn = crop_offset(image_dn, (row_offs,), (col_offs,))
    #     image_dn /= crop_offset(patch_cnt, (row_offs,), (col_offs,))

    #     return image_dn

    # def run_pdn_final(self,patches_n0,noise):
    #     pdn = self.patch_denoise_net
    #     patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
    #     patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
    #     patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
    #     patch_weights = th.exp(-pdn.beta.abs() * patch_exp_weights)
    #     return patches_dn,patch_weights

    # def run_agg0(self,patches,dist0,h,w):

    #     # -- compute weights --
    #     pdn = self.patch_denoise_net
    #     weights0 = pdn.weights_net0(th.exp(-pdn.alpha0.abs() * dist0)).unsqueeze(-1)
    #     weighted_patches = patches * weights0

    #     # -- compute sep layers --
    #     sep_net = self.patch_denoise_net.separable_fc_net
    #     x0 = sep_net.sep_part1_s0(weighted_patches)
    #     y_out0 = sep_net.ver_hor_agg0_pre(x0)
    #     y_out0,fold_out0 = sep_net.agg0(y_out0, h, w,both=True)
    #     y_out0 = sep_net.ver_hor_bn_re_agg0_post(y_out0)

    #     return y_out0,x0,fold_out0

    # def run_agg1(self,patches,dist1,h,w):

    #     # -- compute weights --
    #     pdn = self.patch_denoise_net
    #     weights1 = pdn.weights_net1(th.exp(-pdn.alpha1.abs() * dist1)).unsqueeze(-1)
    #     weighted_patches = patches * weights1
    #     weights1 = weights1[:, :, 0:1, :]
    #     wpatches = weighted_patches

    #     # -- compute patches output --
    #     sep_net = self.patch_denoise_net.separable_fc_net
    #     x1 = sep_net.sep_part1_s1(wpatches)
    #     y_out1 = sep_net.ver_hor_agg1_pre(x1)
    #     y_out1,fold_out1 = sep_net.agg1(y_out1 / weights1,h,w,both=True)
    #     y_out1 = weights1 * y_out1
    #     y_out1 = sep_net.ver_hor_bn_re_agg1_post(y_out1)

    #     return y_out1,x1,fold_out1,wpatches

    # def run_nn0(self,image_n,train=False):

    #     # -- pad & unpack --
    #     patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    #     device = image_n.device
    #     image_n0 = self.pad_crop0(image_n, self.pad_offs, train)

    #     # -- get search image --
    #     if self.arch_opt.rgb: img_nn0 = self.rgb2gray(image_n0)
    #     else: img_nn0 = image_n0

    #     # -- get image-based parameters --
    #     im_params0 = get_image_params(image_n0, self.patch_w, 14)
    #     im_params0['pad_patches_w_full'] = im_params0['pad_patches_w']

    #     # -- run knn search --
    #     top_dist0, top_ind0 = self.find_nn(img_nn0, im_params0, self.patch_w)

    #     # -- prepare [dists,inds] --
    #     ip = im_params0['pad_patches']
    #     patch_dist0 = top_dist0.view(top_dist0.shape[0], -1, 14)[:, :, 1:]
    #     top_ind0 += ip * th.arange(top_ind0.shape[0],device=device).view(-1, 1, 1, 1)

    #     # -- get all patches -
    #     patches = unfold(image_n0, (self.patch_w, self.patch_w)).\
    #         transpose(1, 0).contiguous().view(patch_numel, -1).t()

    #     # -- organize patches by knn --
    #     patches = patches[top_ind0.view(-1), :].\
    #         view(top_ind0.shape[0], -1, 14, patch_numel)

    #     # -- append anchor patch spatial variance --
    #     patch_var0 = patches[:, :, 0, :].std(dim=-1).\
    #         unsqueeze(-1).pow(2) * patch_numel
    #     patch_dist0 = th.cat((patch_dist0, patch_var0), dim=-1)

    #     # -- convert to 3d inds --
    #     t,c,h,w = image_n0.shape
    #     ps = self.patch_w
    #     ch,cw = h-(ps-1),w-(ps-1)
    #     k = top_ind0.shape[-1]
    #     inds3d = get_3d_inds(top_ind0.view(-1,1,1,k),ch,cw)

    #     # -- rescale inds --
    #     inds3d[...,1] -= 14
    #     inds3d[...,2] -= 14

    #     # -- format [dists,inds] --
    #     sp = int(np.sqrt(patches.shape[1]))
    #     patches = rearrange(patches,'t (h w) k d -> t h w k d',h=sp)
    #     patch_dist0 = rearrange(patch_dist0,'t (h w) k -> t h w k',h=sp)
    #     inds3d = rearrange(inds3d,'(t h w) k tr -> t h w k tr',t=t,h=sp)

    #     return patches,patch_dist0,inds3d

    # def run_nn1(self,image_n,train=False):

    #     # -- misc unpack --
    #     ps = self.patch_w

    #     # -- pad & unpack --
    #     patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    #     device = image_n.device
    #     image_n1 = self.pad_crop1(image_n, train, 'reflect')
    #     im_n1_b, im_n1_c, im_n1_h, im_n1_w = image_n1.shape

    #     # -- bilinear conv & crop --
    #     image_n1 = image_n1.view(im_n1_b * im_n1_c, 1,im_n1_h, im_n1_w)
    #     image_n1 = self.bilinear_conv(image_n1)
    #     image_n1 = image_n1.view(im_n1_b, im_n1_c, im_n1_h - 2, im_n1_w - 2)
    #     image_n1 = self.pad_crop1(image_n1, train, 'constant')

    #     # -- img-based parameters --
    #     im_params1 = get_image_params(image_n1, 2 * self.patch_w - 1, 28)
    #     im_params1['pad_patches_w_full'] = im_params1['pad_patches_w']

    #     # -- get search image  --
    #     if self.arch_opt.rgb: img_nn1 = self.rgb2gray(image_n1)
    #     else: img_nn1 = image_n1

    #     # -- spatially split image --
    #     img_nn1_00 = img_nn1[:, :, 0::2, 0::2].clone()
    #     img_nn1_10 = img_nn1[:, :, 1::2, 0::2].clone()
    #     img_nn1_01 = img_nn1[:, :, 0::2, 1::2].clone()
    #     img_nn1_11 = img_nn1[:, :, 1::2, 1::2].clone()

    #     # -- get split image parameters --
    #     im_params1_00 = get_image_params(img_nn1_00, self.patch_w, 14)
    #     im_params1_10 = get_image_params(img_nn1_10, self.patch_w, 14)
    #     im_params1_01 = get_image_params(img_nn1_01, self.patch_w, 14)
    #     im_params1_11 = get_image_params(img_nn1_11, self.patch_w, 14)
    #     im_params1_00['pad_patches_w_full'] = im_params1['pad_patches_w']
    #     im_params1_10['pad_patches_w_full'] = im_params1['pad_patches_w']
    #     im_params1_01['pad_patches_w_full'] = im_params1['pad_patches_w']
    #     im_params1_11['pad_patches_w_full'] = im_params1['pad_patches_w']

    #     # -- run knn search! --
    #     top_dist1_00, top_ind1_00 = self.find_nn(img_nn1_00, im_params1_00,
    #                                              self.patch_w, scale=1, case='00')
    #     top_dist1_10, top_ind1_10 = self.find_nn(img_nn1_10, im_params1_10,
    #                                              self.patch_w, scale=1, case='10')
    #     top_dist1_01, top_ind1_01 = self.find_nn(img_nn1_01, im_params1_01,
    #                                              self.patch_w, scale=1, case='01')
    #     top_dist1_11, top_ind1_11 = self.find_nn(img_nn1_11, im_params1_11,
    #                                              self.patch_w, scale=1, case='11')

    #     # -- aggregate results [dists] --
    #     top_dist1 = th.zeros(im_params1['batches'], im_params1['patches_h'],
    #                             im_params1['patches_w'], 14, device=device)
    #     top_dist1 = top_dist1.fill_(float('nan'))
    #     top_dist1[:, 0::2, 0::2, :] = top_dist1_00
    #     top_dist1[:, 1::2, 0::2, :] = top_dist1_10
    #     top_dist1[:, 0::2, 1::2, :] = top_dist1_01
    #     top_dist1[:, 1::2, 1::2, :] = top_dist1_11

    #     # -- aggregate results [inds] --
    #     ipp = im_params1['pad_patches']
    #     top_ind1 = ipp * th.ones(top_dist1.shape,dtype=th.int64,device=device)
    #     top_ind1[:, 0::2, 0::2, :] = top_ind1_00
    #     top_ind1[:, 1::2, 0::2, :] = top_ind1_10
    #     top_ind1[:, 0::2, 1::2, :] = top_ind1_01
    #     top_ind1[:, 1::2, 1::2, :] = top_ind1_11
    #     top_ind1 += ipp * th.arange(top_ind1.shape[0],device=device).view(-1, 1, 1, 1)

    #     # -- get all patches --
    #     im_patches_n1 = unfold(image_n1, (self.patch_w, self.patch_w),
    #                            dilation=(2, 2)).transpose(1, 0).contiguous().\
    #                            view(patch_numel, -1).t()
    #     print("[modules] im_patches_n1.shape: ",im_patches_n1.shape)

    #     # -- organize by knn --
    #     np = top_ind1.shape[0]
    #     pn = patch_numel
    #     im_patches_n1 = im_patches_n1[top_ind1.view(-1), :].view(np, -1, 14, pn)
    #     print("[modules] im_patches_n1.shape: ",im_patches_n1.shape)

    #     # -- append anchor patch spatial variance --
    #     patch_dist1 = top_dist1.view(top_dist1.shape[0], -1, 14)[:, :, 1:]
    #     patch_var1 = im_patches_n1[:, :, 0, :].std(dim=-1).unsqueeze(-1).pow(2) * pn
    #     patch_dist1 = th.cat((patch_dist1, patch_var1), dim=-1)

    #     #
    #     # -- Final Formatting --
    #     #

    #     # -- inds -> 3d_inds --
    #     pad = 2*(ps//2) # dilation "= 2"
    #     _t,_c,_h,_w = image_n1.shape
    #     hp,wp = _h-2*pad,_w-2*pad
    #     top_ind1 = get_3d_inds(top_ind1,hp,wp)

    #     # -- rescale inds --
    #     top_ind1[...,1] -= 28
    #     top_ind1[...,2] -= 28

    #     # -- re-shaping --
    #     t,h,w,k = top_dist1.shape
    #     pdist = rearrange(patch_dist1,'t (h w) k -> t h w k',h=h)
    #     top_ind1 = rearrange(top_ind1,'(t h w) k tr -> t h w k tr',t=t,h=h)
    #     ip1 = im_patches_n1
    #     ip1 = rearrange(ip1,'t (h w) k d -> t h w k d',h=h)

    #     return ip1,pdist,top_ind1

    def denoise_image(self, image_n, train, save_memory, max_batch,
                      srch_img=None, srch_flows=None):

        patch_numel = (self.patch_w ** 2) * image_n.shape[1]
        image_n0 = self.pad_crop0(image_n, self.pad_offs, train)
        if self.arch_opt.rgb:
            image_for_nn0 = self.rgb2gray(image_n0)
        else:
            image_for_nn0 = image_n0

        im_params0 = get_image_params(image_n0, self.patch_w, 14)
        im_params0['pad_patches_w_full'] = im_params0['pad_patches_w']
        top_dist0, top_ind0 = self.find_nn(image_for_nn0, im_params0, self.patch_w)
        top_ind0 += im_params0['pad_patches'] * th.arange(top_ind0.shape[0],
                                                             device=image_n0.device).view(-1, 1, 1, 1)
        patch_dist0 = top_dist0.view(top_dist0.shape[0], -1, 14)[:, :, 1:]

        im_patches_n0 = unfold(image_n0, (self.patch_w, self.patch_w)).transpose(1, 0)\
            .contiguous().view(patch_numel, -1).t()
        im_patches_n0 = im_patches_n0[top_ind0.view(-1), :].view(top_ind0.shape[0], -1, 14, patch_numel)
        patch_var0 = im_patches_n0[:, :, 0, :].\
            std(dim=-1).unsqueeze(-1).pow(2) * patch_numel
        patch_dist0 = th.cat((patch_dist0, patch_var0), dim=-1)
        image_n1 = self.pad_crop1(image_n, train, 'reflect')
        im_n1_b, im_n1_c, im_n1_h, im_n1_w = image_n1.shape
        image_n1 = self.bilinear_conv(image_n1.view(im_n1_b * im_n1_c, 1,
                                                    im_n1_h, im_n1_w))\
            .view(im_n1_b, im_n1_c, im_n1_h - 2, im_n1_w - 2)
        image_n1 = self.pad_crop1(image_n1, train, 'constant')

        # -- params --
        im_params1 = get_image_params(image_n1, 2 * self.patch_w - 1, 28)
        im_params1['pad_patches_w_full'] = im_params1['pad_patches_w']

        # -- convert search image  --
        if self.arch_opt.rgb: img_nn1 = self.rgb2gray(image_n1)
        else: img_nn1 = image_n1

        # -- assignment --
        img_nn1_00 = img_nn1[:, :, 0::2, 0::2].clone()
        img_nn1_10 = img_nn1[:, :, 1::2, 0::2].clone()
        img_nn1_01 = img_nn1[:, :, 0::2, 1::2].clone()
        img_nn1_11 = img_nn1[:, :, 1::2, 1::2].clone()

        # -- diff --
        diff_00_01 = img_nn1_00-img_nn1_10
        diff_00_01 = th.abs(diff_00_01)
        diff_00_01 -= diff_00_01.min()
        diff_00_01 /= diff_00_01.max()

        # -- get image --
        im_params1_00 = get_image_params(img_nn1_00, self.patch_w, 14)
        im_params1_10 = get_image_params(img_nn1_10, self.patch_w, 14)
        im_params1_01 = get_image_params(img_nn1_01, self.patch_w, 14)
        im_params1_11 = get_image_params(img_nn1_11, self.patch_w, 14)
        im_params1_00['pad_patches_w_full'] = im_params1['pad_patches_w']
        im_params1_10['pad_patches_w_full'] = im_params1['pad_patches_w']
        im_params1_01['pad_patches_w_full'] = im_params1['pad_patches_w']
        im_params1_11['pad_patches_w_full'] = im_params1['pad_patches_w']
        top_dist1_00, top_ind1_00 = self.find_nn(img_nn1_00, im_params1_00,
                                                 self.patch_w, scale=1, case='00')
        top_dist1_10, top_ind1_10 = self.find_nn(img_nn1_10, im_params1_10,
                                                 self.patch_w, scale=1, case='10')
        top_dist1_01, top_ind1_01 = self.find_nn(img_nn1_01, im_params1_01,
                                                 self.patch_w, scale=1, case='01')
        top_dist1_11, top_ind1_11 = self.find_nn(img_nn1_11, im_params1_11,
                                                 self.patch_w, scale=1, case='11')
        top_dist1 = th.zeros(im_params1['batches'], im_params1['patches_h'],
                                im_params1['patches_w'], 14,
                                device=image_n.device).fill_(float('nan'))
        top_dist1[:, 0::2, 0::2, :] = top_dist1_00
        top_dist1[:, 1::2, 0::2, :] = top_dist1_10
        top_dist1[:, 0::2, 1::2, :] = top_dist1_01
        top_dist1[:, 1::2, 1::2, :] = top_dist1_11


        top_ind1 = im_params1['pad_patches'] * th.ones(top_dist1.shape,
                                                          dtype=th.int64,
                                                          device=image_n.device)
        top_ind1[:, 0::2, 0::2, :] = top_ind1_00
        top_ind1[:, 1::2, 0::2, :] = top_ind1_10
        top_ind1[:, 0::2, 1::2, :] = top_ind1_01
        top_ind1[:, 1::2, 1::2, :] = top_ind1_11

        top_ind1 += im_params1['pad_patches'] * th.arange(top_ind1.shape[0],
                                                             device=image_n1.device).\
                                                             view(-1, 1, 1, 1)

        im_patches_n1 = unfold(image_n1, (self.patch_w, self.patch_w),
                               dilation=(2, 2)).transpose(1, 0).contiguous().\
                               view(patch_numel, -1).t()

        im_patches_n1 = im_patches_n1[top_ind1.view(-1), :].\
            view(top_ind1.shape[0], -1, 14, patch_numel)
        patch_dist1 = top_dist1.view(top_dist1.shape[0], -1, 14)[:, :, 1:]
        patch_var1 = im_patches_n1[:, :, 0, :].std(dim=-1).\
            unsqueeze(-1).pow(2) * patch_numel

        # print("[pre cat] patch_dist1.shape: ",patch_dist1.shape)
        patch_dist1 = th.cat((patch_dist1, patch_var1), dim=-1)

        # # print("patch_dist1.shape: ",patch_dist1.shape)
        # save_image(patch_dist1,"patch_dist1")
        # print("im_patches_n0.shape: ",im_patches_n0.shape)
        # print("im_patches_n1.shape: ",im_patches_n1.shape)
        # print("patch_dist0.shape: ",patch_dist0.shape)
        # print("patch_dist1.shape: ",patch_dist1.shape)
        # print("im_params0 ",im_params0)

        #
        # -- denoising --
        #

        print_extrema("im_patches_n0",im_patches_n0)
        print_extrema("im_patches_n1",im_patches_n1)
        image_dn, patch_weights = self.patch_denoise_net(im_patches_n0, patch_dist0,
                                                         im_patches_n1, patch_dist1,
                                                         im_params0, im_params1,
                                                         save_memory, max_batch)
        # print("image_dn.shape: ",image_dn.shape)
        # print("patch_weights.shape: ",patch_weights.shape)
        image_dn = image_dn * patch_weights
        ones_tmp = th.ones(1, 1, patch_numel, device=im_patches_n0.device)
        patch_weights = (patch_weights * ones_tmp).transpose(2, 1)
        image_dn = image_dn.transpose(2, 1)
        # print("[pre-fold]: image_dn.shape: ",image_dn.shape)
        image_dn = fold(image_dn, (im_params0['pixels_h'], im_params0['pixels_w']),
                        (self.patch_w, self.patch_w))
        patch_cnt = fold(patch_weights, (im_params0['pixels_h'],im_params0['pixels_w']),
                         (self.patch_w, self.patch_w))
        # print("[post-fold]: image_dn.shape: ",image_dn.shape)
        # print("patch_cnt.shape: ",patch_cnt.shape)

        row_offs = min(self.patch_w - 1, im_params0['patches_h'] - 1)
        col_offs = min(self.patch_w - 1, im_params0['patches_w'] - 1)
        image_dn = crop_offset(image_dn, (row_offs,), (col_offs,)) / crop_offset(patch_cnt, (row_offs,), (col_offs,))

        return image_dn

    def forward_patches(self,agg_vid,agg_weight,
                        im_patches_n0,patch_dist0,inds0,
                        im_patches_n1,patch_dist1,inds1,
                        im_params0,im_params1,save_memory,max_batch):
        """

        Forward pass after patches are retrieved.
        im_patches_nX = (nframes,npatches,nsims,dsize)

        im_paramsX = get_image_params(image_n0, self.patch_w, 14)
        -> image_n0.shape = (t,c,h,w)

        save_memory = True

        """

        # -- forward pass --
        patch_numel = im_patches_n0.shape[-1]
        image_dn, patch_weights = self.patch_denoise_net(im_patches_n0, patch_dist0,
                                                         im_patches_n1, patch_dist1,
                                                         im_params0, im_params1,
                                                         save_memory, max_batch)
        image_dn = image_dn * patch_weights
        ones_tmp = th.ones(1, 1, patch_numel, device=im_patches_n0.device)
        patch_weights = (patch_weights * ones_tmp).transpose(2, 1)
        image_dn = image_dn.transpose(2, 1)

        # -- update aggregation --


    def pad_crop0(self, image, pad_offs, train):
        if not train:
            reflect_pad = [self.patch_w - 1] * 4
            constant_pad = [14] * 4
            image = nn_func.pad(nn_func.pad(image, reflect_pad, 'reflect'), constant_pad, 'constant', -1)
        else:
            image = crop_offset(image, (pad_offs,), (pad_offs,))

        return image

    def pad_crop1(self, image, train, mode):
        if not train:
            if mode == 'reflect':
                bilinear_pad = 1
                averaging_pad = (self.patch_w - 1) // 2
                patch_w_scale_1 = 2 * self.patch_w - 1
                find_nn_pad = (patch_w_scale_1 - 1) // 2
                reflect_pad = [averaging_pad + bilinear_pad + find_nn_pad] * 4
                image = nn_func.pad(image, reflect_pad, 'reflect')
            elif mode == 'constant':
                constant_pad = [28] * 4
                image = nn_func.pad(image, constant_pad, 'constant', -1)
            else:
                assert False

        return image

    def forward(self, image_n, train=True, save_memory=False, max_chunk=None, srch_img=None, srch_flows=None):

        save_image(image_n,"image_n")
        image_n_mean = image_n.mean(dim=(-2, -1), keepdim=True)
        image_n = image_n - image_n_mean
        image_dn = self.denoise_image(image_n, train, save_memory,
                                      max_chunk, srch_img, srch_flows)
        image_dn = image_dn + image_n_mean

        return image_dn


class Logger(object):
    def __init__(self, fname="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def __del__(self):
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
