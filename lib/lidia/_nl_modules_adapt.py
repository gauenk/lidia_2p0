"""
Functions for internal domain adaptation.
"""


import math

from torch.nn.functional import unfold
from torch.nn.functional import fold
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
from .model_io import select_sigma,get_default_opt


import torch as th
from pathlib import Path
from einops import repeat

# -- separate class and logic --
from .utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Run Adaptation of the Network to Image
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_adapt(self,noisy,sigma,srch_img=None,flows=None):

    opt = get_default_opt(sigma)
    total_pad = 20
    nadapts = 1
    for astep in range(nadapts):
        clean = self.run_parts(noisy,sigma,srch_img,flows).clamp(-1, 1)
        nl_denoiser = adapt_step(self, clean,
                                 srch_img, flows, opt, total_pad)

def adapt_step(nl_denoiser, clean, srch_img, flows, opt, total_pad):

    # -- optims --
    criterion = th.nn.MSELoss(reduction='mean')
    optim = th.optim.Adam(nl_denoiser.parameters(), lr=opt.lr,
                              betas=(0.9, 0.999), eps=1e-8)

    # -- get data --
    loader,batch_last_it = get_adapt_dataset(clean,opt,total_pad)

    # -- train --
    noisy = add_noise_to_image(clean, opt.sigma)

    # -- epoch --
    for epoch in range(opt.epoch_num):

        # -- info --
        print('Training epoch {} of {}'.format(epoch + 1, opt.epoch_num))

        # -- garbage collect --
        sys.stdout.flush()
        gc.collect()
        torch.cuda.empty_cache()

        # -- loaders --
        device = next(nl_denoiser.parameters()).device
        iloader = enumerate(loader)
        for i, noisy in iloader:

            clean = clean.to(device=device)
            noisy = clean + sigma_255_to_torch(opt.sigma) * torch.randn_like(clean)

            optim.zero_grad()
            image_dn = nl_denoiser.run_parts(noisy,opt.sigma,srch_img,flows)
            image_dn = image_dn.clamp(-1,1)

            total_pad = (clean.shape[-1] - image_dn.shape[-1]) // 2
            image_ref = crop_offset(clean, (total_pad,), (total_pad,))
            loss = torch.log10(criterion(image_dn, image_ref))
            assert not np.isnan(loss.item())
            loss.backward()
            optim.step()

            if i == batch_last_it and (epoch + 1) % opt.epochs_between_check == 0:
                gc.collect()
                torch.cuda.empty_cache()
                deno = self.run_parts(noisy,opt.sigma,srch_img,flows)
                deno = deno.clamp(-1, 1).cpu()
                mse = criterion(deno / 2,clean / 2).item()
                train_psnr = -10 * math.log10(mse)
                a,b,c = epoch + 1, opt.epoch_num, train_psnr
                msg = 'Epoch {} of {} done, training PSNR = {:.2f}'.format(a,b,c)
                print(msg)
                sys.stdout.flush()

    return nl_denoiser


def get_adapt_dataset(clean,opt,total_pad):
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    RandomTranspose(),
                                    transforms.ToTensor(),
                                    ShiftImageValues(),
                                    ])
    block_w_pad = opt.block_w + 2 * total_pad
    print("clean.shape: ",clean.shape,total_pad)
    th_img = tensor_to_ndarray_uint8(clean)
    print("th_img.shape: ",th_img.shape)
    th_img = rearrange(th_img,'b h c w -> b c h w')
    print("th_img.shape: ",th_img.shape)
    dset = ImageDataSet(block_w=block_w_pad, images=th_img,
                             transform=transform, stride=opt.dset_stride)
    loader = data.DataLoader(dset,batch_size=opt.train_batch_size,
                             shuffle=True, num_workers=0)
    dlen = loader.dataset.__len__()
    dbs = loader.batch_size
    batch_last_it = dlen // dbs - 1
    return loader,batch_last_it


def add_noise_to_image(clean, sigma):
    noisy = clean + sigma_255_to_torch(sigma) * torch.randn_like(clean)
    return noisy

def sigma_255_to_torch(sigma_255):
    return (sigma_255 / 255) / 0.5



