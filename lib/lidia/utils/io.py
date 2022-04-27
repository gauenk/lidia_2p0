
import torch as th
import numpy as np
from PIL import Image
from pathlib import Path
from einops import repeat,rearrange


def save_burst(burst,name):
    nframes = len(burst)
    for t in range(nframes):
        img_t = burst[t]
        name_t = "%s_%02d" % (name,t)
        save_image(img_t,name_t)

def save_image(img,name):

    # -- paths --
    root = Path("./output/nl_modules/")
    if not root.exists(): root.mkdir()

    # -- add batch dim --
    batch_dim = img.dim() == 4
    if batch_dim is False: img = img[None,:]

    # -- format --
    img = img.cpu().numpy()
    img = rearrange(img,'t c h w -> t h w c')
    if img.max().item() < 10:
        img = (img * 255.)

    # -- clip to legal values --
    img = np.clip(img,0.,255.)

    # -- final formatting --
    img = img.astype(np.uint8)
    msg = f"Can't Save It: img.shape = {img.shape}"
    assert img.shape[-1] in [1,3],msg

    # -- bw to color --
    if img.shape[-1] == 1:
        img = repeat(img,'t h w 1 -> t h w c',c=3)

    # -- save --
    nframes = img.shape[0]
    for t in range(nframes):
        fn = str(root / ("%s_%05d.png" % (name,t)))
        img_t = img[t]
        img_t = Image.fromarray(img_t)
        img_t.save(fn)
