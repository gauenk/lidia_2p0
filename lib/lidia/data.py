
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- vision --
from PIL import Image


def save_burst(burst,path):
    # -- append burst dim --
    if burst.dim() == 4:
        burst = burst[None,:]

    # -- save each frame --
    nframes = burst.shape[0]
    for t in range(nframes):
        img_t = burst[t]
        fn = str(path / ("%05d.png" % t))
        save_image(img_t,fn)

def save_image(image,path):

    # -- to numpy --
    if th.is_tensor(image):
        image = image.cpu().numpy()

    # -- to uint8 --
    if image.max() < 100:
        image = image*255.
    image = np.clip(image.astype(np.uint8),0,255)

    # -- save --
    image = rearrange(image,'c h w -> h w c')
    img = Image.fromarray(image)
    img.save(path)


