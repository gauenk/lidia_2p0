
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- vision --
from PIL import Image

# -- paths --
from pathlib import Path


def save_burst(burst,path,name):
    # -- append burst dim --
    if burst.dim() == 3:
        burst = burst[None,:]
    assert burst.dim() == 4,"must be 4-dims"

    # -- create path --
    if isinstance(path,str):
        path = Path(path)
        if not path.exists():
            path.mkdir()

    # -- save each frame --
    nframes = burst.shape[0]
    for t in range(nframes):
        img_t = burst[t]
        fn_t = str(path / ("%s_%05d.png" % (name,t)))
        save_image(img_t,fn_t)

def save_image(image,path):

    # -- to numpy --
    if th.is_tensor(image):
        image = image.cpu().numpy()

    # -- to uint8 --
    if image.max() < 100:
        image = image*255.
    image = np.clip(image,0,255).astype(np.uint8)

    # -- save --
    image = rearrange(image,'c h w -> h w c')
    img = Image.fromarray(image)
    img.save(path)


