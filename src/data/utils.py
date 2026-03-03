import os
import random
import torch
import pathlib
import tifffile
import openslide
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from csbdeep.data import Normalizer, normalize_mi_ma


class CustomNormalizer(Normalizer):
    def __init__(self, mi, ma):
        self.mi, self.ma = mi, ma

    def before(self, x, axes):
        return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)

    def after(*args, **kwargs):
        assert False

    @property
    def do_after(self):
        return False


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def read_image(
    img_path: str,
    single_patch=False,
    return_res=False,
    loc=(0, 0),
    level=0,
    size=(4000, 4000),
):
    """
    Read the image in either tif, png, jpg, or svs format.
    """

    suffix = pathlib.Path(img_path).suffix

    if not os.path.exists(img_path):
        print(f"This image was not found: {img_path}")
        return

    if suffix in [".tif", ".tiff"]:
        img = tifffile.imread(img_path)
        return img

    elif suffix in [".png", ".jpg", ".jpeg"]:
        img_PIL = Image.open(img_path)
        img = np.array(img_PIL.convert("RGB"))
        return img

    elif suffix in [".svs"]:
        wsi = openslide.OpenSlide(img_path)
        if single_patch:
            img = np.array(wsi.read_region(loc, level, size).convert("RGB"))
        else:
            img = np.array(
                wsi.read_region((0, 0), 0, wsi.level_dimensions[0]).convert("RGB")
            )

        if return_res:
            mpp_x = float(wsi.properties["openslide.mpp-x"])
            mpp_y = float(wsi.properties["openslide.mpp-y"])
            dims = wsi.dimensions
            assert mpp_x == mpp_y, "MPP x and y are different"
            return img, mpp_x, dims

        else:
            return img

    else:
        raise NotImplementedError


def coords_to_roi(coords: np.ndarray, p_dims: tuple) -> np.ndarray:
    """
    Converts the list of coordinations to an array of segmentations where
    zero is background and the other values represent the cells.
    """
    mask = Image.fromarray(np.zeros(p_dims, dtype=np.int16))
    draw = ImageDraw.Draw(mask)

    nuc_idx = 1
    for nuc in range(coords.shape[0]):
        nuc_coords = coords[nuc]  # size 2, 32
        nuc_coords = nuc_coords.T
        nuc_coords = [(row[1], row[0]) for row in nuc_coords]
        draw.polygon(nuc_coords, fill=(nuc_idx))
        nuc_idx += 1

    return np.array(mask)
