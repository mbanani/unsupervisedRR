"""
Abstract class that takes care of a lot of the boiler plate code.
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, name, split, data_root):
        # dataset parameters
        self.name = name
        self.root = data_root
        self.split = split

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def get_rgba(self, path, bbox=None):
        path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGBA")
                if bbox is not None:
                    img = img.crop(box=bbox)
                return img

    def get_rgb_alpha(self, path, bbox=None):
        path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                _, _, _, a = img.split()
                img = img.convert("RGB")
                if bbox is not None:
                    img = img.crop(box=bbox)
                    a = a.crop(box=bbox)
                a = np.array(a).astype(dtype=np.float)
                return img, a

    def get_alpha(self, path, bbox=None):
        path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                r, g, b, a = img.split()
                if bbox is not None:
                    a = a.crop(box=bbox)
                a = np.array(a).astype(dtype=np.float)
        return a

    def get_img(self, path, bbox=None):
        path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                if bbox is not None:
                    img = img.crop(box=bbox)
                return np.array(img)

    def get_npy(self, path):
        path = os.path.join(self.root, path)
        return np.load(path)

    def get_rgb(self, path, bbox=None):
        path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGB")
                if bbox is not None:
                    img = img.crop(box=bbox)
                return img

    def get_exr(self, path, bbox=None):
        """
        Source: https://github.com/mlagunas/pytorch-np_transform
        loads an .exr file as a numpy array

        only kept the single dim version
        """
        # get absolute path
        path = os.path.join(self.root, path)

        depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth[:, :, 0]  # all channels are the same

        # crop -- bbox is  (left, upper, right, lower)-tuple.
        if bbox is not None:
            l, u, r, d = bbox
            depth[l:r, u:d]

        return depth
