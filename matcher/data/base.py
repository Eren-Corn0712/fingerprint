import glob
import math
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from ..utils import colorstr
from .utils import HELP_URL, IMG_FORMATS, LOCAL_RANK


class BaseDataset(Dataset):
    def __init__(self,
                 img_path: str = None,
                 label_path: str = None,
                 prefix: str = "",
                 cache: bool = False
                 ):
        super().__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.prefix = prefix

        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()

        self.ni = len(self.labels)
        # cache stuff
        self.ims = [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache:
            self.cache_images(cache)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        return im_files

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.get_label_info(index)

    def get_label_info(self, index):
        label = self.labels[index].copy()
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"] = self.load_image(index)

    def load_image(self, i):
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw

            return im, (h0, w0)
        return self.ims[i], self.im_hw0[i]

    def update_labels_info(self, label):
        """custom your label format here"""
        return label
