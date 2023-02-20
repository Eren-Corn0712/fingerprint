import contextlib
import hashlib
import os
import subprocess
import time
from pathlib import Path
from tarfile import is_tarfile
from zipfile import is_zipfile

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps

HELP_URL = "See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def img2label_paths(img_paths):
    # Define labels paths as a function of image paths
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def verify_image_label(args):
    # Verify one image-labels pair
    im_file, lb_file, prefix = args
    # number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg", "png"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # labels found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            nl = len(lb)

        return im_file, lb, shape, nm, nf, ne, nc, msg

    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, nm, nf, ne, nc, msg]


def find_file(root: str, fmt: str = 'png', recursive: bool = False) -> List[Path]:
    if recursive:
        p = Path(root).rglob(f'*.{fmt}')
    else:
        p = Path(root).glob(f'*.{fmt}')

    return sorted(list(p))


def register_enroll(path: Path, fmt: str = 'png'):
    enroll_paths = find_file(str(path / 'enroll'), fmt, recursive=True)
    return enroll_paths


def register_verify(path: Path, fmt: str = 'png'):
    verify_paths = find_file(str(path / 'verify'), fmt, recursive=True)
    return verify_paths


def find_dir(root: str) -> List[Path]:
    p_list = []
    for p in Path(root).glob('*'):
        if p.is_dir():
            p_list.append(p)

    return p_list


def register_user(root: str) -> List:
    users = find_dir(root)
    if len(users) == 0:
        raise ValueError("We did not find users")
    return users


def register_finger(users, fmt: str = 'png'):
    d = {}
    log = ('\n' + '%12s' * 4) % ('USER', 'Finger-num', 'Enroll', 'Verify')
    total_enroll_n, total_verify_n, total_finger_n = 0, 0, 0
    for user in users:
        d.setdefault(user.name, {})
        fingers = find_dir(user)
        enroll_n, verify_n = 0, 0
        for finger in fingers:
            d[user.name].setdefault(finger.name, {})
            enroll_paths = register_enroll(finger, fmt)
            verify_paths = register_verify(finger, fmt)
            d[user.name][finger.name]['enroll'] = enroll_paths
            d[user.name][finger.name]['verify'] = verify_paths
            enroll_n += len(enroll_paths)
            verify_n += len(verify_paths)
        total_enroll_n += enroll_n
        total_verify_n += verify_n
        total_finger_n += len(fingers)

        log += ('\n' + '%12s' * 4) % (user.name, len(fingers), enroll_n, verify_n)
    # log output
    log += ('\n' + '%12s' * 4) % ('Tot-User', 'Tot-Finger', 'Tot-Enroll', 'Tot-Verify')
    log += ('\n' + '%12s' * 4) % (len(list(d.keys())), total_finger_n, total_enroll_n, total_verify_n)
    print(log)

    return d, log


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield key, value


def count_keys(dictionary):
    count = 0
    for key, value in dictionary.items():
        if type(value) is dict:
            count += count_keys(value)
        elif type(value) is list:
            count += len(value)
        else:
            count += 1
    return count


def nested_dict_to_list(d):
    result = []
    for key, value in d.items():
        if isinstance(value, dict):
            result.extend(nested_dict_to_list(value))
        elif type(value) is list:
            result.extend(value)
        else:
            result.append((key, value))
    return result
