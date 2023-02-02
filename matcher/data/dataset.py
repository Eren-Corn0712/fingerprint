from typing import Union, Any, Tuple, Dict, List

import numpy as np

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import torchvision
from tqdm import tqdm

from .base import BaseDataset
from pathlib import Path
from .utils import HELP_URL, LOCAL_RANK, img2label_paths, get_hash, verify_image_label
from matcher.utils import NUM_THREADS, TQDM_BAR_FORMAT, is_dir_writeable, LOGGER


class FingerPrintDataset(BaseDataset):
    cache_version = 1.0

    def __init__(
            self,
            img_path: str,
            label_path: str = None
    ):
        super().__init__(img_path, label_path)
        self.label_files = None
        self.user_finger_info: Dict[List] = self.get_user_finger_list()
        print(self.user_finger_info)

    def cache_labels(self, path=Path("./labels.cache")):
        # Cache dataset labels, check images and read shapes
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix)))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                user, finger, enrl_verf = lb[0][0], lb[0][1], lb[0][2]
                if im_file:
                    x["labels"].append(dict(
                        im_file=im_file,
                        shape=shape,
                        user=user,
                        finger=finger,
                        enrl_verf=enrl_verf,
                    ))
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")

        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        self.im_files = [lb["im_file"] for lb in x["labels"]]  # update im_files
        if is_dir_writeable(path.parent):
            np.save(str(path), x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{self.prefix}New cache created: {path}")
        else:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable")  # not writeable
        return x

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache: Dict[str, Union[Union[list, Tuple[int, int, int, int, int], float], Any]]
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

            # Display cache
            nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
            if exists and LOCAL_RANK in {-1, 0}:
                d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                tqdm(None, desc=self.prefix + d, total=n, initial=n,
                     bar_format=TQDM_BAR_FORMAT)  # display cache results
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            assert nf > 0, f"{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]

        return labels

    def get_user_finger_list(self):
        d = {}
        for l in self.labels:
            if l['user'] not in d:
                d[l['user']] = []
            if l['finger'] not in d[l['user']]:
                d[l['user']].append(l['finger'])
        return d
