import copy
import os
import random
import numpy as np
import torchvision.transforms.functional as F

from pathlib import Path
from typing import Tuple, Dict
from torch.utils.data import DataLoader

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from matcher_tool.data.base import BaseDataset
from matcher_tool.data.utils import find_dir, register_verify, register_enroll, nested_dict_to_list
from matcher_tool.data.augment import FingerPrintDataAug_1
from matcher_tool.data.dataset_wrappers import WrappersDataset

def register_finger_print_data(root, fmt: str = 'png') -> Tuple[Dict, str]:
    users = find_dir(root)
    if len(users) == 0:
        raise ValueError("We did not find users")

    d = {}
    log = ('\n' + '%12s' * 4) % ('USER', 'Finger-num', 'Enroll', 'Verify')
    total_enroll_n, total_verify_n, total_finger_n = 0, 0, 0
    for user in users:
        d.setdefault(user.name, {})
        fingers = find_dir(str(user))
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
    return d, log


class FingerPrintDataset(BaseDataset):
    def __init__(self, img_path: str, label_path: str = None, transform=None):
        super().__init__(img_path, label_path)
        self.user_finger = None
        self.im_files = self.get_img_files(img_path)
        self.txt_files = self.get_txt_files(img_path)
        self.labels = self.get_labels()
        self.transform = transform

    def get_labels(self):
        labels = []
        for im_file in self.im_files:
            file_name, st, enrl_verf, finger, user, *args = im_file.parts[::-1]
            txt_file = Path(im_file).with_suffix(".txt")
            labels.append(dict(
                im_file=str(im_file),
                st=st,
                enrl_verf=enrl_verf,
                finger=finger,
                user=user,
                txt_file=str(txt_file)
            ))

        return labels

    def get_img_files(self, img_path):
        img, log = register_finger_print_data(img_path)
        new_dict = {}
        for key1, inner_dict in img.items():
            new_dict[key1] = list(inner_dict.keys())
        self.user_finger = copy.deepcopy(new_dict)
        return nested_dict_to_list(img)

    def get_txt_files(self, txt_path):
        txt, log = register_finger_print_data(txt_path, 'txt')
        return nested_dict_to_list(txt)

    def get_label_info(self, index):
        label = self.labels[index].copy()
        label = self.update_labels_info(label)
        return label

    def update_labels_info(self, label):
        txt_file = label['txt_file']
        if Path(txt_file).is_file():
            label['overlap'] = self.read_txt(txt_file)
        else:
            label['overlap'] = {}
        return label

    def read_txt(self, txt_file):
        d = {'path': [],
             'score': []}
        with open(str(txt_file), mode="r") as f:
            for line in f.readlines():
                l = line.split(" ")
                score, p = l[0], l[1]
                p = p.rstrip().replace("\\", os.sep)
                d['path'].append(p)
                d['score'].append(score)
        return d

    def load_image(self, im_file) -> Image.Image:
        with open(im_file, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        label = self.get_label_info(index)
        label['img1'] = self.load_image(str(label['im_file']))

        if label['overlap'] == {}:
            label['img2'] = copy.deepcopy(label['img1'])
        else:
            paths_scores = zip(label['overlap']['path'], label['overlap']['score'])
            selected_path = [Path(self.img_path) / p for p, s in paths_scores if float(s) != 0.0]
            if selected_path:
                label['img2'] = self.load_image(str(random.choice(selected_path)))
            else:
                label['img2'] = copy.deepcopy(label['img1'])

        if self.transform:
            label = self.transform(label)
        label.pop('img1', None), label.pop('img2', None), label.pop('overlap', None)
        return label

    def select_dataset(self):
        pass

    def __len__(self):
        return len(self.labels)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == '__main__':
    transform = FingerPrintDataAug_1(global_crops_scale=(1.0, 1.0),
                                     local_crops_number=(16,),
                                     local_crops_scale=(0.15, 0.50),
                                     local_crops_size=(32,))
    fpd = FingerPrintDataset("/home/corn/PycharmProjects/fingerprint/train",
                             transform=transform)
    fpd_loader = DataLoader(fpd, batch_size=4)

    index = 0
    for label in fpd_loader:
        images = label['multi']
        global_view = [view[index] for view in images[:2]]
        local_view = [view[index] for view in images[2:]]
        show(make_grid(global_view))
        show(make_grid(local_view))
        plt.show()
