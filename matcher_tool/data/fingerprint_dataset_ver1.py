import copy
import os
import random
import numpy as np
import torchvision.transforms.functional as F

from pathlib import Path
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from itertools import product
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

from matcher_tool.data.base import BaseDataset
from matcher_tool.data.utils import find_dir, register_verify, register_enroll, nested_dict_to_list
from matcher_tool.data.augment import FingerPrintDataAug_1, \
    FingerPrintDataAug_2, \
    FingerPrintDataAug_3, \
    PairFingerPrintAug, \
    FingerPrintDataAug


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
            return img.convert("L")

    def __getitem__(self, index):
        label = self.get_label_info(index)
        label['img1'] = self.load_image(str(label['im_file']))

        if label['overlap'] == {}:
            label['img2'] = copy.deepcopy(label['img1'])
        else:
            paths_scores = zip(label['overlap']['path'], label['overlap']['score'])
            selected_path = [Path(self.img_path) / p for p, s in paths_scores if float(s) != 0.0]
            selected_path = [p for p in selected_path if p.is_file()]
            if selected_path:
                label['img2'] = self.load_image(str(random.choice(selected_path)))
            else:
                label['img2'] = copy.deepcopy(label['img1'])

        if self.transform:
            label = self.transform(label)
        label.pop('img1', None), label.pop('img2', None), label.pop('overlap', None)
        return label

    def __len__(self):
        return len(self.labels)


class PairImageFingerPrintDataset(FingerPrintDataset):
    def __init__(self, img_path: str, transform):
        super().__init__(img_path, transform=transform)
        self.method = 0
        self.pairs_labels = self.get_pair_labels_impl1()

    def get_pair_labels_impl1(self):
        p = []
        for user, finger_list in self.user_finger.items():
            for finger in finger_list:
                print(user, finger, len(p))
                enroll_labels, verify_labels, fake_labels = [], [], []
                for label in self.labels:
                    u = label['user']
                    f = label['finger']
                    s = label['enrl_verf']
                    if u == user and f == finger and s == 'enroll':
                        enroll_labels.append(label)
                    elif u == user and f == finger and s == 'verify':
                        verify_labels.append(label)
                    else:
                        fake_labels.append(label)
                enroll_labels = random.sample(enroll_labels, min(10, len(enroll_labels)))
                # verify_labels = random.sample(verify_labels, min(5, len(verify_labels)))
                fake_labels = random.sample(fake_labels, min(len(verify_labels), len(fake_labels)))
                for verify_label in verify_labels:
                    verify_label = self.update_labels_info(verify_label)
                    if verify_label['overlap'] == {}:
                        for enroll_label in enroll_labels:
                            d = dict(im_file1=verify_label['im_file'],
                                     im_file2=enroll_label['im_file'],
                                     match=0)
                            p.append(d)
                    else:
                        paths_scores = zip(verify_label['overlap']['path'], verify_label['overlap']['score'])
                        selected_path = [Path(self.img_path) / p for p, s in paths_scores if float(s) != 0.0]
                        if selected_path:
                            for s in selected_path:
                                if not s.is_file():
                                    continue
                                d = dict(im_file1=verify_label['im_file'],
                                         im_file2=str(s),
                                         match=1)
                                p.append(d)
                        else:
                            pass
                            # for enroll_label in enroll_labels:
                            #     d = dict(im_file1=verify_label['im_file'],
                            #              im_file2=enroll_label['im_file'],
                            #              match=0)
                            #     self.pairs_labels.append(d)
                for fake_label in fake_labels:
                    for enroll_label in enroll_labels:
                        d = dict(im_file1=fake_label['im_file'],
                                 im_file2=enroll_label['im_file'],
                                 match=0)
                        p.append(d)
        del self.labels, self.txt_files, self.npy_files
        return p

    def get_pair_labels_impl2(self):
        p = []
        for user, finger_list in self.user_finger.items():
            for finger in finger_list:
                print(user, finger, len(p))
                enroll_labels, verify_labels, fake_labels = [], [], []
                for label in self.labels:
                    u = label['user']
                    f = label['finger']
                    s = label['enrl_verf']
                    if u == user and f == finger and s == 'enroll':
                        enroll_labels.append(label)
                    elif u == user and f == finger and s == 'verify':
                        verify_labels.append(label)
                    else:
                        fake_labels.append(label)
                enroll_labels = random.sample(enroll_labels, min(10, len(enroll_labels)))
                # verify_labels = random.sample(verify_labels, min(5, len(verify_labels)))
                fake_labels = random.sample(fake_labels, min(len(verify_labels), len(fake_labels)))
                for verify_label in verify_labels:
                    for enroll_label in enroll_labels:
                        d = dict(im_file1=verify_label['im_file'],
                                 im_file2=enroll_label['im_file'],
                                 match=1)
                        p.append(d)

                for fake_label in fake_labels:
                    for enroll_label in enroll_labels:
                        d = dict(im_file1=fake_label['im_file'],
                                 im_file2=enroll_label['im_file'],
                                 match=0)
                        p.append(d)

        del self.labels, self.txt_files, self.npy_files
        return p
    @staticmethod
    def select_fake(x, user, finger, enrl_verf):
        return x['user'] != user and x['finger'] != finger and x['enrl_verf'] != enrl_verf

    def __len__(self):
        return len(self.pairs_labels)

    def getitem_impl(self, index):
        label = self.get_label_info(index)
        label['img1'] = self.load_image(str(label['im_file']))
        if label['overlap'] == {}:
            filtered_labels = list(
                filter(
                    lambda x: self.select_fake(x, label['user'], label['finger'], label['enrl_verf']), self.labels))
            filtered_label = random.choice(filtered_labels)
            label['img2'] = self.load_image(str(filtered_label['im_file']))
            label['match'] = 0
        else:
            paths_scores = zip(label['overlap']['path'], label['overlap']['score'])
            selected_path = [Path(self.img_path) / p for p, s in paths_scores if float(s) != 0.0]
            if selected_path:
                label['img2'] = self.load_image(str(random.choice(selected_path)))
                label['match'] = 1
            else:
                label['img2'] = copy.deepcopy(label['img1'])
                label['match'] = 1

        if self.transform:
            label = self.transform(label)
        label.pop('overlap', None)
        return label

    def getitem_impl2(self, index):
        pairs_label = self.pairs_labels[index]
        pairs_label['img1'] = self.load_image(str(pairs_label['im_file1']))
        pairs_label['img2'] = self.load_image(str(pairs_label['im_file2']))
        if self.transform:
            pairs_label = self.transform(pairs_label)
        return pairs_label

    def __getitem__(self, index):
        return self.getitem_impl2(index)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_mutil_crop_arg(data_loader):
    index = 0
    for label in data_loader:
        images = label['multi']
        global_view = [view[index] for view in images[:2]]
        local_view = [view[index] for view in images[2:]]
        show(make_grid(global_view))
        show(make_grid(local_view))
        plt.show()


class Test_Dataset(object):
    def __init__(self):
        self.datasets = FingerPrintDataset("/home/corn/PycharmProjects/fingerprint/train")
        # self.pair_datasets = PairImageFingerPrintDataset("/home/corn/PycharmProjects/fingerprint/train",
        #                                                  transform=PairFingerPrintAug())

    def test_transform(self):
        transform = FingerPrintDataAug(global_crops_scale=(0.5, 1.0),
                                       local_crops_number=(16,),
                                       local_crops_scale=(0.15, 0.50),
                                       local_crops_size=(32,))
        self.datasets.transform = transform
        show_mutil_crop_arg(DataLoader(self.datasets, batch_size=4))

    def test_transform1(self):
        transform = FingerPrintDataAug_1(global_crops_scale=(0.5, 1.0),
                                         local_crops_number=(16,),
                                         local_crops_scale=(0.15, 0.50),
                                         local_crops_size=(32,))
        self.datasets.transform = transform
        show_mutil_crop_arg(DataLoader(self.datasets, batch_size=4))

    def test_transform2(self):
        transform = FingerPrintDataAug_2(global_crops_scale=(0.5, 1.0),
                                         local_crops_number=(16,),
                                         local_crops_scale=(0.15, 0.50),
                                         local_crops_size=(32,))
        self.datasets.transform = transform
        show_mutil_crop_arg(DataLoader(self.datasets, batch_size=4))

    def test_transform3(self):
        transform = FingerPrintDataAug_3(global_crops_scale=(0.5, 1.0),
                                         local_crops_number=(16,),
                                         local_crops_scale=(0.15, 0.50),
                                         local_crops_size=(32,))
        self.datasets.transform = transform
        show_mutil_crop_arg(DataLoader(self.datasets, batch_size=4))

    def test_pair_dataset(self):
        self.pair_datasets.transform = PairFingerPrintAug()
        data_loader = DataLoader(self.pair_datasets, batch_size=128)
        for i, label in enumerate(data_loader):
            print(f"Successfully load {i}")


if __name__ == '__main__':
    test_class = Test_Dataset()
    test_class.test_transform3()