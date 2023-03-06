import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from pathlib import Path

from models import custom_model
from matcher_tool.engine.matcher import BaseMatcher
from matcher_tool.utils import DEFAULT_CFG, TQDM_BAR_FORMAT, is_dir_writeable
from matcher_tool.data.fingerprint_dataset_ver1 import FingerPrintDataset
from matcher_tool.utils.torch_utils import select_device, FeatureExtractor, init_seeds
from matcher_tool.utils.files import increment_path
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from matcher_tool.data.augment import InferenceFingerPrintAug, EGISInferenceFingerPrintAug
from tqdm import tqdm


class DINOModelMatcher(BaseMatcher):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super().__init__(cfg, overrides)
        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers
        init_seeds()
        self.train_dataset = self.get_dataset(self.args.train_data)
        self.test_dataset = self.get_dataset(self.args.test_data)
        self.train_dataloader = self.get_dataloader(self.train_dataset)
        self.test_dataloader = self.get_dataloader(self.test_dataset)

        self.device = select_device(self.args.device, self.args.batch_size)
        self.model = self.get_model().to(self.device, non_blocking=True)
        self.model.eval()

        self.load_esvit_pretrained_weights(self.model, self.args.weights, "teacher")

        self.feature_extractor = FeatureExtractor(self.model, self.args.extract_layers)
        self.distance = self.args.distance
        if self.distance == 'linear':
            self.classifier = self.get_classifier(960 * 2, num_classes=2).to(self.device, non_blocking=True)
            self.load_classifier(self.classifier, self.args.classifier_weights)
            self.classifier.eval()

    @staticmethod
    def load_esvit_pretrained_weights(model, pretrained_weights, checkpoint_key):
        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            state_dict = {k.replace("module.backbone.backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        else:
            raise ValueError("Error loading")

    @staticmethod
    def load_classifier(model, pretrained_weights):
        if os.path.exists(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            msg = model.load_state_dict(state_dict["state_dict"], strict=True)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        else:
            raise ValueError(f"The path {pretrained_weights}")

    def get_model(self):
        arch = self.args.model
        if arch in torchvision.models.__dict__.keys():
            model = torchvision.models.__dict__[arch]()
            model.fc = nn.Identity()
        elif arch in custom_model.__dict__.keys():
            model = custom_model.__dict__[arch]()
            model.fc = nn.Identity()
        return model

    def get_classifier(self, dim: int, num_classes: int = 2):
        from eval_linear_finger import LinearClassifier, Classifier
        return Classifier(dim, num_classes)

    def get_transform(self):
        if self.args.transform == 'normal':
            transform = InferenceFingerPrintAug()
        elif self.args.transform == 'egis':
            transform = EGISInferenceFingerPrintAug()
        else:
            raise ValueError(f"Unknown Transform : {self.args.transform}.")
        return transform

    def get_dataset(self, data):
        return FingerPrintDataset(data,
                                  transform=self.get_transform())

    def get_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True)

    def get_feature(self, dataloader, prefix=""):
        path = self.save_dir / f"{prefix}.cache"

        if path.exists():
            cache = np.load(str(path), allow_pickle=True).item()
            return cache

        self.log(f"{prefix} data feature extracting.")
        self.log("Set Model to Evaluation model.")

        feature = {}
        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    bar_format=TQDM_BAR_FORMAT)  # progress bar
        for i, data in pbar:
            imgs, paths = data['img'], data['im_file']
            imgs = imgs.to(self.device, non_blocking=True).float()

            if self.feature_extractor:
                self.feature_extractor.get_hooks()
            with torch.no_grad():
                output = self.model(imgs)

            batch_concat_feature = [output.cpu().detach().numpy()]
            if self.feature_extractor:
                for k, v in self.feature_extractor.features.items():
                    avg_fea = F.adaptive_avg_pool2d(v, (1, 1)).flatten(1, 3)  # B C H W -> B C 1 1 -> B X C
                    if avg_fea is not None:
                        batch_concat_feature.append(avg_fea.cpu().detach().numpy())

            batch_concat_feature = np.concatenate(batch_concat_feature, axis=1)

            if self.feature_extractor:
                self.feature_extractor.remove_hooks()

            for p, fea in zip(paths, batch_concat_feature):
                p = Path(p)
                file_name, st, enrl_verf, finger, user, *args = p.parts[::-1]
                k = Path(user) / finger / enrl_verf / st / file_name
                feature[k] = fea

        path.parent.mkdir(parents=True, exist_ok=True)  # make directory
        if is_dir_writeable(path.parent):
            np.save(str(path), feature)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            self.log(f"{prefix} New cache created: {path}")

        return feature

    def select_enroll_verify_fake(self, features: Dict, user, finger):
        # TODO: We can optimize this function by using the filter.
        enroll_features, verify_features, fake_features = {}, {}, {}
        for path, feature in features.items():
            u, f, s = path.parts[:3]
            if u == user and f == finger and s == 'enroll':
                enroll_features[path] = feature
            elif u == user and f == finger and s == 'verify':
                verify_features[path] = feature
            else:
                fake_features[path] = feature
        return enroll_features, verify_features, fake_features

    def log_title(self):
        self.log(('\n' + '%10s' * 5) % ('User', 'Finger', 'Enroll', 'Verify', 'Fake'))

    def match(self, enroll_features, verify_features):
        result = {}
        pbar = tqdm(verify_features.items(), total=len(verify_features), bar_format=TQDM_BAR_FORMAT)
        for v_p, verify_feature in pbar:
            # u, f, s = v_p.parts[:3]
            scores = []
            for e_p, enroll_feature in enroll_features.items():
                if self.distance == "cos":
                    score = np.dot(verify_feature, enroll_feature) / (
                            np.linalg.norm(verify_feature) * np.linalg.norm(enroll_feature))
                    scores.append(score)
                else:
                    raise ValueError(f"Not support {self.distance}")
            result[v_p] = np.array(scores, dtype=np.float64)
        return result

    def classifier_match(self, enroll_features: Dict, verify_features: Dict):
        result = {}
        pbar = tqdm(verify_features.items(), total=len(verify_features), bar_format=TQDM_BAR_FORMAT)
        enroll_features = list(enroll_features.values())
        enroll_features = torch.tensor(enroll_features)
        enroll_features = enroll_features.to(self.device)

        for v_p, verify_feature in pbar:
            verify_feature = torch.from_numpy(verify_feature)[None, ...].to(self.device)
            verify_feature = verify_feature.repeat(enroll_features.shape[0], 1)

            input = torch.cat([verify_feature, enroll_features], dim=1)
            with torch.no_grad():
                score = self.classifier(input)
            scores = score.tolist()
            result[v_p] = np.array(scores, dtype=np.float64)

        return result

    def dataset_match(self, prefix=""):
        features = self.get_feature(getattr(self, f"{prefix}_dataloader"), prefix)
        dataset = getattr(self, f"{prefix}_dataset")
        result_verf2enrl = []
        result_fake2enrl = []

        for user, finger_list in dataset.user_finger.items():
            for finger in finger_list:
                enroll_features, verify_features, fake_features = \
                    self.select_enroll_verify_fake(features, user, finger)
                self.log_title()
                self.log(('%10s' * 5) % (
                    f'{user}', f'{finger}',
                    f'{len(enroll_features)}',
                    f'{len(verify_features)}',
                    f'{len(fake_features)}'))
                if self.distance in ['cos']:
                    verf2enrl = self.match(enroll_features, verify_features)
                    fake2enrl = self.match(enroll_features, fake_features)
                elif self.distance == 'linear':
                    verf2enrl = self.classifier_match(enroll_features, verify_features)
                    fake2enrl = self.classifier_match(enroll_features, fake_features)
                else:
                    raise ValueError(f"{self.distance} is not supported.")
                result_verf2enrl.append(verf2enrl)
                result_fake2enrl.append(fake2enrl)

        file = self.save_dir / f"{prefix}_verf"
        f = str(increment_path(file).with_suffix('.npy'))
        np.save(f, result_verf2enrl)

        file = self.save_dir / f"{prefix}_fake"
        f = str(increment_path(file).with_suffix('.npy'))
        np.save(f, result_fake2enrl)

    def do_match(self):
        self.dataset_match("train")
        self.dataset_match("test")
