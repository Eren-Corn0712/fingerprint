import math
import random
import cv2
import numpy as np
import torchvision.transforms.functional as F

from torchvision import datasets, transforms

from torchvision.transforms.functional import get_dimensions
from skimage.filters.rank import equalize
from skimage.morphology import disk
from typing import List, Tuple
from PIL import ImageFilter, ImageOps, Image


def pil_to_cv2(pil_image):
    if pil_image.mode == 'L':  # 灰度圖像
        cv2_image = np.array(pil_image)
    else:  # RGB圖像
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        cv2_image = open_cv_image[:, :, ::-1].copy()
    return cv2_image


def cv2_to_pil(cv2_image: np.ndarray) -> Image:
    if len(cv2_image.shape) == 2:  # 灰度圖像
        pil_image = Image.fromarray(cv2_image)
    else:  # RGB圖像
        # Convert BGR to RGB
        cv2_image = cv2_image[:, :, ::-1]
        pil_image = Image.fromarray(cv2_image)
    return pil_image


def image_remove_bound(image: np.ndarray, size: int = 2):
    return image[size:-size, size:-size]


def crop_to_size(img: Image, target_width, target_height):
    width, height = img.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


class PartialDetection:
    def __init__(self, kernel_size: int = 7, remove_size: int = 2):
        self.kernel_size = kernel_size
        self.remove_size = remove_size

    def __call__(self, image):
        h, w = image.shape[:2]
        image = np.float32(image)
        crop_img = image_remove_bound(image, size=self.remove_size)

        partial_img = crop_img - crop_img.max()  # Must be converted to np.float32, otherwise, it will cause overflow.
        partial_mask = np.zeros((h - self.remove_size * 2, w - self.remove_size * 2))
        partial_mask[partial_img < -20] = 1  # Background Part

        # Morphological Transformations : Close -> Erode
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        partial_mask = cv2.morphologyEx(partial_mask, cv2.MORPH_CLOSE, kernel)
        partial_mask = cv2.morphologyEx(partial_mask, cv2.MORPH_ERODE, kernel)

        # Recover the mask size
        top = bottom = left = right = self.remove_size
        re_partial_mask = cv2.copyMakeBorder(src=1 - partial_mask, top=top, bottom=bottom, left=left, right=right,
                                             borderType=cv2.BORDER_REPLICATE)

        re_partial_mask = 1 - re_partial_mask
        return re_partial_mask.astype('bool')


class MaskMean(object):
    def __init__(self, ):
        pass

    def __call__(self, image, mask):
        if mask is None:
            raise ValueError("Mask is none")
        masked_region = image[mask]
        normalized_image = image - masked_region.mean() + 128
        normalized_image[normalized_image < 0] = 0
        normalized_image[normalized_image > 255] = 255

        return normalized_image.astype('uint8')


class Equalize(object):
    def __init__(self, radius=20):
        self.radius = radius
        self.footprint = disk(radius)

    def __call__(self, image):
        enhanced_image = equalize(image, self.footprint)
        return enhanced_image.astype('uint8')


class EGISPreprocess(object):
    def __init__(self):
        # Default EGIS Preprocess on fingerprint image!
        self.partial_detection = PartialDetection()
        self.mask_mean = MaskMean()
        self.equalize = Equalize()

    def __call__(self, img):
        img = pil_to_cv2(img)
        fg_mask = self.partial_detection(img)
        img = self.mask_mean(img, fg_mask)
        img = self.equalize(img)
        img = cv2_to_pil(img)
        return img


class RandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if not random.random() <= self.prob:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class RandomRotateAndCrop(object):
    def __init__(self, angle: List, p: float = 0.5):
        self.p = p
        self.angle = angle

    def __call__(self, img):
        do_it = random.random() <= self.p
        if not do_it:
            return img
        angle = random.choice(self.angle)
        return self.rotate_and_crop(img, int(angle))

    def rotate_and_crop(self, img, angle):
        _, img_h, img_w = get_dimensions(img)
        rotated_img = F.rotate(img, angle=int(angle), expand=True)
        w, h = self.largest_rotated_rect(img_w, img_h, math.radians(angle))
        largest_rotated_rect_img = F.center_crop(rotated_img, [int(h), int(w)])
        return self.crop_square_image(largest_rotated_rect_img)

    def crop_square_image(self, img):
        _, h, w = get_dimensions(img)
        left, upper, right, lower = self.get_crop_params(h, w)
        return img.crop((left, upper, right, lower))

    def get_crop_params(self, h, w):
        size = np.array([h, w])
        min_ind, d = np.argmin(size), np.min(size)
        offset_d = int((np.max(size) - np.min(size)) / 2)
        offset_seq = self.create_arithmetic_sequence(-offset_d, offset_d, 1)
        offset_d = random.choice(offset_seq)
        w_offset, h_offset = 0, 0
        if min_ind == 0:
            w_offset = offset_d
            # Calculate the starting point (left, upper, right, lower) for cropping the image
        if min_ind == 1:
            h_offset = offset_d
        left = (w - d) // 2 + w_offset
        upper = (h - d) // 2 + h_offset
        right = (w + d) // 2 + w_offset
        lower = (h + d) // 2 + h_offset
        return left, upper, right, lower

    @staticmethod
    def create_arithmetic_sequence(a, b, d):
        return [x for x in range(a, b + 1, d)]

    @staticmethod
    def largest_rotated_rect(w, h, angle) -> Tuple[float, float]:
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)
        return bb_w - 2 * x, bb_h - 2 * y


class FingerPrintDataAug(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size=96):
        if not isinstance(local_crops_size, tuple) or not isinstance(local_crops_size, list):
            local_crops_size = list(local_crops_size)

        if not isinstance(local_crops_number, tuple) or not isinstance(local_crops_number, list):
            local_crops_number = list(local_crops_number)

        self.local_crops_number = local_crops_number

        normalize = transforms.Compose([
            transforms.ToTensor(),
        ])
        resize_filp_affine = transforms.Compose([transforms.Resize(size=(32, 128)),
                                                 transforms.RandomHorizontalFlip(0.5),
                                                 transforms.RandomVerticalFlip(0.5)])
        # first global crop
        self.glo_trans = transforms.Compose([resize_filp_affine,
                                             RandomGaussianBlur(0.5), normalize])
        # transformation for the local small crops

        self.loc_trans = []
        for l_size in local_crops_size:
            self.loc_trans.append(transforms.Compose([
                # transforms.RandomResizedCrop(l_size, scale=local_crops_scale, ratio=(1.0, 1.0)),
                # MyRotationTransform([0, 45, 90, 135, 180, 225, 270, 315]),
                RandomRotateAndCrop(angle=list(range(0, 360 + 1, 45)), p=1.0),
                transforms.Resize(size=(l_size, l_size)),
                transforms.RandomEqualize(p=0.1),
                RandomGaussianBlur(0.5),
                normalize,
            ]))

    def __call__(self, labels):
        crops = []
        labels['img1'] = crop_to_size(labels['img1'], 128, 32)
        labels['img2'] = crop_to_size(labels['img2'], 128, 32)
        crops.append(self.glo_trans(labels['img1']))
        crops.append(self.glo_trans(labels['img2']))
        for i, n_crop in enumerate(self.local_crops_number):
            for _ in range(n_crop):
                crops.append(self.loc_trans[i](labels['img1'])) if i < n_crop // 2 else crops.append(
                    self.loc_trans[i](labels['img2']))
        labels['multi'] = crops
        return labels


class FingerPrintDataAug_1(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size=96):
        if not isinstance(local_crops_size, tuple) or not isinstance(local_crops_size, list):
            local_crops_size = list(local_crops_size)

        if not isinstance(local_crops_number, tuple) or not isinstance(local_crops_number, list):
            local_crops_number = list(local_crops_number)

        self.local_crops_number = local_crops_number

        normalize = transforms.Compose([
            transforms.ToTensor(),
        ])
        resize_filp_affine = transforms.Compose([transforms.RandomResizedCrop(size=(32, 128),
                                                                              scale=global_crops_scale,
                                                                              ratio=(4, 4)),
                                                 transforms.RandomHorizontalFlip(0.5),
                                                 transforms.RandomVerticalFlip(0.5)])
        # first global crop
        self.glo_trans = transforms.Compose([resize_filp_affine,
                                             RandomGaussianBlur(0.5),
                                             normalize])
        # transformation for the local small crops

        self.loc_trans = []
        for l_size in local_crops_size:
            self.loc_trans.append(transforms.Compose([
                transforms.RandomResizedCrop(l_size, scale=local_crops_scale, ratio=(1.0, 1.0)),
                transforms.RandomRotation(degrees=(-180, 180)),
                RandomGaussianBlur(0.5),
                normalize,
            ]))

    def __call__(self, labels):
        crops = []
        labels['img1'] = crop_to_size(labels['img1'], 128, 32)
        labels['img2'] = crop_to_size(labels['img2'], 128, 32)
        crops.append(self.glo_trans(labels['img1']))
        crops.append(self.glo_trans(labels['img2']))
        for i, n_crop in enumerate(self.local_crops_number):
            for _ in range(n_crop):
                crops.append(self.loc_trans[i](labels['img1'])) if i < n_crop // 2 else crops.append(
                    self.loc_trans[i](labels['img2']))
        labels['multi'] = crops
        return labels


class FingerPrintDataAug_2(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size=96):
        if not isinstance(local_crops_size, tuple) or not isinstance(local_crops_size, list):
            local_crops_size = list(local_crops_size)

        if not isinstance(local_crops_number, tuple) or not isinstance(local_crops_number, list):
            local_crops_number = list(local_crops_number)

        self.local_crops_number = local_crops_number

        normalize = transforms.Compose([
            transforms.ToTensor(),
        ])

        random_select = transforms.RandomChoice([
            transforms.RandomResizedCrop(size=(32, 128), scale=global_crops_scale, ratio=(4, 4)),
            EGISPreprocess(),
        ], p=[0.5, 0.5])
        # first global crop
        self.glo_trans = transforms.Compose([random_select,
                                             transforms.RandomHorizontalFlip(0.5),
                                             transforms.RandomVerticalFlip(0.5),
                                             RandomGaussianBlur(0.5),
                                             normalize])
        # transformation for the local small crops

        self.loc_trans = []
        for l_size in local_crops_size:
            self.loc_trans.append(transforms.Compose([
                transforms.RandomResizedCrop(l_size, scale=local_crops_scale, ratio=(1.0, 1.0)),
                transforms.RandomRotation(degrees=(-180, 180)),
                RandomGaussianBlur(0.5),
                normalize,
            ]))

    def __call__(self, labels):
        crops = []
        labels['img1'] = crop_to_size(labels['img1'], 128, 32)
        labels['img2'] = crop_to_size(labels['img2'], 128, 32)
        crops.append(self.glo_trans(labels['img1']))
        crops.append(self.glo_trans(labels['img2']))
        for i, n_crop in enumerate(self.local_crops_number):
            for _ in range(n_crop):
                crops.append(self.loc_trans[i](labels['img1'])) if i < n_crop // 2 else crops.append(
                    self.loc_trans[i](labels['img2']))
        labels['multi'] = crops
        return labels


class InferenceFingerPrintAug(object):
    def __init__(self, size=(128, 32)):
        self.size = size

    def __call__(self, labels):
        labels['img'] = crop_to_size(labels['img1'], 128, 32)
        labels['img'] = transforms.ToTensor()(labels['img'])
        return labels


class PairFingerPrintAug(object):
    def __init__(self, size: Tuple[int, int] = (128, 32), is_train: bool = True):
        self.size = size
        self.is_train = True
        self.train_transform = transforms.Compose([
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, labels):
        if self.is_train:
            labels['img1'] = crop_to_size(labels['img1'], 128, 32)
            labels['img1'] = self.train_transform(labels['img1'])
            labels['img2'] = crop_to_size(labels['img1'], 128, 32)
            labels['img2'] = self.train_transform(labels['img2'])
        else:
            labels['img1'] = crop_to_size(labels['img1'], 128, 32)
            labels['img1'] = self.test_transform(labels['img1'])
            labels['img2'] = crop_to_size(labels['img1'], 128, 32)
            labels['img2'] = self.test_transform(labels['img2'])
        return labels
