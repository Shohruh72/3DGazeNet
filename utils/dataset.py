import os

import cv2
import torch
import numpy as np
import mxnet as mx
from torch.utils import data
import albumentations as album
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


class Dataset(data.Dataset):
    def __init__(self, args, train=True, num_eye=481):
        self.args = args
        self.train = train
        self.num_eye = num_eye
        self.transform = self._get_transform()
        self.images, self.labels = self._load_label()
        print(f'Number of images: {len(self.labels)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        idx = self.labels[index]
        header, img = mx.recordio.unpack(self.images.read_idx(idx))
        img = mx.image.imdecode(img).asnumpy()
        points = np.array(header.label, dtype=np.float32).reshape((-1, 3))
        assert points.shape[0] == self.num_eye * 2
        eye_l, eye_r = self.normalize_points(points)
        points = np.concatenate((eye_l, eye_r), axis=0)
        kps_xy = []
        kps_z = []
        for i in range(points.shape[0]):
            kps_xy.append((points[i][0], points[i][1]))
            kps_z.append(points[i][2])
        if self.transform is not None:
            t = self.transform(image=img, keypoints=kps_xy)
            img = t['image']
            label_xy = np.array(t['keypoints'], dtype=np.float32)
            label_xy /= (self.args.input_size / 2)
            label_xy -= 1.0
            label_z = np.array(kps_z, dtype=np.float32).reshape((-1, 1))
            label = np.concatenate((label_xy, label_z), axis=1)
            label = torch.tensor(label, dtype=torch.float32)
        return img, label

    def _get_transform(self):
        _transforms = [
            album.geometric.resize.Resize(self.args.input_size,
                                          self.args.input_size,
                                          interpolation=cv2.INTER_LINEAR, always_apply=True),
            album.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()]

        if self.train:
            train_transforms = [
                album.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
                album.ToGray(p=0.1),
                album.ISONoise(p=0.1),
                album.MedianBlur(blur_limit=(1, 7), p=0.1),
                album.GaussianBlur(blur_limit=(1, 7), p=0.1),
                album.MotionBlur(blur_limit=(5, 13), p=0.1),
                album.ImageCompression(quality_lower=10, quality_upper=90, p=0.05),
                album.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
                                       interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0,
                                       mask_value=0, p=0.6),
                # A.HorizontalFlip(p=0.5),
                RectangleBorderAugmentation(limit=0.2, fill_value=0, p=0.1, always_apply=False),
            ]
            _transforms = train_transforms + _transforms

        return album.ReplayCompose(_transforms,
                                   keypoint_params=album.KeypointParams(format='xy', remove_invisible=False))

    def _load_label(self):
        image_rec = os.path.join(self.args.data_dir, 'train.rec' if self.train else 'val.rec')
        label_rec = os.path.join(self.args.data_dir, 'train.idx' if self.train else 'val.idx')
        images = mx.recordio.MXIndexedRecordIO(label_rec, image_rec, 'r')
        labels = np.array(list(images.keys))
        return images, labels

    def normalize_points(self, points):
        eye_l = points[:self.num_eye, :]
        eye_r = points[self.num_eye:, :]
        mean_z_l = np.mean(eye_l[:32, 2])
        mean_z_r = np.mean(eye_r[:32, 2])
        std_z_l = np.max(np.abs(eye_l[:32, 2]))
        std_z_r = np.max(np.abs(eye_r[:32, 2]))
        eye_l[:, 2] -= mean_z_l
        eye_r[:, 2] -= mean_z_r
        eye_l[:, 2] /= std_z_l
        eye_r[:, 2] /= std_z_r

        return eye_l, eye_r


class RectangleBorderAugmentation(ImageOnlyTransform):
    def __init__(self, limit=0.3, fill_value=0, always_apply=False, p=1.0):
        super(RectangleBorderAugmentation, self).__init__(always_apply=always_apply, p=p)
        assert 0.0 < limit < 1.0, "Limit must be between 0 and 1."
        self.fill_value = fill_value
        self.limit = limit

    def apply(self, image, border_size_limit=None, **params):
        if border_size_limit is None:
            border_size_limit = self.get_params()['border_size_limit']
        border_size = border_size_limit.copy()
        border_size[0] *= image.shape[1]
        border_size[2] *= image.shape[1]
        border_size[1] *= image.shape[0]
        border_size[3] *= image.shape[0]
        border_size = border_size.astype(int)
        image[:, :border_size[0], :] = self.fill_value
        image[:border_size[1], :, :] = self.fill_value
        image[:, border_size[2]:, :] = self.fill_value
        image[border_size[3]:, :, :] = self.fill_value
        return image

    def get_params(self):
        border_size_limit = np.random.uniform(0.0, self.limit, size=4)
        return {'border_size_limit': border_size_limit}

    def get_transform_init_args_names(self):
        return ('fill_value', 'limit')
