#!usr/bin/python
# -*- coding: utf-8 -*-
import os
import cv2
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.augmentations import *

class CassavaLeafDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None):
        self.root_dir = root_dir
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[index, 1]

        if self.transforms:
            img = self.transforms(img)

        return img, label, self.df.iloc[index, 0]

def data_transforms(img_size):

    data_transforms = {
        "train": A.Compose([
            A.RandomResizedCrop(img_size, img_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            ToTensorV2()], p=1.),

        "valid": A.Compose([
            A.CenterCrop(img_size, img_size, p=1.),
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)
    }
    return data_transforms


def get_transform(resize, phase='train'):
    if phase == 'train':
        return Compose([
            Resize(size=(int(resize / 0.8), int(resize / 0.8))),
            RandomCrop([resize, resize]),
            HorizontalFilp(),
            VerticalFlip(),
            RandomRotate(30),
            dropout(rate=(0,0.2)),       #Cutout
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor(),

        ])
    elif phase == 'test':
        return Compose([
            Resize(size=(int(resize), int(resize))),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor(),
        ])

    elif phase == 'tta':
        aug0 = Compose([
            Resize(size=(int(resize/0.8), int(resize/0.8))),
            RandomCrop([resize, resize]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor(),
        ])
        aug1 =  Compose([
            Resize(size=(int(resize/0.8), int(resize/0.8))),
            RandomCrop([resize, resize]),
            HorizontalFilp(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor(),
        ])
        aug2 = Compose([
            Resize(size=(int(resize / 0.8), int(resize / 0.8))),
            RandomCrop([resize, resize]),
            VerticalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor(),
        ])

        return [aug0, aug1, aug2]











