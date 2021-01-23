#!usr/bin/python
# -*- coding: utf-8 -*-
from config import cfg

class CFG:
    def __init__(self, cfg=cfg):
        self.model_name = 'tf_efficientnet_b4_ns'
        self.img_size = cfg.MODEL.CLASSES
        scheduler = 'CosineAnnealingWarmRestarts'
        T_max = 10
        T_0 = 10
        lr = 1e-4
        min_lr = 1e-6
        batch_size = 16
        weight_decay = 1e-6
        seed = 42
        num_classes = 5
        num_epochs = 10
        n_fold = 5
        NUM_FOLDS_TO_RUN = [2, ]
        smoothing = 0.2
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def a(x):
    b = x+1
    return b





