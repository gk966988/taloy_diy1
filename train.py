#!usr/bin/python
# -*- coding: utf-8 -*-
import time
import torch
import random
import os, sys
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import logging
from tqdm import tqdm
from torch.cuda import amp
from torch.optim import lr_scheduler
from config import cfg
import numpy as np
import torch.optim as optim
from models.net import get_net
import torch.nn as nn
from dataset import CassavaLeafDataset, data_transforms, get_transform
from torch.utils.data import DataLoader
from utils.utils import AverageMeter, TopKAccuracyMetric
from loss import TaylorCrossEntropyLoss

ROOT_DIR = r"../data/cassava-leaf-disease-classification"
TRAIN_DIR = r"../data/cassava-leaf-disease-classification/train_images"
TEST_DIR = r"../data/cassava-leaf-disease-classification/test_images"
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1,2))

torch.backends.cudnn.benchmark = True

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(
            save_dir, "train_log_{}.txt".format(name)), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def fetch_scheduler(optimizer, cfg):
    if cfg.SOLVER.SCHEDULER == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    elif cfg.SOLVER.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    elif cfg.SOLVER.SCHEDULER == None:
        return None
    return scheduler


set_seed(cfg.MODEL.SEED)

df = pd.read_csv(f"{ROOT_DIR}/train.csv")

skf = StratifiedKFold(n_splits=cfg.SOLVER.N_FOLD)
for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
    df.loc[val_, "kfold"] = int(fold)

df['kfold'] = df['kfold'].astype(int)



def train_model(dataloaders, fold, log, cfg):
    best_acc = 0.0
    # history = defaultdict(list)
    scaler = amp.GradScaler()
    model = get_net(cfg.MODEL.NAME, 5, 'github')
    weight_path = f"./weights/Fold{fold}.bin"
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        state_dict = checkpoint['state_dict']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(state_dict)
        log.info('Network loaded from {}'.format(weight_path))
    model.to(device)
    # model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=1e-6, amsgrad=False)
    criterion = TaylorCrossEntropyLoss(n=2, smoothing=0.2)
    scheduler = fetch_scheduler(optimizer, cfg)
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        print('Epoch {}/{}'.format(epoch, cfg.SOLVER.MAX_EPOCHS))
        log.info('Epoch {} / {}'.format(epoch, cfg.SOLVER.MAX_EPOCHS))
        loss_container = AverageMeter(name='loss')
        raw_metric = TopKAccuracyMetric(topk=(1, 2))
        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:
            if (phase == 'train'):
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            pbar = tqdm(enumerate(dataloaders[phase]),
                        total=int(len(dataloaders[phase].dataset) / cfg.DATASETS.BATCH_SIZE))
            pbar.set_description('Train Epoch {}/{}'.format(epoch, cfg.SOLVER.MAX_EPOCHS))
            # Iterate over data
            for batch_idx, (inputs, labels, _) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        epoch_loss = loss_container(loss.item())
                        epoch_acc = raw_metric(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                pbar.set_postfix_str('Loss: {:.2f}  Train acc@1: {:.2f}'.format(epoch_loss, epoch_acc[0]))
            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc[0]))
            log.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc[0]))

            # deep copy the model
            if phase == 'valid' and epoch_acc[0] >= best_acc:
                best_acc = epoch_acc[0]
                path = f"./weights/Fold{fold+1}.bin"
                if torch.cuda.device_count() > 1:
                    torch.save({'best_acc':best_acc, 'state_dict':model.module.state_dict()}, path)
                else:
                    torch.save({'best_acc': best_acc, 'state_dict': model.state_dict()}, path)


def run_fold(fold, log, cfg):
    valid_df = df[df.kfold == fold]
    train_df = df[df.kfold != fold]
    # transforms = data_transforms(cfg.DATASETS.IMG_SIZE)

    train_data = CassavaLeafDataset(TRAIN_DIR, train_df, transforms=get_transform(cfg.DATASETS.IMG_SIZE, phase='train'))
    valid_data = CassavaLeafDataset(TRAIN_DIR, valid_df, transforms=get_transform(cfg.DATASETS.IMG_SIZE, phase='test'))

    train_loader = DataLoader(dataset=train_data, batch_size=cfg.DATASETS.IMG_SIZE, num_workers=4, pin_memory=True,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.DATASETS.IMG_SIZE, num_workers=4, pin_memory=True,
                              shuffle=False)

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    train_model(dataloaders, fold, log, cfg)



def main(file_name, log):
    set_seed(cfg.MODEL.SEED)
    config_file = './configs/' + file_name
    cfg.merge_from_file(config_file)
    cfg.freeze()
    for fold in range(cfg.SOLVER.N_FOLD):
        print(f"\n\nFOLD: {fold}\n\n")
        log.info(f"\n\nFOLD: {fold}\n\n")

        run_fold(fold, log, cfg)


if __name__=="__main__":
    congfig_files = {'efficientnetb4.yaml'}
    for file_name in congfig_files:
        log = setup_logger(file_name, './log')
        main(file_name, log)









