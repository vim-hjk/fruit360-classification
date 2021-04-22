import os
import torch
import random
import numpy as np

from importlib import import_module
from adamp import AdamP
from madgrad import MADGRAD
from pytorch_ranger import Ranger
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from loss import *
from dataset import Fruit360Dataset


# Fix random seed
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def get_optimizer_and_scheduler(optimizers):
    param, optimizer_name, lr, weight_decay, scheduler_name = optimizers

    optimizer_module = getattr(import_module('adamp'), optimizer_name)
    scheduler_module = getattr(import_module('torch.optim.lr_scheduler'), scheduler_name)

    optimizer = optimizer_module(param, lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_module(optimizer, T_max=50, eta_min=0)

    return optimizer, scheduler     


def get_dataloader(df, transform, batch_size, shuffle):
    dataset = Fruit360Dataset(df=df, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def rand_bbox(size, lam):
    width = size[2]
    height = size[3]
    cut_ratio = np.sqrt(1. - lam)
    cut_width = np.int(width * cut_ratio)
    cut_height = np.int(height * cut_ratio)

    # uniform
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_width // 2, 0, width)
    bby1 = np.clip(cy - cut_height // 2, 0, height)
    bbx2 = np.clip(cx + cut_width // 2, 0, width)
    bby2 = np.clip(cy + cut_height // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


class CutMix(object):
    def __init__(self, beta, cutmix_prob) -> None:
        super().__init__()
        self.beta = beta
        self.cutmix_prob = cutmix_prob 

    def forward(self, images, labels):
        # generate mixed sample
        lam = np.random.beta
        rand_index = torch.randperm(images.size()[0]).cuda()
        label_1 = labels
        label_2 = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        return {'lam' : lam, 'image' : images, 'label_1' : label_1, 'label_2' : label_2}


class ComputeMetric(object):
    def __init__(self, metric) -> None:
        super().__init__() 
        self.metric = metric    

    def cutmix_accuracy(self, logits, labels, topk=(1, 5)):
        max_k = max(topk)
        batch_size = labels.size(0)

        _, pred = logits.topk(max_k, 1, True, True)
        pred = pred.t()
        matches = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            matches_k = matches[:k].reshape(-1).float().sum(0, keepdim=True)
            wrong_k = batch_size - matches_k
            res.append(matches_k.mul_(100.0 / batch_size))

        return res

    def compute(self, logits, labels, topk=(1, 5)):
        if self.metric == 'accuracy':
            out = self.cutmix_accuracy(logits=logits, labels=labels, topk=topk)

        return out


class AverageMeter(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count