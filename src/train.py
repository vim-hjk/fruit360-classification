import os
import wandb
import numpy as np
import pandas as pd
import albumentations
import albumentations.pytorch
import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm
from importlib import import_module
from torch.nn import CrossEntropyLoss
from torchsummary import summary as summary_
from torch.utils.tensorboard import SummaryWriter

from utils import get_optimizer_and_scheduler, CutMix, AverageMeter, ComputeMetric
from model import PretrainedModel
from loss import LabelSmoothingLoss, F1Loss, FocalLoss


def train(cfg, k, train_loader, val_loader):
    # Set Config
    MODEL_ARC = cfg.values.model_arc
    OUTPUT_DIR = cfg.values.output_dir
    NUM_CLASSES = cfg.values.num_classes
    TRAIN_ONLY = cfg.values.train_only    

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set train arguments
    num_epochs = cfg.values.train_args.num_epochs
    train_batch_size = cfg.values.train_args.train_batch_size
    loss_fn = cfg.values.train_args.loss_fn
    log_intervals = cfg.values.train_args.log_intervals

    # Set CutMix arguments
    USE_CUTMIX = cfg.values.cutmix_args.use_cutmix
    beta = cfg.values.cutmix_args.beta
    cutmix_prob = cfg.values.cutmix_args.cutmix_prob    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PretrainedModel(model_arc=MODEL_ARC, num_classes=NUM_CLASSES)
    model.to(device)
    summary_(model, (3, 224, 224), batch_size=train_batch_size)

    optimizers = [
        model.parameters(),
        cfg.values.train_args.optimizer,
        cfg.values.train_args.lr,
        cfg.values.train_args.weight_decay,
        cfg.values.train_args.scheduler
    ]

    optimizer, scheduler = get_optimizer_and_scheduler(optimizers)
    loss_module = getattr(import_module('torch.nn'), loss_fn)
    criterion = loss_module()

    eval_metric = ComputeMetric(cfg.values.train_args.eval_metric)
    best_acc = 0.
    os.makedirs(os.path.join(OUTPUT_DIR, MODEL_ARC), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()

        loss_values = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, train_batch in enumerate(tqdm(train_loader, desc=f'Training')):
            sample = train_batch
            images = sample['image'].float().to(device)
            labels = sample['label'].long().to(device)

            ratio = np.random.rand(1)

            if USE_CUTMIX:
                if beta > 0 and ratio < cutmix_prob:
                    # generate mixed sample
                    sample = CutMix(images, labels)

                    logits = model(sample['image'])                    
                    loss = criterion(logits, sample['label_1']) * sample['lam'] + criterion(logits, sample['label_2'] * (1. - sample['lam']))
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            # measure evaluation metric and record loss
            top1_err, top5_err = eval_metric.compute(logits.data, labels, topk=(1, 5))

            loss_values.update(loss.item(), images.size(0))
            top1.update(top1_err.item(), images.size(0))
            top5.update(top5_err.item(), images.size(0))

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % log_intervals == 0:
                current_lr = scheduler.get_last_lr()[0]
                tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}][{i}/{len(train_loader)}] || '
                           f'LR : {current_lr:.5f} || '
                           f'Train Loss : {loss_values.val:.4f} ({loss_values.avg:.4f}) || '                        
                           f'Train Top 1-acc : {top1.val:.3f}% ({top1.avg:.3f})% || '
                           f'Train Top 5-acc : {top5.val:.3f}% ({top5.avg:.3f})%')

        loss_values = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        if not TRAIN_ONLY:
            with torch.no_grad():
                model.eval()

                loss_values = AverageMeter()
                top1 = AverageMeter()
                top5 = AverageMeter()

                for i, val_batch in enumerate(tqdm(val_loader, desc=f'Validation')):
                    sample = val_batch
                    images = sample['image'].float().to(device)
                    labels = sample['label'].long().to(device)

                    logits = model(images)
                    loss = criterion(logits, labels)
                    preds = torch.argmax(logits, -1)

                    top1_err, top5_err = eval_metric.compute(logits.data, labels, topk=(1, 5))
                    loss_values.update(loss.item(), images.size(0))
                    top1.update(top1_err.item(), images.size(0))
                    top5.update(top5_err.item(), images.size(0))

            tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}] || '
                       f'Val Loss : {loss_values.avg:.4f} || '                        
                       f'Val Top 1-acc : {top1.avg:.3f}% || '
                       f'Val Top 5-acc : {top5.avg:.3f}%')

            is_best = top1.avg >= best_acc
            best_acc = max(top1.avg, best_acc)

            if is_best:
                if k > 0:
                    os.makedirs(os.path.join(OUTPUT_DIR, MODEL_ARC, f'/{k}fold'), exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_ARC, f'/{k}_fold/{epoch + 1}_epoch_{best_acc:.2f}%_with_val.pth'))
                else:                    
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_ARC, f'{epoch + 1}_epoch_{best_acc:.2f}%_with_val.pth'))
        
        else:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_ARC, f'_{epoch + 1}_epoch_{top1.avg:.2f}%_only_train.pth'))


                
            




                


            


