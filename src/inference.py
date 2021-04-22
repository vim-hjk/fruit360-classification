import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import albumentations
import albumentations.pytorch

from tqdm import tqdm
from torch.utils.data import DataLoader

from model import PretrainedModel
from dataset import Fruit360Dataset
from main import YamlConfigManager


parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default='./config.yml')
parser.add_argument('--config', type=str, default='base')

args = parser.parse_args()
cfg = YamlConfigManager(args.config_file_path, args.config)

info_df = pd.read_csv('./prediction/info.csv')

test_transform = albumentations.Compose([            
                 albumentations.Resize(100, 100),
                 albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
                 albumentations.pytorch.transforms.ToTensorV2()
])

test_data = Fruit360Dataset(df=info_df, transform=test_transform)

test_loader = DataLoader(
    test_data,
    batch_size=cfg.values.train_args.val_batch_size,
    shuffle=False
)

device = torch.device('cuda:0')

ckpt_path = 'C:/Users/KHJ/workspace/fruit_recognition/results/resnet18d/5_epoch_100.00%_with_val.pth'
model = PretrainedModel(cfg.values.model_arc, num_classes=cfg.values.num_classes)
model.load_state_dict(torch.load(ckpt_path))
model.to(device)
model.eval()

preds = []
with torch.no_grad():
    for sample in tqdm(test_loader):
        images = sample['image'].float().to(device)
        logits = model(images)
        pred = logits.argmax(-1)
        preds.extend(pred.cpu().numpy())

info_df['label'] = preds
info_df.to_csv('./prediction/submission.csv', index=False)
print(f'Inference Done!')



        
