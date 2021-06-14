import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import albumentations
import albumentations.pytorch

from tqdm import tqdm
from easydict import EasyDict
from torch.utils.data import DataLoader

from model import PretrainedModel
from dataset import Fruit360Dataset
from main import YamlConfigManager


parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default='.configs/inference_config.yml')
parser.add_argument('--config', type=str, default='base')

args = parser.parse_args()
cfg = YamlConfigManager(args.config_file_path, args.config)

INFO_PATH = cfg.values.info_path
SAVE_PATH = cfg.values.save_path

info_df = pd.read_csv('./prediction/info.csv')

test_transform = albumentations.Compose([            
                 albumentations.Resize(100, 100),
                 albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
                 albumentations.pytorch.transforms.ToTensorV2()
])

test_data = Fruit360Dataset(df=info_df, transform=test_transform)

test_loader = DataLoader(
    test_data,
    batch_size=cfg.values.batch_size,
    shuffle=False
)

device = torch.device('cuda:0')

models = [PretrainedModel(model_arc, num_classes=cfg.values.num_classes) for model_arc in cfg.values.model_arcs]
weights = cfg.values.weights

preds = []

for i, (model, weight) in enumerate(zip(models, weights)):
    model.load_state_dict(torch.load(weight))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for j, sample in enumerate(tqdm(test_loader)):
            images = sample['image'].float().to(device)
            logits = model(images)
            if j == 0:
                temp = logits
            else:
                temp = torch.cat((temp, logits), dim=0)
    if i == 0:
        preds = temp.reshape(1, -1, cfg.values.num_classes)
    else:
        preds = torch.cat((preds, temp.reshape(1, -1, cfg.values.num_classes)), 0)

pred = torch.mean(preds, 0).argmax(-1).flatten().cpu().numpy()

info_df['label'] = pred
info_df.to_csv(f'./prediction/{args.config}_submission.csv', index=False)
print(f'Inference Done!')