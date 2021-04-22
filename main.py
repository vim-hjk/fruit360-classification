import torch
import yaml
import argparse
import pandas as pd
import albumentations
import albumentations.pytorch

from easydict import EasyDict
from prettyprinter import cpprint
from sklearn.model_selection import StratifiedKFold, train_test_split

from dataset import AugMix
from utils import seed_everything, get_dataloader
from train import train

# Set Config
class YamlConfigManager:
    def __init__(self, config_file_path, config_name):
        super().__init__()
        self.values = EasyDict()        
        if config_file_path:
            self.config_file_path = config_file_path
            self.config_name = config_name
            self.reload()
    
    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()
    
    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                yaml.dump(dict(self.values), f)

def main(cfg):
    SEED = cfg.values.seed
    INPUT_DIR = cfg.values.input_dir
    USE_KFOLD_CV = cfg.values.val_args.use_kfold
    TRAIN_ONLY = cfg.values.train_only
    TRAIN_BATCH_SIZE = cfg.values.train_args.train_batch_size
    VAL_BATCH_SIZE = cfg.values.train_args.val_batch_size

    seed_everything(SEED)

    print(f'Cuda is Available ? : {torch.cuda.is_available()}\n')

    whole_df = pd.read_csv(INPUT_DIR)
    whole_label = whole_df['label'].values

    train_transform = albumentations.Compose([
        albumentations.OneOf([
            albumentations.HorizontalFlip(),
            albumentations.GaussNoise(),
            albumentations.ToGray(),
        ]), 
        albumentations.Resize(100, 100),
        albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        albumentations.pytorch.transforms.ToTensorV2()])

    val_transform = albumentations.Compose([
        albumentations.Resize(100, 100),
        albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        albumentations.pytorch.transforms.ToTensorV2()])

    if USE_KFOLD_CV:
        kfold = StratifiedKFold(n_splits=cfg.values.val_args.n_splits)
        k = 1

        for train_idx, val_idx in kfold.split(whole_df, whole_label):
            print('\n')
            cpprint('=' * 15 + f'{k}-Fold Cross Validation' + '=' * 15)
            train_df = whole_df.iloc[train_idx]
            val_df = whole_df.iloc[val_idx]

            train_loader = get_dataloader(df=train_df, transform=train_transform, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            val_loader = get_dataloader(df=val_df, transform=val_transform, batch_size=VAL_BATCH_SIZE, shuffle=False)

            train(cfg, k, train_loader, val_loader)

            k += 1

    else:
        cpprint('=' * 20 + f'START TRAINING' + '=' * 20)

        if not TRAIN_ONLY:
            train_df, val_df = train_test_split(whole_df, test_size=cfg.values.val_args.test_size, random_state=SEED)            
            train_loader = get_dataloader(df=train_df, transform=train_transform, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            val_loader = get_dataloader(df=val_df, transform=val_transform, batch_size=VAL_BATCH_SIZE, shuffle=False)

            train(cfg, 0, train_loader, val_loader)

        else:
            train_loader = get_dataloader(df=whole_df, transform=train_transform, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            train(cfg, 0, train_loader, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='base')
    
    args = parser.parse_args()
    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    main(cfg)