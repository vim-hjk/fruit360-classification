import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations.pytorch.transforms import ToTensorV2

from src.models.timm import PretrainedModel
from src.utils.util import get_dataloader


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', type=str, default='./checkpoints/resnet18d/5_epoch_100.00%_with_val.pth')
parser.add_argument('--data_path', type=str, default='./data/train.csv')
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_arc', type=str, default='resnet18d')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

whole_df = pd.read_csv(args.data_path)

val_transform = A.Compose([
        A.Resize(100, 100),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ToTensorV2()
    ])

train_df, val_df = train_test_split(whole_df, test_size=args.test_size, random_state=args.seed)            
val_loader = get_dataloader(df=val_df, transform=val_transform, batch_size=1, shuffle=False)

model = PretrainedModel(model_arc=args.model_arc)
model.load_state_dict(torch.load(args.checkpoints, map_location=device))

final_conv = model.net.layer4[1]._modules.get('conv2')
# print(final_conv)
# print(model.net._modules.get('fc'))
fc_params = list(model.net._modules.get('fc').parameters())


class SaveFeatures():
    """ Extract pretrained activations"""
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()


def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv[0,:, :, ].reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img


def denormalize_image(image, mean=(0.5 ,0.5, 0.5), std=(0.25, 0.25, 0.25)):
    img_cp = image.copy()
    img_cp *= std
    img_cp += mean
    img_cp *= 255.0
    img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
    
    return img_cp


def plotGradCAM(model, final_conv, fc_params, val_loader, 
                row=1, col=8, img_size=256, device='cpu', original=False):
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    # save activated_features from conv
    activated_features = SaveFeatures(final_conv)
    # save weight from fc
    weight = np.squeeze(fc_params[0].cpu().data.numpy())
    # original images
    if original:
        fig = plt.figure(figsize=(60, 30))
        for i, sample in enumerate(val_loader):
            img = sample['image']
            target = sample['label']
            output = model(img.to(device))
            pred_idx = output.to('cpu').numpy().argmax(1)            
            cur_images = denormalize_image(img.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy())
            ax = fig.add_subplot(row, col, i + 1, xticks=[], yticks=[])
            plt.imshow(cur_images)
            ax.set_title('Label:%d, Predict:%d' % (target, pred_idx), fontsize=28)
            if i == row * col - 1:
                break
        plt.savefig('./debug_result/org.jpg')
    # heatmap images
    fig = plt.figure(figsize=(60, 30))
    for i, sample in enumerate(val_loader):
        img = sample['image']
        target = sample['label']
        output = model(img.to(device))
        pred_idx = output.to('cpu').numpy().argmax(1)
        cur_images = denormalize_image(img.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy())
        heatmap = getCAM(activated_features.features, weight, pred_idx)
        ax = fig.add_subplot(row, col, i+1, xticks=[], yticks=[])
        plt.imshow(cur_images)
        plt.imshow(cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR), alpha=0.4, cmap='jet')
        ax.set_title('Label:%d, Predict:%d' % (target, pred_idx), fontsize=28)
        if i == row * col - 1:
            break
    plt.savefig('./debug_result/grad_cam.jpg')


plotGradCAM(model, final_conv, fc_params, val_loader, img_size=100, device=device, original=True)