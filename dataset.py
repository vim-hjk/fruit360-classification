import os
import cv2
import numpy as np

from torch.utils.data import Dataset
from albumentations.core.transforms_interface import ImageOnlyTransform


class Fruit360Dataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df.reset_index()
        self.image_path = self.df.image_path
        self.labels = self.df.label
        self.transform = transform        
    
    def __len__(self):
        return len(self.df)

    def set_transform(self, transform):        
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return {'image' : image, 'label' : label}


class AugMix(ImageOnlyTransform):
    def __init__(self, width=2, depth=2, alpha=0.5, augmentations=[], always_apply=False, p=0.5):
        super(AugMix, self).__init__(always_apply, p)
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = augmentations
        self.ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        self.m = np.float32(np.random.beta(self.alpha, self.alpha))

    def apply_op(self, image, op):
        image = op(image=image)["image"]
        return image

    def apply(self, img, **params):
        mix = np.zeros_like(img)
        for i in range(self.width):
            image_aug = img.copy()

            for _ in range(self.depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(image_aug, op)

            mix = np.add(mix, self.ws[i] * image_aug, out=mix, casting="unsafe")

        mixed = (1 - self.m) * img + self.m * mix
        if img.dtype in ["uint8", "uint16", "uint32", "uint64"]:
            mixed = np.clip((mixed), 0, 255).astype(np.uint8)
        return mixed

    def get_transform_init_args_names(self):
        return ("width", "depth", "alpha")