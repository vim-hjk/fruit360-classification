import os
import torch
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prettyprinter import cpprint
from dataset import Fruit360Dataset


def dataset_unittest(n):
    # Dataset Unit Test
    label_list = os.listdir('./fruits-360/Training')

    whole_df = pd.read_csv('./data/train.csv')
    data = Fruit360Dataset(df=whole_df)
    print(data.__len__())
    sample = data.__getitem__(n)

    image = sample['image']
    label = sample['label']

    print('\n')
    cpprint(f'Label : {label_list[label]}')
    plt.imshow(image)
    plt.show()


def main(args):
    if args.dataset_unittest:
        index = random.randint(0, 67692)
        print(index)
        dataset_unittest(index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_unittest', type=str, default=True)
    
    args = parser.parse_args()    
    main(args)