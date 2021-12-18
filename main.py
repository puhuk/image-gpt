import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import os
import argparse
import dataset

import requests
from tqdm import tqdm

from model import GPT2

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--palette", action='store_true')

    args = parser.parse_args()

    if args.palette:
        url = "https://openaipublic.blob.core.windows.net/image-gpt/color-clusters/kmeans_centers.npy"
        filename = url.split("/")[-1]

        r = requests.get(url, stream=True)
        with open(f"{filename}", "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=80, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

    color_palette = np.load('/root/downloads/kmeans_centers.npy')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader = dataset.load_cifar(batch_size=args.batch_size, palette=color_palette)

    model = GPT2()
    

