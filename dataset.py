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

import requests
import tqdm

### Wide Residual Networks: 4 pixels are reflection padded on each side, and
### a 32 × 32 crop is randomly sampled from the padded image or its horizontal flip

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_cifar(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,transform=transform)
    ##### USE 20% of data for test
    indices = np.arange(len(trainset)*0.2)
    trainset = Subset(trainset, indices)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True, sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape) # b,c,h,w

    imshow(torchvision.utils.make_grid(images))

    return trainloader, testloader


##### test #####
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    

    args = parser.parse_args()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
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

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape) # b,c,h,w

    imshow(torchvision.utils.make_grid(images))

