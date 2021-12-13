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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

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

'''
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

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape) # b,c,h,w

    imshow(torchvision.utils.make_grid(images))
'''
