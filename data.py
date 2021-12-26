import torch
from torchvision import datasets
import torchvision.transforms as T

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import numpy as np

DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}

class ImageDataset(Dataset):

    def __init__(self, pt_dataset, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.perm = torch.arange(32*32) if perm is None else perm
        
        self.vocab_size = clusters.size(0)
        self.block_size = 32*32 - 1
        
    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = self.quantize(x, self.clusters)
        a = x.view(x.shape[0] * x.shape[1], -1)
        a = a.transpose(0, 1).contiguous()
        
        return a[0][:-1], a[0][1:] # always just predict the next one in the sequence

    def squared_euclidean_distance(self, a, b):
        b = torch.transpose(b, 0, 1)
        a2 = torch.sum(torch.square(a), dim=1, keepdims=True)
        b2 = torch.sum(torch.square(b), dim=0, keepdims=True)
        ab = torch.matmul(a, b)
        d = a2 - 2 * ab + b2
        return d

    def quantize(self, x, centroids):
        c, h, w = x.shape
        # [B, C, H, W] => [B, H, W, C]
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(-1, c)  # flatten to pixels
        d = self.squared_euclidean_distance(x, centroids)
        x = torch.argmin(d, 1)
        x = x.view(h, w)
        return x



def train_transforms(dataset):
    if dataset == "cifar10":
        # "When full-network fine-tuning on CIFAR-10 and CIFAR100, we use the augmentation popularized by Wide Residual
        # Networks: 4 pixels are reflection padded on each side, and
        # a 32 × 32 crop is randomly sampled from the padded image or its horizontal flip"
        return T.Compose(
            [
                T.RandomCrop(32, padding=4, padding_mode="reflect"),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )

    elif dataset == "mnist" or dataset == "fmnist":
        return T.ToTensor()


def test_transforms(dataset):
    return T.ToTensor()


def dataloaders(dataset, batch_size, datapath="data"):
    train_ds = DATASETS[dataset](
        datapath, train=True, download=True, transform=train_transforms(dataset)
    )
    valid_ds = DATASETS[dataset](
        datapath, train=True, download=True, transform=test_transforms(dataset)
    )
    test_ds = DATASETS[dataset](
        datapath, train=False, download=True, transform=test_transforms(dataset)
    )

    # TODO paper uses 90/10 split for every dataset besides ImageNet (96/4)
    train_size = int(0.9 * len(train_ds))

    # reproducable split
    # NOTE: splitting is done twice as datasets have different transforms attributes
    train_ds, _ = random_split(
        train_ds,
        [train_size, len(train_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )
    _, valid_ds = random_split(
        valid_ds,
        [train_size, len(valid_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )

    color_palette = np.load('kmeans_centers.npy')
    centroids = nn.Parameter(torch.from_numpy(color_palette), requires_grad=False)

    train_ds = ImageDataset(train_ds, centroids)
    valid_ds = ImageDataset(valid_ds, centroids)
    test_ds = ImageDataset(test_ds, centroids)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    return train_dl, valid_dl, test_dl
