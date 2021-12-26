import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import argparse
import yaml
from torchvision import datasets

from image_gpt import ImageGPT
from data import dataloaders
import numpy as np

import random
from torchmetrics.image.fid import FID
from torch.nn import functional as F
import torch
import torchvision.transforms as T
import torch.nn as nn
from data import ImageDataset
import cv2
import glob

def train(args):

    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    # experiment name
    #name = f"{config['name']}_{args.dataset}"
    name = "loss_with_self_log"

    if args.pretrained is not None:
        model = ImageGPT.load_from_checkpoint(args.pretrained)
        # potentially modify model for finetuning
        model.learning_rate = config["learning_rate"]
        model.classify = config["classify"]
    else:
        model = ImageGPT(centroids=args.centroids, **config)

    train_dl, valid_dl, test_dl = dataloaders(args.dataset, config["batch_size"])
    logger = pl_loggers.TensorBoardLogger("logs", name=name)

    # pretraining
    checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(
        max_steps=config["steps"],
        gpus=config["gpus"],
        precision=config["precision"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        checkpoint_callback=checkpoint,
        logger=logger,
    )

    trainer.fit(model, train_dl, valid_dl)

def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    # block_size = model.get_block_size()
    block_size = 512
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond, x_cond)
        print(logits.shape)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def fid_calculate(args):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    trainer = pl.Trainer(gpus=config["gpus"])
    model = ImageGPT.load_from_checkpoint(args.checkpoint).to('cuda')

    train_ds = datasets.CIFAR10(
        'data', train=True, download=True, transform=T.ToTensor()
    )

    randomlist = random.sample(range(0, len(train_ds)), 5000)
    train_ds = torch.utils.data.Subset(train_ds, randomlist)

    images_list = list()
    n_samples = 10
    i = 0
    for image, label in train_ds:
        if i<5000:
            # print(i, image.shape)
            # resize with nearest neighbor interpolation
            # store
            images_list.append((image*255).numpy().astype(np.uint8))
            i+=1
        else:
            break

    images_list = torch.Tensor(images_list).type(torch.uint8)
    fid = FID(feature=64)

    if config['image_create']:
        color_palette = np.load('kmeans_centers.npy')
        centroids = nn.Parameter(torch.from_numpy(color_palette), requires_grad=False)
        
        train_dataset = ImageDataset(train_ds, centroids)
        counts = torch.ones(512) # start counts as 1 not zero, this is called "smoothing"
        rp = torch.randperm(len(train_dataset))
        nest = 5000 # how many images to use for the estimation
        for i in range(nest):
            a, _ = train_dataset[int(rp[i])]
            t = a[0].item() # index of first token in the sequence
            counts[t] += 1
        prob = counts/counts.sum()

        for k in range(50):
            start_pixel = np.random.choice(np.arange(centroids.size(0)), size=(n_samples, 1), replace=True, p=prob)
            start_pixel = torch.from_numpy(start_pixel).to('cuda')
            
            pixels = sample(model, start_pixel, 32*32-1, temperature=1.0, sample=True, top_k=100)

            iperm = torch.argsort(train_dataset.perm)

            for i, pixel in enumerate(pixels):
                pxi = pixels[i][iperm] # note: undo the encoding permutation

                img = centroids[pxi].view(32,32,3)
                npimg = img.numpy()
                npimg = (npimg*255).astype(np.uint8)
                cv2.imwrite("./img/"+str(k)+"_"+str(i)+".png", npimg)


    images_list_fake = []
    img_list_fake_list = glob.glob('./img/*.png')
    
    for img_file in img_list_fake_list:
        img = cv2.imread(img_list_fake_list[0]).transpose(2,0,1)
        images_list_fake.append(img)

    images_list_fake = torch.Tensor(images_list_fake).type(torch.uint8)

    fid.update(images_list, real=True)
    fid.update(images_list_fake, real=False)
    print(fid.compute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cifar10")

    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.add_argument("config", type=str)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("checkpoint", type=str)
    parser_test.add_argument("config", type=str)
    parser_test.set_defaults(func=fid_calculate)

    args = parser.parse_args()
    args.centroids = f"kmeans_centers.npy"

    args.func(args)
