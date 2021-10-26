#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: oliver@kuhles.net
# DATE CREATED: 25/Oct/2021
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Loading and preprocessing the image
#

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import PIL
from PIL import Image
import numpy as np
import time
from datetime import datetime
import sys

def load_checkpoint(file, gpu):
    '''
    Loads a model from a saved checkpoint. Moves it to the device (CPU or GPU).

    Returns the model along with other model parameters.
    '''
    if gpu == True and torch.cuda.is_available():
        checkpoint = torch.load(file, map_location=lambda storage, loc: storage.cuda(0))
    else:
        checkpoint = torch.load(file, map_location=lambda storage, loc: storage)

    model = getattr(models, checkpoint['arch'])(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    lr_scheduler = checkpoint['lr_scheduler']
    epochs = checkpoint['epochs']

    return model, optimizer, lr_scheduler, epochs


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    #imshow(img)
    # using thumbnail because it respects ratio
    img.thumbnail((255,255))

    # center crop
    crop_size = 224
    left  = int(img.size[0]/2-crop_size/2)
    upper = int(img.size[1]/2-crop_size/2)
    right = left+crop_size
    lower = upper+crop_size

    cropped_img = img.crop((left,upper,right,lower))

    # use std and mean for ImageNet, same as for normalization in transforms
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    norm_img = transform(cropped_img)

    return norm_img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Pre-process the image
    image = process_image(image_path)

    # Add a dimension for the batch
    image = torch.unsqueeze(image,0)

    # Move image and model to the correct device
    #image.to(device)
    model.to(device)

    with torch.no_grad():
        model.eval()
        ps = torch.exp(model.forward(image.to(device)))
        top_ps, top_class = ps.topk(topk, dim=1)
        model.train()

    return(top_ps, top_class)
