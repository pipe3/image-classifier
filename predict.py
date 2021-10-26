#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: oliver@kuhles.net
# DATE CREATED: 25/Oct/2021
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Image classifier prediction

# Imports
import matplotlib.pyplot as plt
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
import argparse
import model_functions, utility_functions
import sys

def main():

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('imagepath', help='path to single image')
    parser.add_argument('checkpoint', help='path to model checkpoint')
    parser.add_argument('--top_k', help='Number of top k most likely classes to predict', type=int, default=5)
    parser.add_argument('--category_names', help='path to file with category names')
    parser.add_argument('--gpu', help='Use GPU instead of CPU',  action="store_true", default=False)

    args = parser.parse_args()
    #print(args)

    image = args.imagepath
    checkpoint = args.checkpoint
    topk = args.top_k
    catfile = args.category_names
    gpu = args.gpu

    print('Image Classifier prediction starting up')

    # Doublecheck if we have a GPU if the user has selected to use GPU
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Model running on', device)
    elif gpu and not torch.cuda.is_available():
        device = torch.device('cpu')
        print('GPU selected but no GPU identified. Model running on', device)
    else:
        device = torch.device('cpu')
        print('Model running on', device)

    # Load checkpoint
    model, optimizer, lr_scheduler, epochs = utility_functions.load_checkpoint(checkpoint, gpu)
    #print(model.classifier)
    #model.to(device)

    #utility_functions.imshow(utility_functions.process_image(image))

    print('Predicting class')
    probs, labels = utility_functions.predict(image, model, topk, device)

    p_list = probs.squeeze().tolist()
    l_list = labels.squeeze().tolist()

    #print(model.class_to_idx)
    #class_list = utility_functions.get_category_names(catfile, model.class_to_idx, l_list, p_list)

    #plt.figure()
    #plt.barh(class_list, p_list)
    #plt.title('Probability Prediction')
    #plt.show()

    i = 0

    if catfile != None:
        with open(catfile, 'r') as f:
            cat_to_name = json.load(f)
        for l in l_list:
            classification = list(model.class_to_idx)[l]
            name = cat_to_name.get(str(classification))
            prob = p_list[i]
            print('Prob: {:.2f}% .. Class: {} .. Name: {}  '.format(prob*100, classification, name))
            i+=1
    else:
        for l in l_list:
            classification = list(model.class_to_idx)[l]
            prob = p_list[i]
            print('Prob: {:.2f}% .. Class: {} '.format(prob*100, classification))
            i+=1


if __name__ == '__main__':
    main()
