#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: oliver@kuhles.net
# DATE CREATED: 15/Oct/2021
# REVISED DATE: 25/Oct/2021            <=(Date Revised - if any)
# PURPOSE: Image classifier trainer

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
    parser.add_argument('data_directory', help='path to training data')
    parser.add_argument('--arch', help='ImageNet model pretrained', default='vgg16_bn' )
    parser.add_argument('--hidden_units', help='hidden units', type=int, default=4096)
    parser.add_argument('--output_units', help='output units', type=int, default=102)
    parser.add_argument('--epochs', help='numbe of epochs',  type=int, default=2)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('--dropout', help='dropout', type=float, default=0.5)
    parser.add_argument('--lr_decay', help='Learning rate decay', default=0.1)
    parser.add_argument('--gpu', help='Use GPU instead of CPU',  action="store_true", default=False)
    parser.add_argument('--save_dir', help='path to store the trained model', default='.')
    parser.add_argument('--print_every', help='print validation every number', type=int, default=20)

    args = parser.parse_args()
    #print(args)

    data_directory = args.data_directory
    arch = args.arch
    hidden_units = args.hidden_units
    output_units = args.output_units
    epochs = args.epochs
    learningrate = args.learning_rate
    lr_decay = args.lr_decay
    dropout = args.dropout
    filepath = args.save_dir
    print_every = args.print_every
    gpu = args.gpu

    print('Image Classifier training starting up')

    # Load the predefined model
    print('Loading the predefined model')
    model = model_functions.load_predefined_model(arch)
    #print('Standard classifier')
    #print(model.classifier)
    model = model_functions.replace_classifier(model, hidden_units, output_units, dropout)
    #print('Replaced classifier')
    #print(model.classifier)

    # lets define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learningrate)

    # introducing learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    # Doublecheck if we have a GPU if the user has selected to use GPU
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        print('Model running on', device)
    elif gpu and not torch.cuda.is_available():
        device = torch.device('cpu')
        model.to(device)
        print('GPU selected but no GPU identified. Model running on', device)
    else:
        device = torch.device('cpu')
        model.to(device)
        print('Model running on', device)

    # Get the data
    trainloader, validloader, testloader, class_to_idx = model_functions.load_data(str(data_directory))


    # some vars not affecting the hyperparameters of the model
    t_losses, v_losses, t_accs, v_accs = [], [], [], []

    starttime = time.time()
    print('Classifier training startup at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    for e in range(epochs):
        print('Starting epoch {}/{}'.format(e+1, epochs), 'with learningrate', lr_scheduler.get_lr()[0])
        # train epoch
        ttime, tloss, tacc = model_functions.train(print_every, model, optimizer, criterion, device, trainloader, validloader)
        # validate epoch
        vtime, vloss, vacc = model_functions.validate(model, criterion, device, validloader)
        # decrease lr
        lr_scheduler.step()

        print(f"Completed epoch {e+1}/{epochs}.. "
            f"tloss {tloss:.3f}.. "
            f"tacc {tacc:.3f}.. "
            f"ttime {ttime:.0f} sec.. "
            f"vloss {vloss:.3f}.. "
            f"vacc {vacc:.3f}.. "
            f"vtime {vtime:.0f} sec.. ")

        # keep data for plotting later
        t_losses.append(tloss)
        v_losses.append(vloss)
        t_accs.append(tacc)
        v_accs.append(vacc)

    elapsedtime = time.time() - starttime
    print(f"Classifier training completed at "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"in {elapsedtime:.0f} sec")

    # Save checkpoint
    model_functions.save_model(filepath, arch, model, optimizer, lr_scheduler, epochs, learningrate, class_to_idx)

if __name__ == '__main__':
    main()
