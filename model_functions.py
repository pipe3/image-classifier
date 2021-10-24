# functions and classes related to the model
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
import sys

def save_model(filepath, arch, model, optimizer, lr_scheduler, epochs, learningrate, class_to_idx):

    print('Saving model to', filepath+'/'+arch+'_'+'checkpoint.pth')
    checkpoint = {'arch': arch,
                  'classifier': model.classifier,
                  'optimizer': optimizer,
                  'lr_scheduler': lr_scheduler,
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'learningrate': learningrate,
                  'class_to_idx': class_to_idx
                 }

    try:
        torch.save(checkpoint, filepath+'/'+checkpoint['arch']+'_'+'checkpoint.pth')
        print('Model successfully saved')
    except e:
        print('An error occurred while saving the model: ', e)

def load_predefined_model(arch):
    accepted_archs=['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
    if arch in accepted_archs:
        model = getattr(models, arch)(pretrained=True)
        # freeze model parameters
        for parms in model.parameters():
            parms.requires_grad = False
    else:
        print('Error: Model', arch, 'is not in list of allowed models. Please choose from the following:')
        for model in accepted_archs:
            print(model)
        sys.exit()
    return model

def replace_classifier(model, hidden_units, output_units, dropout):
    simpleclassifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units, output_units)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    # replace it in the model
    model.classifier = simpleclassifier
    return model


def load_data(data_dir='flowers'):
    '''
    Loads data from a give data directory. Expects the data to be available in
    three sub directories: /train, /valid and /test
    then tranforms them and loads them as datasets.

    Loads the cat_to_name file from json into a dict

    Returns the datasets and the cat_to_name
    '''
    # data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    train_transforms = transforms.Compose([transforms.Resize(224),
                                         transforms.RandomCrop(224),
                                         transforms.RandomRotation(45),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    valid_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


    return trainloader, validloader, testloader, train_data.class_to_idx

def get_cat_to_name(filename='cat_to_name.json'):
    # cat_to_name
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

#
# Train
#
def train(validate_every, model, optimizer, criterion, device, trainloader, validloader):
    tloss = 0
    starttime = time.time()
    steps = 0
    tacc = 0

    for images, labels in trainloader:
        steps += 1

        # move the data tensors to the correct device
        images, labels = images.to(device), labels.to(device)

        # let the model determine our probabilities
        logps = model.forward(images)

        # calculate loss
        loss = criterion(logps, labels)

        # zero out the gradients
        optimizer.zero_grad()

        # backward step to calculate the gradients
        loss.backward()

        # forward step to optimize the model
        optimizer.step()

        # sum up the loss
        tloss += loss.item()

        # calculate accuracy
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equals = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        tacc += torch.mean(equals).item()

        # fake vars for testing the loop
        #tloss = 3
        #tacc = 4

        if validate_every != 0 and steps % validate_every == 0:
            #print('Validating batch {}/{}'.format(steps, len(trainloader)))
            vtime, vloss, vacc = validate(model, criterion, device, validloader)
            print(f"Validating batch {steps}/{len(trainloader)}.. "
                  f"tloss {tloss/steps:.3f}.. "
                  f"tacc {tacc/steps:.3f}.. "
                  f"vloss {vloss:.3f}.. "
                  f"vacc {vacc:.3f}.. "
                  f"t {vtime:.0f} sec")

    elapsedtime = time.time() - starttime

    return elapsedtime, tloss/len(trainloader), tacc/len(trainloader)

#
# Validate
#
def validate(model, criterion, device, validloader):
    vloss = 0
    vacc = 0
    starttime = time.time()

    # disable dropout
    model.eval()

    with torch.no_grad():
        vsteps = 0

        for images, labels in validloader:
            vsteps +=1
            #print('Beginning validloader step {}/{}'.format(vsteps, len(validloader)))

            # make sure it runs on the correct device
            images, labels = images.to(device), labels.to(device)

            # get the logps from model
            logps = model.forward(images)

            # calculate loss and sum up
            batch_loss = criterion(logps, labels)
            vloss += batch_loss.item()

            # calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            vacc += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()
    elapsedtime = time.time() - starttime

    # fake vars for testing the loop
    #vloss = 8
    #vaccuracy = 80

    return elapsedtime, vloss/len(validloader), vacc/len(validloader)
