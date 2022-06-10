import albumentations as A
import cv2
import numpy as np
import pandas as pd 
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import sys
import timm
from tqdm import tqdm
import time
import copy




def data_loaders(data_dir, batch_size=64, train = False):
        """This functions load in data which is later fed into the model"""
        if train:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(240),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(), 
                                                            transforms.GaussianBlur(3)]), p=0.1),
  
                transforms.ToTensor(),
            ])
            train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform=transform)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            return train_loader, len(train_data)
        
        elif train == False:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(240),
                transforms.ToTensor(),
            ])
            val_data = datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform=transform)
            test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
            return (val_loader, test_loader, len(val_data), len(test_data))



def train(dataloaders,device,dataset_sizes,train_list, vali_list,model, criterion, optimizer, scheduler, num_epochs=10):
        """This function trains the model and output the best performing model"""
        # since = time.time()
        param = copy.deepcopy(model.state_dict())
        # best_acc = 0.0
        loss_values = []
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                cur_loss = 0.0
                corrects = 0
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    cur_loss += loss.item() * inputs.size(0)
                    
    #                 loss_values.append(running_loss/len())
                    corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()



        model.load_state_dict(param)
        return model,loss_values