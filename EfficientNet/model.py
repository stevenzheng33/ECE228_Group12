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

# import sys
# sys.path.insert(0, 'helper.py')

from functions import *
from config import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':    
    data_path = './data'
    class_dict = pd.read_csv(os.path.join(data_path, "class_dict.csv"))
    classes = list(class_dict['class'])

    (train_loader, train_data_len) = data_loaders(data_path, batch_s, train=True)
    (val_loader, test_loader, valid_data_len, test_data_len) = data_loaders(data_path, size_b, train=False)

    dataloaders = {
        "train":train_loader,
        "val": val_loader
    }
    dataset_sizes = {
        "train":train_data_len,
        "val": valid_data_len
    }

    
    model = models.efficientnet_b2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs,2048),
        nn.SiLU(),
        nn.Linear(2048, len(classes))
    )

    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)

    train_list = {'accuracy':[],'loss':[]}
    valid_list = {'accuracy':[],'loss':[]}

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size_config, gamma=gamma_config)
    model,loss_values = train(dataloaders,device,dataset_sizes,train_list, valid_list,model, loss, optimizer, scheduler,num_epochs=number_epochs)