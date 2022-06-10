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


network = models.efficientnet_b1(pretrained=True)
learning_rate = 0.0005
step_size_config = 2
gamma_config = 0.5
number_epochs = 10
batch_s= 256
size_b = 64