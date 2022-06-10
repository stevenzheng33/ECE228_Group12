import os
import cv2
# import time
import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


import torchvision.transforms as T
# import torchvision.models as models
# from torchvision.utils import make_grid
# from torchvision.datasets import ImageFolder
#
# from matplotlib import pyplot as plt

# DIR_TRAIN = "../input/100-bird-species/train/"
# DIR_VALID = "../input/100-bird-species/valid/"
# DIR_TEST = "../input/100-bird-species/test/"
#
# classes = os.listdir(DIR_TRAIN)
# print("Total Classes: ",len(classes))

#Counting total train, valid & test images

# train_count = 0
# valid_count = 0
# test_count = 0
# for _class in classes:
#     train_count += len(os.listdir(DIR_TRAIN + _class))
#     valid_count += len(os.listdir(DIR_VALID + _class))
#     test_count += len(os.listdir(DIR_TEST + _class))
#
# print("Total train images: ",train_count)
# print("Total valid images: ",valid_count)
# print("Total test images: ",test_count)
#
# train_imgs = []
# valid_imgs = []
# test_imgs = []
#
# for _class in classes:
#
#     for img in os.listdir(DIR_TRAIN + _class):
#         train_imgs.append(DIR_TRAIN + _class + "/" + img)
#
#     for img in os.listdir(DIR_VALID + _class):
#         valid_imgs.append(DIR_VALID + _class + "/" + img)
#
#     for img in os.listdir(DIR_TEST + _class):
#         test_imgs.append(DIR_TEST + _class + "/" + img)
#
# class_to_int = {classes[i]: i for i in range(len(classes))}


def get_transform():
    return T.Compose([T.ToTensor()])


class BirdDataset(Dataset):

    def __init__(self, imgs_list, class_to_int, transforms=None):
        super().__init__()
        self.imgs_list = imgs_list
        self.class_to_int = class_to_int
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.imgs_list[index]

        # Reading image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # Retriving class label
        label = image_path.split("/")[-2]
        label = self.class_to_int[label]

        # Applying transforms on image
        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.imgs_list)


class BirdLoader(object):

    def __init__(self):
        self.train_path = "./data/archive/train/"
        self.valid_path = "./data/archive/valid/"
        self.test_path = "./data/archive/test/"
        self.train_count = 0
        self.valid_count = 0
        self.test_count = 0
        self.train_imgs = []
        self.valid_imgs = []
        self.test_imgs = []
        self.class_to_int = {}

    def _data_process(self):
        classes = os.listdir(self.train_path)

        for _class in classes:
            self.train_count += len(os.listdir(self.train_path + _class))
            self.valid_count += len(os.listdir(self.valid_path + _class))
            self.test_count += len(os.listdir(self.test_path + _class))

        for _class in classes:

            for img in os.listdir(self.train_path + _class):
                self.train_imgs.append(self.train_path + _class + "/" + img)

            for img in os.listdir(self.valid_path + _class):
                self.valid_imgs.append(self.valid_path + _class + "/" + img)

            for img in os.listdir(self.test_path + _class):
                self.test_imgs.append(self.test_path + _class + "/" + img)

        self.class_to_int = {classes[i]: i for i in range(len(classes))}

    def getData(self, mode="train"):
        # 先对数据就行预处理
        self._data_process()
        if mode == "trian":
            dataset = BirdDataset(self.train_imgs, self.class_to_int, get_transform())
        elif mode == "valid":
            dataset = BirdDataset(self.valid_imgs, self.class_to_int, get_transform())
        else:
            dataset = BirdDataset(self.test_imgs, self.class_to_int, get_transform())

        # random_sampler = RandomSampler(dataset)
        # data_loader = DataLoader(dataset=dataset, batch_size=8, sampler=random_sampler, num_workers=4)
        return dataset

# train_dataset = BirdDataset(train_imgs, class_to_int, get_transform())
# valid_dataset = BirdDataset(valid_imgs, class_to_int, get_transform())
# test_dataset = BirdDataset(test_imgs, class_to_int, get_transform())
#
# #Data Loader  -  using Sampler (YT Video)
# train_random_sampler = RandomSampler(train_dataset)
# valid_random_sampler = RandomSampler(valid_dataset)
# test_random_sampler = RandomSampler(test_dataset)
#
# #Shuffle Argument is mutually exclusive with Sampler!
# train_data_loader = DataLoader(
#     dataset = train_dataset,
#     batch_size = 16,
#     sampler = train_random_sampler,
#     num_workers = 4,
# )
#
# valid_data_loader = DataLoader(
#     dataset = valid_dataset,
#     batch_size = 16,
#     sampler = valid_random_sampler,
#     num_workers = 4,
# )
#
# test_data_loader = DataLoader(
#     dataset = test_dataset,
#     batch_size = 16,
#     sampler = test_random_sampler,
#     num_workers = 4,
# )