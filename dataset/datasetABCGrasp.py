import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image


class train_dataset(Dataset):

    def __init__(self, trainTrialsList, dataPath, device, transform=False):
        super(train_dataset, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        # agrees = [0, 90, 180, 270]
        # agrees = [0]
        idx = 0
        for clcPath in os.listdir(dataPath):
            datas[idx] = []
            for subClcPath in os.listdir(os.path.join(dataPath, clcPath)):
                t = os.listdir(os.path.join(dataPath, clcPath, subClcPath))
                for samplePath in os.listdir(os.path.join(dataPath, clcPath, subClcPath)):
                    name, ext = os.path.splitext(samplePath)
                    if ext != '.pickle':
                        continue
                    # if clcPath == '0':
                    #     agrees = range(0, 360, 30)
                    # else:
                    #     agrees = range(0, 360, 45)
                    # for agree in agrees:

                    filePath = os.path.join(dataPath, clcPath, subClcPath, samplePath)
                    datas[idx].append(filePath)
            idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def __len__(self):
        return 21000000

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        traj1 = None
        traj2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class test_dataset(Dataset):

    def __init__(self, testTrialsList, dataPath, device, transform=False, times=200, way=20, shuffle=False):
        np.random.seed(1)
        super(test_dataset, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)
        if shuffle:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
        else:
            self.img1 = None
            self.c1 = None

    # generate image pair from different class

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for clcPath in os.listdir(dataPath):
            datas[idx] = []
            for subClcPath in os.listdir(os.path.join(dataPath, clcPath)):
                for samplePath in os.listdir(os.path.join(dataPath, clcPath, subClcPath)):
                    filePath = os.path.join(dataPath, clcPath, subClcPath, samplePath)
                    datas[idx].append(filePath) # Yantian
                    # Add .rotate(0) to fix IOError: broken data stream when reading image file
                    # https://github.com/python-pillow/Pillow/issues/1510
                    # ... it lazy loads the data. Resizing or converting the format will force the data to load
            idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    # def __getitem__(self, index):
    #     idx = index % self.way
    #     label = None
    #     if idx == 0:
    #         self.c1 = random.randint(0, self.num_classes - 1)
    #         self.img1 = random.choice(self.datas[self.c1])
    #         # img2 = random.choice(self.datas[self.c1])
    #
    #     # Only need to generate image pair from different class
    #     c2 = random.randint(0, self.num_classes - 1)
    #     while self.c1 == c2:
    #         c2 = random.randint(0, self.num_classes - 1)
    #     img2 = random.choice(self.datas[c2])
    #
    #     if self.transform:
    #         img1 = self.transform(self.img1)
    #         img2 = self.transform(img2)
    #     return img1, img2

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


# test
if __name__=='__main__':
    train_dataset = train_dataset('./images_background', 30000*8)
    print(train_dataset)
