# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from itertools import permutations

# visualization
import time


class DataSet(torch.utils.data.Dataset):
    """ Dataset for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 #label_path,
                 num_video,
                 num_select_frames,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        #self.label_path = label_path
        self.num_video = num_video
        self.num_select_frames = num_select_frames
        self.permutations = list(permutations(list(range(self.num_video))))
        random.shuffle(self.permutations)
        self.permutations = self.permutations[:100]

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C T V M

        # load label
        #with open(self.label_path, 'rb') as f:
        #   self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            #self.label = self.label[0:100]
            self.data = self.data[0:100]
            #self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        #label = self.label[index]
        interval = (data_numpy.shape[1] - self.num_select_frames) // self.num_video
        selects = np.array([list(range(i * interval, i * interval + self.num_select_frames)) for i in range(self.num_video)])
        label = random.randint(0, 99)
        selects = selects[list(self.permutations[label])]
        data_list = []
        for select in selects:
            data_list.append(data_numpy[:, select, :, :])
        return np.array(data_list), label
