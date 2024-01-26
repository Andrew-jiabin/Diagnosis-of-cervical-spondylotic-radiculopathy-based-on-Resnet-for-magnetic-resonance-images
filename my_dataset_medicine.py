import pickle
import random

import torch.nn.functional as F
import torch
from matplotlib.pyplot import pause
from torch.utils.data import Dataset
import numpy as np


class MyDataSet(Dataset):
    def __init__(self, mode, labelclass, transform=None, seed=0):
        self.seed = seed
        self.mode = mode
        self.labelclass = labelclass
        self.indexlist_val = []
        self.indexlist_train = []
        random.seed(self.seed)
        with open(
            r"./train_data_104_dict.pkl",
            "rb",
        ) as x:
            self.original_data = pickle.load(x)
        self.traindata, self.valdata = self.sample_with_remainder(
            self.original_data, 0.9
        )
        if self.mode == "train":
            self.data = self.traindata
            self.data_index = self.indexlist_train
            # print("self.data_index is : ", self.data_index)
        if self.mode == "val":
            self.data = self.valdata
            self.data_index = self.indexlist_val
        self.transform = (
            transform  # transform can be a function or method
        )

    def getlsit(self, sampled_indexes, flag):
        sampled_elements = []
        for i in sampled_indexes:
            temp = []
            for j in range(12):
                temp.append(self.original_data["HC" + str(i + 1)]["axial"][0][j])
            sampled_elements.append([temp, "HC"])

        for i in sampled_indexes:
            temp = []
            for j in range(12):
                temp.append(self.original_data["CSR" + str(i + 1)]["axial"][0][j])
            sampled_elements.append([temp, "CSR"])
        if flag == 1:
            self.indexlist_val = self.get_index(sampled_elements)
        if flag == 0:
            self.indexlist_train = self.get_index(sampled_elements)

        return sampled_elements  # Complete the image and label import


    def get_index(self, lists):
        indexs = []
        for i in range(len(lists)):
            indexs.append(lists[i][1])
        return indexs

    def keeplist(self, lists):  # Extract 24 arrays
        list = []
        for item in lists:
            list.append(item[0])

        return list

    def sample_with_remainder(self, input_dict: dict, sample_ratio):
        random.seed(self.seed)
        reminder_indexes = []
        total_elements = int(len(input_dict) / 2)
        sample_size = int(total_elements * sample_ratio)

        sampled_indexes = random.sample(range(total_elements), sample_size)

        for i in range(total_elements):
            if i not in sampled_indexes:
                reminder_indexes.append(i)
        sampled_elements = self.keeplist(
            self.getlsit(sampled_indexes, flag=0)
        )  # For the computation of the validation set
        reminded_list = self.keeplist(
            self.getlsit(reminder_indexes, flag=1)
        )  # For the computation of the test set
        # The getlist function gets the data and imports the labels
        print("sampled_indexes is :", sampled_indexes)
        return sampled_elements, reminded_list

    def padding(self, eachtensor):  # Each image will be padded at the end of the "__getitem__"
        padding_needed_h = max(0, 512 - eachtensor.shape[0])
        padding_needed_w = max(0, 512 - eachtensor.shape[1])
        padding_top = padding_needed_h // 2
        padding_bottom = padding_needed_h - padding_top
        padding_left = padding_needed_w // 2
        padding_right = padding_needed_w - padding_left
        eachtensor = F.pad(
            torch.tensor(eachtensor).clone().detach(),
            # torch.tensor(eachtensor),
            (padding_left, padding_right, padding_top, padding_bottom),
        )
        return eachtensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        tensors = []
        for eachtensor in self.data[item]:
            tensors.append(torch.tensor(self.padding(eachtensor)))
        images = torch.stack(tuple(tensors), dim=0)
        label = self.labelclass[self.data_index[item]]
        return images, label
    @staticmethod
    def collate_fn(batch):  # finish the construction of input of  the net
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
