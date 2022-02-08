import os

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Gisette(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Gisette, self).__init__()
        if train:
            dataset_path = os.path.join(root, 'gisette_scale')
        else:
            dataset_path = os.path.join(root, 'gisette_scale.t')

        data, labels = load_svmlight_file(dataset_path)
        self.data = torch.tensor(np.asarray(data.todense())).type(torch.FloatTensor)
        self.targets = torch.tensor((labels.flatten() + 1)/2).long()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        point, label = self.data[index], self.targets[index]

        if self.transform:
            point = self.transform(point)

        if self.target_transform:
            label = self.target_transform(label)

        return point, label

    def __len__(self):
        return len(self.data)


class Magic(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Magic, self).__init__()
        data = pd.read_csv(os.path.join(root, 'magic04.data'), header=None, skiprows=None, delimiter=',').values
        labels = data[:, -1]
        data = data[:, :-1].astype('float32')
        labels = np.array([1 if labels[i] == 'g' else 0 for i in range(len(labels))])
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25,
                                                                            stratify=labels, random_state=0)
        if train:
            self.data, self.targets = torch.from_numpy(train_data), torch.from_numpy(train_labels)
        else:
            self.data, self.targets = torch.from_numpy(test_data), torch.from_numpy(test_labels)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        point, label = self.data[index], self.targets[index]

        if self.transform:
            point = self.transform(point)

        if self.target_transform:
            label = self.target_transform(label)

        return point, label

    def __len__(self):
        return len(self.data)