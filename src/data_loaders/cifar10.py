import glob
import numpy as np
import os
import sys
import torch
import torchvision
from torchvision import transforms

from . import create_data_loaders


def get_dataloaders(valid_size=10000, batch_size=128, batch_size_labeled=None, batch_size_unlabeled=None,
                    transform='std', data_path='../data/cifar10', prewhitened=False, frac_labeled=None,
                    num_labeled=None, stratified_unlabeling=False, stratified_sampling=False, num_workers=0):
    """
    Create data loaders for CIFAR-10.

    :param valid_size: Size of the validation set
    :param batch_size: Batch size of (labeled+unlabeled) training data and test data
    :param batch_size_labeled: Batch size for the labeled training data
    :param batch_size_unlabeled: Batch size for the unlabeled training data
    :param transform: How the raw data should be transformed. Either 'std' (standardized) or 'grad' (gradient map on the
                      green channel) or None.
    :param data_path: Directory where the data either currently exists or should be downloaded to
    :param prewhitened: Whether to use prewhitened 3x3 patches
    :param frac_labeled: Fraction of the training set that should be labeled. Only one of frac_labeled and num_labeled
                         can be specified.
    :param num_labeled: Number of images in the training set that should be labeled. Only one of frac_labeled and
                        num_labeled can be specified.
    :param stratified_unlabeling: Whether when removing labels to create the unlabeled training set this should be done
                                  in a stratified manner or uniformly at random
    :param stratified_sampling: Whether the labeled training data should be sampled in a stratified manner, with the
                                same number of examples from each class. This is also used to determine whether a
                                stratified dataloader for the labeled+unlabeled training data should be returned
    :param num_workers: Number of workers to use for the dataloaders
    :return: Tuple containing:

            * train_loader: Dataloader for the (labeled+unlabeled) training data
            * train_stratified_loader: If stratified_sampling=True, this is a dataloader for the (labeled+unlabeled)
                                       training data in which each batch has equal representation from each class. Else,
                                       this is None.
            * train_labeled_loader: Dataloader for the labeled training data. If stratified_sampling=True, each batch
                                    has equal representation from each class.
            * train_unlabeled_loader: Dataloader for the unlabeled training data
            * valid_loader: Dataloader for the validation set
            * train_valid_loader: Dataloader for the combined training and validation sets
            * test_loader: Dataloader for the test set
    """
    def compute_gradient(x):
        img = np.array(x)/255.0
        img = img[:, :, 1]
        return np.array(np.gradient(img))

    def get_tensor(x):
        return torch.Tensor(x)

    if transform == 'std':
        mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
    elif transform == 'grad':
        transform = transforms.Compose([
            lambda x: compute_gradient(x),
            lambda x: get_tensor(x)])
    elif transform is not None:
        raise NotImplementedError

    if prewhitened is False:
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

        try:
            train_dataset = create_data_loaders.PreloadedDataset(train_dataset.data,
                                                                 np.array(train_dataset.targets), transform=transform)
            test_dataset = create_data_loaders.PreloadedDataset(test_dataset.data,
                                                                np.array(test_dataset.targets), transform=transform)
        except:
            train_dataset = create_data_loaders.PreloadedDataset(train_dataset.train_data,
                                                                 np.array(train_dataset.train_labels),
                                                                 transform=transform)
            test_dataset = create_data_loaders.PreloadedDataset(test_dataset.test_data,
                                                                np.array(test_dataset.test_labels),
                                                                transform=transform)
        return create_data_loaders.generate_dataloaders(train_dataset,
                                                        test_dataset,
                                                        separate_valid_set=False,
                                                        valid_size=valid_size,
                                                        batch_size=batch_size,
                                                        batch_size_labeled=batch_size_labeled,
                                                        batch_size_unlabeled=batch_size_unlabeled,
                                                        frac_labeled=frac_labeled,
                                                        num_labeled=num_labeled,
                                                        stratified_unlabeling=stratified_unlabeling,
                                                        stratified_sampling=stratified_sampling,
                                                        num_workers=num_workers)

    else:
        print('Loading pre-computed patches into memory... ', end='')
        python_version = sys.version.split(' ')[0]
        dataset = {'train_patches': None, 'train_labels': None, 'test_patches': None, 'test_labels': None}
        for dataset_name in dataset.keys():
            dataset_files = sorted(glob.glob1(data_path, dataset_name + '*' + python_version + '*'))
            data = []
            for filename in dataset_files:
                subset = np.load(os.path.join(data_path, filename))
                if 'patches' in filename:
                    subset = subset.float()
                data.append(subset)
            dataset[dataset_name] = torch.cat(data)
        print('done')

        train_dataset = create_data_loaders.PreloadedDataset(dataset['train_patches'], dataset['train_labels'].numpy())
        valid_dataset = create_data_loaders.PreloadedDataset(dataset['train_patches'], dataset['train_labels'].numpy())
        test_dataset = create_data_loaders.PreloadedDataset(dataset['test_patches'], dataset['test_labels'].numpy())

        return create_data_loaders.generate_dataloaders(train_dataset,
                                                        test_dataset,
                                                        valid_dataset=valid_dataset,
                                                        separate_valid_set=False,
                                                        valid_size=valid_size,
                                                        batch_size=batch_size,
                                                        batch_size_labeled=batch_size_labeled,
                                                        batch_size_unlabeled=batch_size_unlabeled,
                                                        frac_labeled=frac_labeled,
                                                        num_labeled=num_labeled,
                                                        stratified_unlabeling=stratified_unlabeling,
                                                        stratified_sampling=stratified_sampling,
                                                        num_workers=num_workers)
