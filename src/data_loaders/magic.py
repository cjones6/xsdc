import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

from . import create_data_loaders


def get_dataloaders(batch_size=128, batch_size_labeled=None, batch_size_unlabeled=None, data_path='../data/magic',
                    frac_labeled=None, num_labeled=None, balanced=False, stratified_unlabeling=False,
                    stratified_sampling=False, num_workers=4):
    """
    Create data loaders for MAGIC.

    :param batch_size: Batch size of (labeled+unlabeled) training data and test data
    :param batch_size_labeled: Batch size for the labeled training data
    :param batch_size_unlabeled: Batch size for the unlabeled training data
    :param data_path: Directory where the data currently exists
    :param frac_labeled: Fraction of the training set that should be labeled. Only one of frac_labeled and num_labeled
                         can be specified.
    :param num_labeled: Number of images in the training set that should be labeled. Only one of frac_labeled and
                        num_labeled can be specified.
    :param balanced: Whether to make the training set balanced by removing some of the observations
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
    train_data = pd.read_csv(os.path.join(data_path, 'magic04.data'), header=None, skiprows=None, delimiter=',').values
    train_labels = train_data[:, -1]
    train_data = train_data[:, :-1].astype('float64')
    train_labels = np.array([1 if train_labels[i] == 'g' else 0 for i in range(len(train_labels))])
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.25,
                                                                        stratify=train_labels, random_state=0)

    if balanced:
        num_pos = sum(train_labels == 0)
        neg_labels = np.where(train_labels == 1)[0]
        keep_idxs = list(set(range(len(train_labels))).difference(set(neg_labels[num_pos:])))
        train_labels = train_labels[keep_idxs]
        train_data = train_data[keep_idxs]

        num_pos = sum(test_labels == 0)
        neg_labels = np.where(test_labels == 1)[0]
        keep_idxs = list(set(range(len(test_labels))).difference(set(neg_labels[num_pos:])))
        test_labels = test_labels[keep_idxs]
        test_data = test_data[keep_idxs]

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_dataset = create_data_loaders.PreloadedDataset(np.array(train_data), train_labels.astype(int))
    test_dataset = create_data_loaders.PreloadedDataset(np.array(test_data), test_labels.astype(int))

    return create_data_loaders.generate_dataloaders(train_dataset,
                                                    test_dataset,
                                                    valid_size=int(0.2*len(train_labels)),
                                                    separate_valid_set=False,
                                                    batch_size=batch_size,
                                                    batch_size_labeled=batch_size_labeled,
                                                    batch_size_unlabeled=batch_size_unlabeled,
                                                    frac_labeled=frac_labeled,
                                                    num_labeled=num_labeled,
                                                    stratified_unlabeling=stratified_unlabeling,
                                                    stratified_sampling=stratified_sampling,
                                                    num_workers=num_workers)
