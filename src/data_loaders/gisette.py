import numpy as np
import os
from sklearn.datasets import load_svmlight_file

from . import create_data_loaders


def get_dataloaders(batch_size=128, batch_size_labeled=None, batch_size_unlabeled=None,
                    data_path='../data/gisette_scale', frac_labeled=None, num_labeled=None, stratified_unlabeling=False,
                    stratified_sampling=False, num_workers=4):
    """
    Create data loaders for Gisette.

    :param batch_size: Batch size of (labeled+unlabeled) training data and test data
    :param batch_size_labeled: Batch size for the labeled training data
    :param batch_size_unlabeled: Batch size for the unlabeled training data
    :param data_path: Directory where the data currently exists
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
    train_data, train_labels = load_svmlight_file(os.path.join(data_path, 'gisette_scale'))
    test_data, test_labels = load_svmlight_file(os.path.join(data_path, 'gisette_scale.t'))

    train_labels = (train_labels.flatten() + 1)/2
    test_labels = (test_labels.flatten() + 1)/2

    train_dataset = create_data_loaders.PreloadedDataset(np.asarray(train_data.todense()), train_labels.astype(int))
    test_dataset = create_data_loaders.PreloadedDataset(np.asarray(test_data.todense()), test_labels.astype(int))

    return create_data_loaders.generate_dataloaders(train_dataset,
                                                    test_dataset,
                                                    valid_size=int(0.2 * len(train_labels)),
                                                    separate_valid_set=False,
                                                    batch_size=batch_size,
                                                    batch_size_labeled=batch_size_labeled,
                                                    batch_size_unlabeled=batch_size_unlabeled,
                                                    frac_labeled=frac_labeled,
                                                    num_labeled=num_labeled,
                                                    stratified_unlabeling=stratified_unlabeling,
                                                    stratified_sampling=stratified_sampling,
                                                    num_workers=num_workers)
