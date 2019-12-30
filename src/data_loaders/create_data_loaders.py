import torch.utils.data
import random
import torch

from src.data_loaders.data_loader_utils import remove_labels, StratifiedSampler
from src import default_params as defaults


def generate_dataloaders(train_dataset, test_dataset, valid_dataset=None, separate_valid_set=False, valid_size=0,
                         batch_size=128, batch_size_labeled=None, batch_size_unlabeled=None, frac_labeled=None,
                         num_labeled=None, stratified_unlabeling=False, stratified_sampling=False, num_workers=0):
    """
    Create data loaders given the corresponding PyTorch Dataset objects.

    :param train_dataset: Training dataset
    :param test_dataset: Test dataset
    :param valid_dataset: Validation dataset
    :param separate_valid_set: Whether to use a separate validation set or to use part of the training dataset
    :param valid_size: Validation set size
    :param batch_size: Batch size of (labeled+unlabeled) training data and test data
    :param batch_size_labeled: Batch size for the labeled training data
    :param batch_size_unlabeled: Batch size for the unlabeled training data
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
            * train_valid_loader: Dataloader for the combined training and validation sets if there is not a separate
                                  validation set. Else, None.
            * test_loader: Dataloader for the test set
    """
    if batch_size_labeled is None:
        batch_size_labeled = batch_size
    if batch_size_unlabeled is None:
        batch_size_unlabeled = batch_size

    if separate_valid_set:
        if frac_labeled is not None or num_labeled is not None:
            train_dataset, train_labeled_dataset, train_unlabeled_dataset = remove_labels(dataset=train_dataset,
                                                                              frac_labeled=frac_labeled,
                                                                              num_labeled=num_labeled,
                                                                              stratified=stratified_unlabeling)
            if stratified_sampling:
                train_labeled_sampler = StratifiedSampler(train_labeled_dataset, batch_size=batch_size)

        if stratified_sampling:
            train_sampler = StratifiedSampler(train_dataset, batch_size=batch_size)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout)
        valid_dataset.labels = valid_dataset.true_labels.copy()
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout)
        if stratified_sampling:
            train_stratified_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                              sampler=train_sampler, num_workers=num_workers,
                                                              pin_memory=True, drop_last=False,
                                                              timeout=defaults.dataloader_timeout)
            if frac_labeled is not None or num_labeled is not None:
                train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size_labeled,
                                                                 sampler=train_labeled_sampler, num_workers=num_workers,
                                                                 pin_memory=True, drop_last=False,
                                                                 timeout=defaults.dataloader_timeout)
                if len(train_unlabeled_dataset) > 0:
                    train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                         batch_size=batch_size_unlabeled,
                                                                         shuffle=False, num_workers=num_workers,
                                                                         pin_memory=True, drop_last=False,
                                                                         timeout=defaults.dataloader_timeout)
                else:
                    train_unlabeled_loader = None
            else:
                train_labeled_loader = train_unlabeled_loader = None
        else:
            train_stratified_loader = None
            if frac_labeled is not None or num_labeled is not None:
                train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size_labeled,
                                                                   shuffle=True, num_workers=num_workers,
                                                                   pin_memory=True, drop_last=False,
                                                                   timeout=defaults.dataloader_timeout)
                if len(train_unlabeled_dataset) > 0:
                    train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                         batch_size=batch_size_unlabeled,
                                                                         shuffle=True, num_workers=num_workers,
                                                                         pin_memory=True, drop_last=False,
                                                                         timeout=defaults.dataloader_timeout)
                else:
                    train_unlabeled_loader = None
            else:
                train_labeled_loader = train_unlabeled_loader = None

    elif valid_size > 0:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        train_novalid_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[valid_size:])
        valid_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[:valid_size])

        if frac_labeled is not None or num_labeled is not None:
            train_novalid_dataset, train_labeled_dataset, train_unlabeled_dataset = remove_labels(
                                                                                      dataset=train_novalid_dataset,
                                                                                      frac_labeled=frac_labeled,
                                                                                      num_labeled=num_labeled,
                                                                                      stratified=stratified_unlabeling)
            if stratified_sampling:
                train_labeled_sampler = StratifiedSampler(train_labeled_dataset, batch_size=batch_size)

        train_loader = torch.utils.data.DataLoader(train_novalid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout)

        if stratified_sampling:
            train_sampler = StratifiedSampler(train_dataset, batch_size=batch_size)
            train_novalid_sampler = StratifiedSampler(train_novalid_dataset, batch_size=batch_size)
            train_stratified_loader = torch.utils.data.DataLoader(train_novalid_dataset, batch_size=batch_size,
                                                                 sampler=train_novalid_sampler, num_workers=num_workers,
                                                                 pin_memory=True, drop_last=False,
                                                                 timeout=defaults.dataloader_timeout)
        else:
            train_stratified_loader = None
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout)
        if stratified_sampling:
            train_valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                             sampler=train_sampler, num_workers=num_workers,
                                                             pin_memory=True, drop_last=True,
                                                             timeout=defaults.dataloader_timeout)
            if frac_labeled is not None or num_labeled is not None:
                train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size_labeled,
                                                                   sampler=train_labeled_sampler,
                                                                   num_workers=num_workers, pin_memory=True,
                                                                   drop_last=False, timeout=defaults.dataloader_timeout)
                if len(train_unlabeled_dataset) > 0:
                    train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                         batch_size=batch_size_unlabeled,
                                                                         shuffle=True, num_workers=num_workers,
                                                                         pin_memory=True, drop_last=False,
                                                                         timeout=defaults.dataloader_timeout)
                else:
                    train_unlabeled_loader = None
            else:
                train_labeled_loader = train_unlabeled_loader = None
        else:
            train_valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                             num_workers=num_workers, pin_memory=True, drop_last=True,
                                                             timeout=defaults.dataloader_timeout)
            if frac_labeled is not None or num_labeled is not None:
                if num_labeled > 0:
                    train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset,
                                                                       batch_size=batch_size_labeled,
                                                                       shuffle=True,
                                                                       num_workers=num_workers,
                                                                       pin_memory=True,
                                                                       drop_last=False,
                                                                       timeout=defaults.dataloader_timeout)
                else:
                    train_labeled_loader = None
                train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                     batch_size=batch_size_unlabeled,
                                                                     shuffle=True,
                                                                     num_workers=num_workers,
                                                                     pin_memory=True,
                                                                     drop_last=False,
                                                                     timeout=defaults.dataloader_timeout)
            else:
                train_labeled_loader = train_unlabeled_loader = None
    else:
        if frac_labeled is not None or num_labeled is not None:
            train_dataset, train_labeled_dataset, train_unlabeled_dataset = remove_labels(dataset=train_dataset,
                                                                                      frac_labeled=frac_labeled,
                                                                                      num_labeled=num_labeled,
                                                                                      stratified=stratified_unlabeling)
            if stratified_sampling:
                train_labeled_sampler = StratifiedSampler(train_labeled_dataset, batch_size=batch_size)
        if stratified_sampling:
            train_sampler = StratifiedSampler(train_dataset, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=defaults.dataloader_timeout)
        if stratified_sampling:
            train_stratified_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                                  sampler=train_sampler, num_workers=num_workers,
                                                                  pin_memory=True, drop_last=False,
                                                                  timeout=defaults.dataloader_timeout)
            if frac_labeled is not None or num_labeled is not None:
                train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset,
                                                                   batch_size=batch_size_labeled,
                                                                   sampler=train_labeled_sampler,
                                                                   num_workers=num_workers,
                                                                   pin_memory=True,
                                                                   drop_last=False,
                                                                   timeout=defaults.dataloader_timeout)
                train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                     batch_size=batch_size_unlabeled,
                                                                     shuffle=True,
                                                                     num_workers=num_workers,
                                                                     pin_memory=True,
                                                                     drop_last=False,
                                                                     timeout=defaults.dataloader_timeout)
            else:
                train_labeled_loader = train_unlabeled_loader = None
        else:
            train_stratified_loader = None
            if frac_labeled is not None or num_labeled is not None:
                train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size_labeled,
                                                                   shuffle=True, num_workers=num_workers,
                                                                   pin_memory=True, drop_last=False,
                                                                   timeout=defaults.dataloader_timeout)
                train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset,
                                                                     batch_size=batch_size_unlabeled,
                                                                     shuffle=True,
                                                                     num_workers=num_workers,
                                                                     pin_memory=True,
                                                                     drop_last=False,
                                                                     timeout=defaults.dataloader_timeout)
            else:
                train_labeled_loader = train_unlabeled_loader = None
        valid_loader = None

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, drop_last=False, pin_memory=True,
                                              timeout=defaults.dataloader_timeout)

    if not separate_valid_set and valid_size > 0:
        return train_loader, train_stratified_loader, train_labeled_loader, train_unlabeled_loader, valid_loader, \
               train_valid_loader, test_loader
    else:
        return train_loader, train_stratified_loader, train_labeled_loader, train_unlabeled_loader, valid_loader, None,\
               test_loader


class PreloadedDataset(torch.utils.data.Dataset):
    """
    Dataset class for data that has already been loaded into memory that allows for transformations

    :param images: Images (or more generally, features) to be in the dataset
    :param labels: Labels for the above images
    :param transform: Transformations that should be applied to the images in the dataset
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.true_labels = labels.copy()

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        true_label = self.true_labels[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label, true_label

    def __len__(self):
        return len(self.images)
