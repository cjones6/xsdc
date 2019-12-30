import torch
import torchvision

from . import create_data_loaders, data_loader_utils


def get_dataloaders(valid_size=10000, batch_size=128, batch_size_labeled=None, batch_size_unlabeled=None,
                    transform='std', data_path='../data/MNIST', frac_labeled=None, num_labeled=None,
                    stratified_unlabeling=False, stratified_sampling=False, unlabeled_imbalance=-1, num_workers=4,
                    seed=None):
    """
    Create data loaders for MNIST.

    :param valid_size: Size of the validation set
    :param batch_size: Batch size of (labeled+unlabeled) training data and test data
    :param batch_size_labeled: Batch size for the labeled training data
    :param batch_size_unlabeled: Batch size for the unlabeled training data
    :param transform: How the raw data should be transformed. Either 'std' (standardized) or None.
    :param data_path: Directory where the data either currently exists or should be downloaded to
    :param frac_labeled: Fraction of the training set that should be labeled. Only one of frac_labeled and num_labeled
                         can be specified.
    :param num_labeled: Number of images in the training set that should be labeled. Only one of frac_labeled and
                        num_labeled can be specified.
    :param stratified_unlabeling: Whether when removing labels to create the unlabeled training set this should be done
                                  in a stratified manner or uniformly at random
    :param stratified_sampling: Whether the labeled training data should be sampled in a stratified manner, with the
                                same number of examples from each class. This is also used to determine whether a
                                stratified dataloader for the labeled+unlabeled training data should be returned
    :param unlabeled_imbalance: Fraction of the data that should have labels 0-4 in the unlabeled training set. If -1,
                                then the unlabeled training set will be balanced if stratified_unlabeling=True and will
                                be close to balanced otherwise.
    :param num_workers: Number of workers to use for the dataloaders
    :param seed: Seed to use when making the training dataset unbalanced (if applicable)
    :return: Tuple containing:

            * train_loader: Dataloader for the (labeled+unlabeled) training data
            * train_stratified_loader: If stratified_sampling=True, this is a dataloader for the (labeled+unlabeled)
                                       training data in which each batch has equal representation from each class. Else,
                                       this is None.
            * train_labeled_loader: Dataloader for the labeled training data. If stratified_sampling=True, each batch
                                    has equal representation from each class.
            * train_unlabeled_loader: Dataloader for the unlabeled training data
            * valid_loader: Dataloader for the validation set
            * None
            * test_loader: Dataloader for the test set
    """
    def std(x):
        return (x.type(torch.get_default_dtype())/255.0-0.1307)/0.3081

    if transform == 'std':
        transform = std
    elif transform is not None:
        raise NotImplementedError

    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    if unlabeled_imbalance == -1:
        train_dataset = create_data_loaders.PreloadedDataset(train_dataset.data.unsqueeze(1),
                                                             train_dataset.targets.numpy(),
                                                             transform)
        test_dataset = create_data_loaders.PreloadedDataset(test_dataset.data.unsqueeze(1),
                                                            test_dataset.targets.numpy(),
                                                            transform)

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
        ntrain = len(train_dataset)-valid_size
        unbalanced_train_data, unbalanced_train_labels = data_loader_utils.unbalance_multiclass(
                                                             train_dataset.train_data.unsqueeze(1)[:ntrain],
                                                             train_dataset.train_labels.numpy()[:ntrain],
                                                             unlabeled_imbalance,
                                                             frac_labeled,
                                                             num_labeled,
                                                             seed=seed)

        valid_dataset = create_data_loaders.PreloadedDataset(train_dataset.data.unsqueeze(1)[ntrain:],
                                                             train_dataset.targets.numpy()[ntrain:],
                                                             transform)
        train_dataset = create_data_loaders.PreloadedDataset(unbalanced_train_data, unbalanced_train_labels, transform)
        test_dataset = create_data_loaders.PreloadedDataset(test_dataset.data.unsqueeze(1),
                                                            test_dataset.targets.numpy(),
                                                            transform)

        return create_data_loaders.generate_dataloaders(train_dataset,
                                                        test_dataset,
                                                        valid_dataset=valid_dataset,
                                                        separate_valid_set=True,
                                                        valid_size=valid_size,
                                                        batch_size=batch_size,
                                                        batch_size_labeled=batch_size_labeled,
                                                        batch_size_unlabeled=batch_size_unlabeled,
                                                        frac_labeled=frac_labeled,
                                                        num_labeled=num_labeled,
                                                        stratified_unlabeling=stratified_unlabeling,
                                                        stratified_sampling=stratified_sampling,
                                                        num_workers=num_workers)
