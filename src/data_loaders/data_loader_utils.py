import numpy as np
import torch


def _get_labels(dataset):
    """
    Extract the labels from the input dataset.

    :param dataset: PyTorch dataset
    :return: labels: Labels from the dataset
    """
    if hasattr(dataset, 'type') and dataset.type() == 'torch.LongTensor':
        labels = dataset
    elif 'tensors' in dataset.__dict__.keys():
        labels = dataset.tensors[1]
    elif 'train_labels' in dataset.__dict__.keys():
        labels = dataset.train_labels
    elif 'labels' in dataset.__dict__.keys():
        labels = dataset.labels
    elif 'dataset' in dataset.__dict__.keys():
        if 'dataset' not in dataset.dataset.__dict__.keys():
            if 'train_labels' in dataset.dataset.__dict__.keys():
                if isinstance(dataset.dataset.train_labels, list):
                    dataset.dataset.train_labels = torch.IntTensor(dataset.dataset.train_labels)
                labels = dataset.dataset.train_labels[dataset.indices]
            elif 'targets' in dataset.dataset.__dict__.keys():
                if isinstance(dataset.dataset.targets, list):
                    dataset.dataset.targets = torch.IntTensor(dataset.dataset.targets)
                labels = dataset.dataset.targets[dataset.indices]
            else:
                if isinstance(dataset.dataset.labels, list):
                    dataset.dataset.labels = torch.IntTensor(dataset.dataset.labels)
                labels = dataset.dataset.labels[dataset.indices]

        else:
            if 'train_labels' in dataset.dataset.dataset.__dict__.keys():
                if isinstance(dataset.dataset.dataset.train_labels, list):
                    dataset.dataset.dataset.train_labels = torch.IntTensor(dataset.dataset.dataset.train_labels)
                labels = dataset.dataset.dataset.train_labels[dataset.indices]
            elif 'targets' in dataset.dataset.dataset.__dict__.keys():
                if isinstance(dataset.dataset.dataset.targets, list):
                    dataset.dataset.dataset.targets = torch.IntTensor(dataset.dataset.dataset.targets)
                labels = dataset.dataset.dataset.targets[dataset.indices]
            else:
                if isinstance(dataset.dataset.dataset.labels, list):
                    dataset.dataset.dataset.labels = torch.IntTensor(dataset.dataset.dataset.labels)
                labels = dataset.dataset.dataset.labels[dataset.indices]

    elif 'targets' in dataset.__dict__.keys():
        labels = dataset.targets
    else:
        raise NotImplementedError

    if isinstance(labels, list):
        labels = torch.IntTensor(labels)

    return labels


def _replace_labels(dataset, labels):
    """
    Replace the labels in the given dataset with the input labels.

    :param dataset: Dataset in which to replace the labels
    :param labels: New labels for the dataset
    :return: dataset: dataset with the labels replaced
    """
    if 'tensors' in dataset.__dict__.keys():
        dataset.tensors[1] = labels
    elif 'train_labels' in dataset.__dict__.keys():
        dataset.train_labels = labels
    elif 'labels' in dataset.__dict__.keys():
        dataset.labels = labels
    elif 'dataset' in dataset.__dict__.keys():
        if hasattr(dataset.dataset, 'train_labels'):
            dataset.dataset.train_labels[dataset.indices] = labels
        elif hasattr(dataset.dataset, 'labels'):
            dataset.dataset.labels[dataset.indices] = labels
    else:
        raise NotImplementedError

    return dataset


def remove_labels(dataset=None, stratified=False, frac_labeled=None, num_labeled=None):
    """
    Remove labels from the data in the input dataset. Keep only frac_labeled or num_labeled points and do so in either
    a stratified or random manner. This function assumes that all of the data is loaded into memory at once.

    :param dataset: The dataset from which some labels should be removed
    :param stratified: Whether the labels should be removed in a stratified manner, so that they are removed equally
                       across classes, or in a random manner
    :param frac_labeled: The fraction of data that should be labeled
    :param num_labeled: The number of observations that should be labeled
    :return: Tuple containing:

            * dataset: Dataset with some of the labels removed
            * labeled_dataset: Dataset containing only the labeled examples from dataset
            * unlabeled_dataset: Dataset containing only the unlabeled examples from dataset
    """
    if frac_labeled is None and num_labeled is None:
        raise ValueError('Either the fraction of labeled data or the number of labeled data points must be specified')
    elif frac_labeled is not None and num_labeled is not None:
        raise ValueError('Only one of the fraction of labeled data or the number of labeled data points can be '
                         'specified')
    labels = _get_labels(dataset)

    if frac_labeled is not None:
        num_labeled = int(np.ceil(len(labels)*frac_labeled))
        max_num_labeled = int(len(labels)*frac_labeled)
        print('Number of labeled data points being used:', max_num_labeled)
    else:
        max_num_labeled = num_labeled
        print('Number of labeled data points being used:', max_num_labeled)

    if stratified:
        if dataset is not None:
            sampler = StratifiedSampler(dataset, num_labeled)
            labeled_idxs = sampler.sample_idxs()
        else:
            raise NotImplementedError
    else:
        labeled_idxs = np.random.choice(range(len(labels)), num_labeled, replace=False)

    labeled_idxs = labeled_idxs[:max_num_labeled]
    labels_selected = labels[labeled_idxs]
    labels[:] = -1
    labels[labeled_idxs] = labels_selected

    if 'dataset' not in dataset.__dict__.keys():
        unlabeled_idxs = [i for i in range(len(dataset)) if i not in labeled_idxs]
        labeled_dataset = torch.utils.data.dataset.Subset(dataset, labeled_idxs)
        unlabeled_dataset = torch.utils.data.dataset.Subset(dataset, unlabeled_idxs)
    else:
        dataset_indices = torch.LongTensor(dataset.indices)
        unlabeled_idxs = list(set(range(len(dataset_indices))).difference(set(labeled_idxs)))
        labeled_dataset = torch.utils.data.dataset.Subset(dataset.dataset, dataset_indices[labeled_idxs])
        unlabeled_dataset = torch.utils.data.dataset.Subset(dataset.dataset, dataset_indices[unlabeled_idxs])
    dataset = _replace_labels(dataset, labels)

    return dataset, labeled_dataset, unlabeled_dataset


class StratifiedSampler(torch.utils.data.Sampler):
    """
    Sampler to create batches with balanced classes.

    :param dataset: Dataset to sample from
    :param batch_size: Size of the batches output by the sampler
    :param weights: Class weights for stratified sampling
    """
    def __init__(self, dataset, batch_size, weights=None):
        self.labels = _get_labels(dataset)
        self.n_splits = int(np.ceil(len(self.labels) / batch_size))
        self.batch_size = batch_size
        self.nclasses = max(self.labels)+1
        self.class_idxs = [np.where(self.labels == i)[0] for i in range(-1, self.nclasses)]
        self.weights = weights

    def sample_idxs(self):
        """
        Sample indices of labels.

        :return: idxs[:self.batch_size]: A single batch of indices
        """
        idxs = []
        for i in range(len(self.class_idxs)-1, -1, -1):
            if len(self.class_idxs[i]) > 0:
                if i != 0:
                    num_to_sample = min(int(np.ceil(self.batch_size/self.nclasses)), len(self.class_idxs[i]))
                else:
                    # Oversample the unlabeled data if there isn't enough labeled data
                    num_to_sample = min(len(self.class_idxs[i]), self.batch_size-len(idxs))
                if self.weights is None:
                    cls_idxs = np.random.choice(self.class_idxs[i], num_to_sample, replace=False, p=None)
                else:
                    probs = self.weights[self.class_idxs[i]]/torch.sum(self.weights[self.class_idxs[i]])
                    sum_probs = torch.sum(probs)
                    if sum_probs != 1:  # Take care of the case where there's roundoff error
                        probs[-1] = torch.clamp(probs[-1] + (1 - sum_probs), min=0)
                    cls_idxs = np.random.choice(self.class_idxs[i], num_to_sample, replace=False, p=probs)
                idxs.extend(cls_idxs)

        np.random.shuffle(idxs)
        return idxs[:self.batch_size]

    def __iter__(self):
        """
        Obtain an iterator with self.n_splits batches

        :return: iter(idxs): Iterator with self.n_splits batches
        """
        idxs = np.concatenate([self.sample_idxs() for _ in range(self.n_splits)]).tolist()
        return iter(idxs)

    def __len__(self):
        """
        Obtain the number of labels in the dataset

        :return: len(self.labels): Number of labels in the dataset
        """
        return len(self.labels)


def unbalance_multiclass(features, labels, imbalance, frac_labeled, num_labeled, nclasses=10, seed=None):
    """
    Create an unbalanced set of (features, labels) given the original training set features and labels. The unlabeled
    data will be unbalanced, but the labeled data will still be balanced.

    :param features: Training set features from which to extract an unbalanced subset
    :param labels: Training set labels from which to extract an unbalanced subset
    :param imbalance: Fraction of observations to be in classes 0-nclasses//2 in the unlabeled dataset
    :param frac_labeled: Fraction of observations will be labeled
    :param num_labeled: Number of observations that will be labeled
    :param seed: Seed to use when shuffling the data
    :return: Tuple containing

        * features[all_idxs]: The features for use in the unbalanced dataset
        * labels[all_idxs]: The corresponding labels
    """
    if seed is not None:
        np.random.seed(seed)
    num_examples = len(features)
    if frac_labeled is not None:
        num_labeled = int(num_examples * frac_labeled)

    assert len(np.unique(labels)) == nclasses

    # Number of examples per class 0-nclasses//2 = # of unlabeled examples per class with labels 0-nclasses//2
    # (imbalanced across classes 0-nclasses but balanced across classes 0-nclasses//2) + # of labeled examples in each
    # class (balanced across all classes)
    num_first_half_classes = int((num_examples/2-num_labeled)/nclasses*2*imbalance + num_labeled/nclasses)
    num_second_half_classes = int((num_examples/2-num_labeled)/nclasses*2*(1-imbalance) + num_labeled/nclasses)

    all_idxs = []
    for i in range(nclasses//2):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        all_idxs.extend(list(idxs[:num_first_half_classes]))

    for i in range(nclasses//2, nclasses):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        all_idxs.extend(list(idxs[:num_second_half_classes]))

    return features[all_idxs], labels[all_idxs]
