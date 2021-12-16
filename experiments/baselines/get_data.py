import os
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from baseline.datasets.datasets import Gisette, Magic
from baseline.pipeline_utils import save, load, format_files, var_to_str
import scipy.sparse.linalg as splinalg
from baseline.kernels_utils import RBF

datasets = dict(mnist=torchvision.datasets.MNIST, cifar=torchvision.datasets.CIFAR10,
                gisette=Gisette, magic=Magic)
size_datasets = dict(gisette=6000, magic=10032, mnist=60000, cifar=50000)
n_class_datasets = dict(gisette=2, magic=2, mnist=10, cifar=10)
mean_pix = dict(mnist=(0.1307,), cifar=(0.4914, 0.4822, 0.4465))
std_pix = dict(mnist=(0.3081,), cifar=(0.247, 0.243, 0.261))


# todo: add center, standardized for all datasets, augmented data for CIFAR
def get_data(dataset, valid=False, data_aug=0,
             centering=True, scaling=False, vectorize=True):
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', dataset)

    data_transforms = []
    if dataset in ['mnist', 'cifar']:
        data_transforms.append(transforms.ToTensor())
        data_transforms.append(transforms.Normalize(mean_pix[dataset], std_pix[dataset]))
        if vectorize:
            data_transforms.append(Vectorize())
    if centering or scaling:
        mean, scale = get_mean_scale(dataset)
        data_transforms.append(Center(mean))
        if scaling:
            data_transforms.append(Scale(scale))
    transform = transforms.Compose(data_transforms)
    train_data = datasets[dataset](root=dataset_path, train=True, transform=transform, download=True)
    if valid:
        n_valid = int(len(train_data)/5)
        train_data, test_data = random_split(train_data, [len(train_data)-n_valid, n_valid])
    else:
        test_data = datasets[dataset](root=dataset_path, train=False, transform=transform, download=True)

    return train_data, test_data


def get_data_loaders(dataset, seed=0, n_labels=0, valid=False, data_aug=0,
                     centering=True, scaling=False,
                     vectorize=True, stratified=True, batch_size=128):
    torch.manual_seed(seed)
    train_data, test_data = get_data(dataset, valid, data_aug, centering, scaling, vectorize)

    if stratified:
        train_sampler = StratifiedSampler(train_data, batch_size)
    else:
        train_sampler = RandomSampler(train_data)

    labeled = keep_labels(train_data.targets, seed, n_labels)
    train_data = DatasetWithIdx(train_data, labeled)
    test_data = DatasetWithIdx(test_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


def get_full_data(dataset, scaling=False, n_train=None):
    data_path = os.path.dirname(os.path.abspath(__file__))
    data_cfg = dict(scaling=scaling)
    if n_train is not None:
        data_cfg.update(n_train=n_train)
    file_path = os.path.join(data_path, 'datasets', dataset, var_to_str(data_cfg) + '_full_data') + format_files
    if os.path.exists(file_path):
        full_data = load(open(file_path, 'rb'))
    else:
        train_data, test_data = get_data(dataset, scaling=scaling)

        train_loader = DataLoader(train_data, batch_size=128)
        test_loader = DataLoader(test_data, batch_size=128)

        full_data = dict(train=dict(X=[], y=[]), test=dict(X=[], y=[]))

        for datasplit, loader in zip(full_data.values(), [train_loader, test_loader]):
            for x, y in loader:
                datasplit['X'].append(x)
                datasplit['y'].append(y)
            datasplit['X'] = torch.cat(datasplit['X'])
            datasplit['y'] = torch.cat(datasplit['y'])
        if n_train is not None:
            full_data['train']['X'] = full_data['train']['X'][:n_train]
            full_data['train']['y'] = full_data['train']['y'][:n_train]
        full_data.update(n_class=n_class_datasets[dataset])
        save(full_data, open(file_path, 'wb'))
    return full_data


def preprocess_gram(dataset, scaling, n_train=None, kernel='linear', reg=None, sigma=None):
    data_path = os.path.dirname(os.path.abspath(__file__))
    data_cfg = dict(scaling=scaling, kernel=kernel)
    if reg is not None:
        data_cfg.update(reg=reg)
    if sigma is not None:
        data_cfg.update(sigma=sigma)
    if n_train is not None:
        data_cfg.update(n_train=n_train)
    file_path = os.path.join(data_path, 'datasets', dataset, var_to_str(data_cfg) + 'gram_data') + format_files

    if os.path.exists(file_path):
        gram_data = load(open(file_path, 'rb'))
    else:
        full_data = get_full_data(dataset, scaling, n_train)
        gram_train = compute_gram(full_data['train']['X'], full_data['train']['X'], kernel, reg, sigma)
        gram_test = compute_gram(full_data['test']['X'], full_data['train']['X'], kernel, reg, sigma)

        gram_data = dict(train=dict(gram=gram_train, y=full_data['train']['y']),
                         test=dict(gram=gram_test, y=full_data['test']['y']), n_class=full_data['n_class'])
        save(gram_data, open(file_path, 'wb'))
    return gram_data


def compute_gram(X, Y, kernel='linear', reg=None, sigma=None):
    if kernel == 'linear':
        gram = X.mm(Y.t())
    elif kernel == 'svd':
        gram = X.mm(torch.solve(Y.t(), Y.t().mm(Y)/Y.size(0) + reg*torch.eye(Y.size(1)))[0])
    elif kernel == 'rbf':
        kernel_func = RBF(sigma)
        gram = kernel_func(X, Y)
    elif kernel == 'rbfsvd':
        kernel_func = RBF(sigma)
        gram = torch.solve(kernel_func(Y, X), kernel_func(Y, Y)/Y.size(0) + reg*torch.eye(Y.size(0)))[0].t()
    else:
        raise NotImplementedError
    return gram


def get_preprocessed_data(dataset, scaling, n_train, seed, n_labels, kernel, reg=None, sigma=None):
    torch.manual_seed(seed)

    hyper_params_ref = get_hyper_params_ref(dataset, scaling)
    if kernel == 'svd' and reg is None:
        reg = hyper_params_ref['reg_ref']
    if kernel == 'rbf' and sigma is None:
        sigma = hyper_params_ref['sigma_ref']
    if kernel == 'rbfsvd':
        raise NotImplementedError

    gram_data = preprocess_gram(dataset, scaling, n_train, kernel, reg, sigma)
    idx_labeled = keep_labels(gram_data['train']['y'], seed, n_labels)
    gram_data.update(idx_labeled=idx_labeled)
    return gram_data


def get_mean_scale(dataset):
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', dataset)
    file_path = os.path.join(dataset_path, 'mean_std') + format_files
    if os.path.exists(file_path):
        mean, scale = load(open(file_path, 'rb'))
    else:
        data_transforms = []
        if dataset in ['mnist', 'cifar']:
            data_transforms.append(transforms.ToTensor())
            data_transforms.append(transforms.Normalize(mean_pix[dataset], std_pix[dataset]))
            data_transforms.append(Vectorize())
        if len(data_transforms) > 0:
            data_transform = transforms.Compose(data_transforms)
        else:
            data_transform = None

        data = datasets[dataset](root=dataset_path, train=True, transform=data_transform, download=True)
        train_loader = DataLoader(data, batch_size=128)

        X = []
        for x, y in train_loader:
            X.append(x)
        X = torch.cat(X).numpy()
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X)
        mean, scale = torch.from_numpy(scaler.mean_), torch.from_numpy(scaler.scale_)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save([mean, scale], open(file_path, 'wb'))
    return mean, scale


def get_hyper_params_ref(dataset, scaling=True):
    subsample = 5
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', dataset)
    data_cfg = dict(scaling=scaling)
    file_path = os.path.join(dataset_path, var_to_str(data_cfg) + '_hyper_params_ref') + format_files

    if os.path.exists(file_path):
        hyperparams_ref = load(open(file_path, 'rb'))
    else:
        full_data = get_full_data(dataset, scaling=scaling)
        X_train = full_data['train']['X']
        X_train = X_train[::subsample]
        del full_data

        cov = (X_train.t().mm(X_train)/len(X_train)).numpy()
        lambda_max = np.real(splinalg.eigs(cov, 1)[0])
        reg_ref = (lambda_max/np.trace(cov)).item()
        reg_ref = float('{:1.2e}'.format(reg_ref))
        print(f'reg ref: {reg_ref}')
        del cov

        dists = torch.cdist(X_train, X_train)**2
        del X_train
        dists = torch.triu(dists)
        sigma_ref = torch.sqrt(torch.median(dists[dists != 0])/2).item()
        sigma_ref = float('{:1.2e}'.format(sigma_ref))
        print(f'sigma ref: {sigma_ref}')

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        hyperparams_ref = dict(reg_ref=reg_ref, sigma_ref=sigma_ref)
        save(hyperparams_ref, open(file_path, 'wb'))
    return hyperparams_ref


def get_reg_ref_rbf(dataset, scaling, nb_train, sigma):
    subsample = 5
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', dataset)
    data_cfg = dict(scaling=scaling, nb_train=nb_train, sigma=sigma)
    file_path = os.path.join(dataset_path, var_to_str(data_cfg) + '_reg_rbf_ref') + format_files

    if os.path.exists(file_path):
        reg_ref_rbf = load(open(file_path, 'rb'))
    else:
        gram_data = preprocess_gram(dataset, scaling=scaling, n_train=nb_train, kernel='rbf', sigma=sigma)
        gram_train = gram_data['train']['gram']
        del gram_data
        gram_train = gram_train[::subsample, ::subsample]/gram_train.size(0)
        lambda_max = np.real(splinalg.eigs(gram_train, 1)[0])
        reg_ref_rbf = (lambda_max / np.trace(gram_train)).item()
        reg_ref_rbf = float('{:1.2e}'.format(reg_ref_rbf))
        print(f'reg_ref_rbf: {reg_ref_rbf}')

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save(reg_ref_rbf, open(file_path, 'wb'))
    return reg_ref_rbf


class DatasetWithIdx(Dataset):
    def __init__(self, dataset, labeled=None):
        super(DatasetWithIdx, self).__init__()
        self.dataset = dataset
        self.labeled = labeled

    def __getitem__(self, index):
        point, label = self.dataset[index]
        labeled = index in self.labeled if self.labeled is not None else True
        return point, label, index, labeled

    def __len__(self):
        return len(self.dataset)


def keep_labels(labels, seed, n_labels=0):
    if n_labels == 0:
        labeled = []
    else:
        n_splits = int(len(labels) / n_labels)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        if type(labels) == torch.Tensor:
            y = labels.numpy()
        elif type(labels) == list:
            y = np.array(labels)
        else:
            raise NotImplementedError
        X = np.random.randn(len(y), 1)
        _, labeled = next(skf.split(X, y))
    return labeled


# todo: check this part
class StratifiedSampler(torch.utils.data.Sampler):
    """
    Sampler to create batches with balanced classes.

    :param dataset: Dataset to sample from
    :param batch_size: Size of the batches output by the sampler
    :param weights: Class weights for stratified sampling
    """
    def __init__(self, dataset, batch_size, weights=None):
        super(StratifiedSampler, self).__init__(dataset)
        self.labels = dataset.targets
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


class Vectorize():
    def __call__(self, tensor):
        return tensor.view(-1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Center():
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        return tensor-self.mean

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Scale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, tensor):
        return tensor/self.scale

    def __repr__(self):
        return self.__class__.__name__ + '()'


def make_synth_data(dist_clust, n, k, d, radius):
    # X_lab = torch.randn(n_lab, d)
    X = torch.cat([dist_clust * i + radius*torch.randn(int(n / k), d) for i in range(k)])
    # y_lab = one_hot_embedding(torch.multinomial(torch.ones(k), n_lab, replacement=True), k)
    y = torch.zeros(n, dtype=torch.long)
    for i in range(k):
        y[i * int(n / k):(i + 1) * int(n / k)] = i
    return X, y


def preprocess_dataset(dataset, kernel, reg_fac, sigma_fac, scaling=True):
    hyper_params_ref = get_hyper_params_ref(dataset, scaling)
    scaling = True
    for dataset in ['gisette', 'magic', 'mnist', 'cifar']:
        n_train = 10000 if dataset in ['mnist', 'cifar'] else None
        # get_mean_scale(dataset)
        for kernel in ['linear', 'rbf', 'svd']:
            reg = sigma = None
            if kernel == 'svd':
                reg = hyper_params_ref['reg_ref'] * reg_fac
            elif kernel == 'rbf':
                sigma = hyper_params_ref['sigma_ref'] * sigma_fac
            # elif kernel == 'rbfsvd':
            #     sigma = hyper_params_ref['sigma_ref'] * sigma_fac
            #     reg_ref = get_reg_ref_rbf(dataset, scaling, n_train, sigma)
            #     reg = reg_ref * reg_fac
            print(f'dataset: {dataset}\nkernel: {kernel}\nsigma: {sigma}\nreg:{reg}')
            preprocess_gram(dataset, scaling, n_train, kernel, reg, sigma)


if __name__ == '__main__':
    scaling = True
    for dataset in ['gisette', 'magic', 'mnist', 'cifar']:
        print(get_hyper_params_ref(dataset, scaling))
    # pass
