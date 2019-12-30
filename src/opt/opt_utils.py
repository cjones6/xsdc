import faiss
import numpy as np
import scipy.stats
import torch

from src import default_params as defaults


def get_batch(data, labeled=False, unlabeled=True, deep_cluster=False):
    """
    Get a batch of training data from the labeled and/or unlabeled datasets.

    :param data: Object with the dataloaders
    :param labeled: Whether to return labeled data
    :param unlabeled: Whether to return unlabeled data
    :param deep_cluster: Whether to use the data loader for deep clustering
    :return: Tuple containing a subset of the following:

        * x_labeled: Batch of observations from the labeled training dataset
        * x_unlabeled: Batch of observations from the unlabeled training dataset
        * y_labeled: Labels for the observations from the labeled training set
        * y_unlabeled: Array with -1's
        * y_unlabeled_truth: True labels for the observations from the unlabeled training set
    """
    if deep_cluster:
        try:
            x_labeled, y_labeled = next(data.deepcluster_loader)
        except:
            data.deepcluster_loader = iter(data.deepcluster_loader)
            x_labeled, y_labeled = next(data.deepcluster_loader)
        return x_labeled, y_labeled

    if labeled:
        try:
            x_labeled, y_labeled, _ = next(data.train_labeled_iter)
        except:
            data.train_labeled_iter = iter(data.train_labeled_loader)
            x_labeled, y_labeled, _ = next(data.train_labeled_iter)
        x_labeled = x_labeled.type(torch.get_default_dtype()).to(defaults.device)
        y_labeled = y_labeled.to(defaults.device)
        if not unlabeled:
            return x_labeled, y_labeled

    if unlabeled:
        try:
            x_unlabeled, y_unlabeled, y_unlabeled_truth = next(data.train_unlabeled_iter)
        except:
            data.train_unlabeled_iter = iter(data.train_unlabeled_loader)
            x_unlabeled, y_unlabeled, y_unlabeled_truth = next(data.train_unlabeled_iter)
        x_unlabeled = x_unlabeled.type(torch.get_default_dtype()).to(defaults.device)
        y_unlabeled = y_unlabeled.to(defaults.device)
        if not labeled:
            return x_unlabeled, y_unlabeled, y_unlabeled_truth

    return x_labeled, x_unlabeled, y_labeled, y_unlabeled, y_unlabeled_truth


def compute_features(x, model, normalize=True, standardize=False, eps=1e-5):
    """
    Compute features for a minibatch of inputs and normalize or standardize them if specified.

    :param x: Inputs to the CKN
    :param model: Model (object from a subclass of the nn.Module class)
    :param normalize: Whether to normalize the data
    :param standardize: Whether to standardize the data
    :param eps: Value to clamp the standard deviation or norm of the features to (if applicable)
    :return: features: Computed features
    """
    features = model(x.to(defaults.device).type(torch.get_default_dtype()))
    features = features.contiguous().view(features.shape[0], -1)

    if normalize:
        mean = torch.mean(features, 1)
        features = features - mean.unsqueeze(1)
        nrm = torch.clamp(torch.mean(torch.norm(features, 2, 1)), min=eps)
        features = features / nrm
    elif standardize:
        mean = torch.mean(features, 0, keepdim=True)
        sd = torch.clamp(torch.std(features, 0, keepdim=True), min=eps)
        features = (features - mean) / sd

    return features


def compute_all_features(train_lab_loader, train_unlab_loader, valid_loader, test_loader, model, normalize=True,
                         standardize=False, eps=1e-5):
    """
    Generate features for all images in the training, validation, and test sets using the given model. Then normalize
    or standardize them if specified.

    :param train_lab_loader: Dataloader for the labeled training data
    :param train_unlab_loader: Dataloader for the unlabeled training data
    :param valid_loader: Dataloader for the validation data
    :param test_loader: Dataloader for the test data
    :param model: Model (object from a subclass of the nn.Module class)
    :param normalize: Whether to normalize the data
    :param standardize: Whether to standardize the data
    :param eps: Value to clamp the standard deviation or norm of the features to (if applicable)
    :return: all_features: Dictionary with the features and labels for each of the input datasets
    """
    with torch.autograd.set_grad_enabled(False):
        all_features = {'train_unlabeled': {'x': [], 'y': [], 'y_true': []}, 'train_labeled': {'x': [], 'y': []},
                        'valid': {'x': [], 'y': []}, 'test': {'x': [], 'y': []}}
        for dataset_name, data_loader in zip(sorted(all_features.keys()), [test_loader, train_lab_loader,
                                                                           train_unlab_loader, valid_loader]):
            if data_loader is not None:
                for i, (x, y, ytrue) in enumerate(data_loader):
                    x = x.type(torch.get_default_dtype()).to(defaults.device)
                    features = model(x)
                    features = features.contiguous().view(features.shape[0], -1).data.cpu()
                    all_features[dataset_name]['x'].append(features)
                    all_features[dataset_name]['y'].append(y)
                    if 'y_true' in all_features[dataset_name]:
                        all_features[dataset_name]['y_true'].append(ytrue)

                all_features[dataset_name]['x'] = torch.cat(all_features[dataset_name]['x'])
                all_features[dataset_name]['y'] = torch.cat(all_features[dataset_name]['y'])
                if 'y_true' in all_features[dataset_name]:
                    all_features[dataset_name]['y_true'] = torch.cat(all_features[dataset_name]['y_true'])

        if normalize:
            for key in all_features.keys():
                if len(all_features[key]['x']) > 0:
                    all_features[key]['x'].sub_(torch.mean(all_features[key]['x'], 1, keepdim=True))
            if train_lab_loader is not None:
                nrm = torch.clamp(torch.mean(torch.norm(all_features['train_labeled']['x'], 2, 1)), eps)
            elif train_unlab_loader is not None:
                nrm = torch.clamp(torch.mean(torch.norm(all_features['train_unlabeled']['x'], 2, 1)), eps)
            else:
                nrm = torch.clamp(torch.mean(torch.norm(all_features['test']['x'], 2, 1)), eps)
            for key in all_features.keys():
                if len(all_features[key]['x']) > 0:
                    all_features[key]['x'].div_(nrm)
        elif standardize:
            if train_lab_loader is not None:
                mean = torch.mean(all_features['train_labeled']['x'], 0, keepdim=True)
                sd = torch.clamp(torch.std(all_features['train_labeled']['x'], 0, keepdim=True), min=eps)
            elif train_unlab_loader is not None:
                mean = torch.mean(all_features['train_unlabeled']['x'], 0, keepdim=True)
                sd = torch.clamp(torch.std(all_features['train_unlabeled']['x'], 0, keepdim=True), min=eps)
            else:
                mean = torch.mean(all_features['test']['x'], 0, keepdim=True)
                sd = torch.clamp(torch.std(all_features['test']['x'], 0, keepdim=True), min=eps)
            for key in all_features.keys():
                if len(all_features[key]['x']) > 0:
                    all_features[key]['x'].sub_(mean)
                    all_features[key]['x'].div_(sd)

    return all_features


def compute_normalizations(model):
    """
    Compute the term k(W^TW)^{-1/2} for each layer.

    :param model: CKN model
    :return model: Model after the normalizations k(W^TW)^{-1/2} have been (re)computed and stored
    """
    for layer_num in range(len(model.layers)):
        model.layers[layer_num].store_normalization = True
        with torch.autograd.set_grad_enabled(False):
            model.layers[layer_num].normalization = model.layers[layer_num].compute_normalization()

    return model


def evaluate_features(params, w_last, b_last, all_features):
    """
    Evaluate the current performance using the given features output by the network.

    :param params: Parameters object
    :param w_last: Estimated coefficients of the output layer
    :param b_last: Estimated intercepts of the output layer
    :param all_features: Dictionary with features and labels for the labeled and unlabeled training sets, the validation
                         set, and the test set
    :return: Dictionary of results with the accuracy and loss on each dataset
    """
    results = {}
    names = ['train_labeled', 'train_unlabeled', 'valid', 'test']
    for name in names:
        x = all_features[name]['x']
        if name != 'train_unlabeled':
            y = all_features[name]['y']
        else:
            y = all_features[name]['y_true']
        if len(x) > 0:
            x = x.to(defaults.device)
            y = y.to(defaults.device)
            if w_last is not None:
                yhat = x.mm(w_last) + b_last
            else:
                yhat = x
            y_one_hot = one_hot_embedding(y, params.nclasses).to(defaults.device)
            loss = (torch.sum((yhat - y_one_hot) ** 2)/len(y)).item()
            yhat = torch.argmax(yhat, 1)
            loss = loss + params.lam*torch.sum(w_last**2)
            acc = (yhat == y.long()).float().mean().item()

            results[name + '_accuracy'] = acc
            results[name + '_loss'] = loss.item()
        else:
            results[name + '_accuracy'] = 0
            results[name + '_loss'] = np.inf

    n_labeled = len(all_features['train_labeled']['y'])
    n_unlabeled = len(all_features['train_unlabeled']['y_true'])
    results['train_accuracy'] = (results['train_labeled_accuracy']*n_labeled +
                                 results['train_unlabeled_accuracy']*n_unlabeled)/(n_labeled+n_unlabeled)
    results['train_loss'] = (results['train_labeled_loss']*n_labeled +
                             results['train_unlabeled_loss']*n_unlabeled)/(n_labeled+n_unlabeled)

    return results


def one_hot_embedding(y, n_dims):
    """
    Generate a one-hot representation of the input vector y. From
    https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23

    :param y: Labels for which a one-hot representation should be created
    :param n_dims: Number of unique labels
    :return: One-hot representation of y
    """
    y_tensor = y.data.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.shape[0], -1)

    return y_one_hot.type(torch.get_default_dtype()).to(defaults.device)


def print_results(iteration, results, header=False):
    """
    Print the results at the current iteration.

    :param iteration: Current iteration number
    :param results: Dictionary with the test_accuracy and test_loss
    :param header: Whether to print the column headers
    """
    if header:
        print('Iteration \t Test accuracy \t Test loss')
    print(iteration, '\t\t',
          '{:06.4f}'.format(results['test_accuracy']), '\t',
          '{:06.4f}'.format(results['test_loss']),
          )


def nearest_neighbor(labeled_features, unlabeled_features, labels, k):
    """
    Find the nearest neighbors to each unlabeled feature in terms of l2 distance. Assign the label of each unlabeled
    feature to be the mode of the k labels from the k nearest labeled features.

    :param labeled_features: Features whose labels are known
    :param unlabeled_features: Features for which you want to perform k-NN
    :param labels: Labels corresponding to the labeled features
    :param k: Number of nearest neighbors to use
    :return: Estimated label for each feature in unlabeled_features
    """
    labeled_features = np.ascontiguousarray(labeled_features).astype('float32')
    unlabeled_features = np.ascontiguousarray(unlabeled_features).astype('float32')
    nq, d = labeled_features.shape
    if defaults.device.type == 'cuda':
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(labeled_features)
    D, idxs = index.search(unlabeled_features, k)
    nn_labels = labels[idxs]
    yhat_unlabeled = scipy.stats.mode(nn_labels, axis=1)[0].flatten()

    return torch.Tensor(yhat_unlabeled).to(defaults.device)
