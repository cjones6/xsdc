import numpy as np
import scipy.optimize
import torch
import torch.nn as nn

from src import default_params as defaults
from src.opt.opt_utils import compute_all_features, one_hot_embedding


def next_lambda(accuracies, idxs):
    """
    Perform the next step of golden-section search to get the next value of the cross-validation parameter to try given
    the accuracies for the ones previously tried and the current interval under consideration.

    :param accuracies: Array of accuracies obtained so far
    :param idxs: Array of four values indexing the four parameter values of current interest in golden-section search
    :return Tuple containing:

        * The index of the next parameter value to try
        * The updated idxs array
    """
    tau = 0.5 + np.sqrt(5)/2
    x1 = idxs[0]
    x4 = idxs[-1]
    if idxs[0] is None:
        idxs[0] = 0
        return 0, idxs
    elif idxs[3] is None:
        idxs[-1] = len(accuracies)-1
        return -1, idxs
    elif idxs[1] is None:
        x2 = int((1/tau*x1 + (1-1/tau)*x4))
        idxs[1] = x2
        return x2, idxs
    elif idxs[2] is None:
        x3 = int(((1-1/tau)*x1 + (1/tau)*x4))
        idxs[2] = x3
        return x3, idxs
    else:
        if accuracies[idxs[1]] > accuracies[idxs[2]]:
            idxs[3] = idxs[2]
            idxs[2] = idxs[1]
            idxs[1] = int(np.rint((1/tau*idxs[2] + (1-1/tau)*idxs[0])))
            return idxs[1], idxs
        else:
            idxs[0] = idxs[1]
            idxs[1] = idxs[2]
            idxs[2] = int(np.rint((1/tau*idxs[1] + (1-1/tau)*idxs[3])))
            return idxs[2], idxs


def train(train_loader, valid_loader, test_loader, model, nclasses, maxiter, w_init=None, normalize=True,
          standardize=False, loss_name='square', lambdas=None, input_features=False):
    """
    Generate features from the given model and then train a classifier. Perform cross-validation for the regularization
    parameter lambda of the loss function.

    :param train_loader: Dataloader for the training set or training set (features, labels)
    :param valid_loader: Dataloader for the validation set or validation set (features, labels)
    :param test_loader: Dataloader for the test set or test set (features, labels)
    :param model: Model (object from a subclass of the nn.Module class)
    :param nclasses: Number of classes in the dataset
    :param maxiter: Maximum number of iterations used to train the classifier during the validation phase and testing
                    phase
    :param w_init: Initial parameter values for the classifier
    :param normalize: Whether to normalize the data
    :param standardize: Whether to standardize the data
    :param loss_name: Name of the loss function to use. Either 'square' or 'mnl'.
    :param lambdas: Values of lambda to consider. If None, it will use 2^-40,2^-39...,2^0.
    :param input_features: Whether the first three inputs are precomputed features (True) or dataloaders (False)
    :return: Tuple containing:

        * test_acc: Accuracy on the test set from the final training of the classifier
        * valid_acc: Accuracy on the validation set from when performing cross-validation
        * train_acc: Accuracy on the training set from the final training of the classifier
        * test_loss: Loss on the test set from the final training of the classifier
        * train_loss: Loss on the training set from the final training of the classifier
        * w: Trained parameters of the classifier
        * best_lambda: Best value of the regularization parameter lambda found
    """
    found_best = False
    if lambdas is None:
        lambdas = [2 ** i for i in range(-40, 1, 1)]
    elif len(lambdas) == 1:
        best_lambda = lambdas[0]
        found_best = True

    accuracies = [None]*len(lambdas)
    idxs = [None]*4

    if input_features:
        x_train, y_train = train_loader[0], train_loader[1]
        x_valid, y_valid = valid_loader[0], valid_loader[1]
        x_test, y_test = test_loader[0], test_loader[1]
    else:
        with torch.autograd.set_grad_enabled(False):
            all_features = compute_all_features(train_loader, None, valid_loader, test_loader, model,
                                                normalize=normalize, standardize=standardize)
            x_train = all_features['train_labeled']['x']
            x_valid = all_features['valid']['x']
            x_test = all_features['test']['x']
            y_train = all_features['train_labeled']['y']
            y_valid = all_features['valid']['y']
            y_test = all_features['test']['y']

    model.cpu()
    x_train, x_valid, x_test = x_train.to(defaults.device), x_valid.to(defaults.device), x_test.to(defaults.device)
    y_train, y_valid, y_test = y_train.to(defaults.device), y_valid.to(defaults.device), y_test.to(defaults.device)

    if valid_loader is not None:
        # Cross-validation over lambda
        while found_best is False:
            lambda_idx, idxs = next_lambda(accuracies, idxs)
            if accuracies[lambda_idx] is None:
                valid_acc, train_acc, valid_loss, train_loss, w = train_classifier(x_train, x_valid, y_train, y_valid,
                                                                                   lambdas[lambda_idx], nclasses,
                                                                                   maxiter=maxiter, w=w_init,
                                                                                   loss_name=loss_name)
                accuracies[lambda_idx] = valid_acc
            else:
                best_lambda = lambdas[lambda_idx]
                found_best = True
    else:
        raise NotImplementedError

    # Final training with best lambda
    accuracies = np.array(accuracies, dtype=np.float64)
    if not np.alltrue(np.isnan(accuracies)):
        valid_acc = np.nanmax(accuracies)
    else:
        valid_acc = np.nan

    test_acc, train_acc, test_loss, train_loss, w = train_classifier(x_train, x_test, y_train, y_test, best_lambda,
                                                                     nclasses, maxiter=maxiter, w=w_init,
                                                                     loss_name=loss_name)

    model.to(defaults.device)

    return test_acc, valid_acc, train_acc, test_loss, train_loss, w, best_lambda


def train_classifier(x_train, x_test, y_train, y_test, lam, nclasses, maxiter=1000, w=None, loss_name='mnl'):
    """
    Train a linear classifier on the training data and evaluate it on the given test data (if not None).

    :param x_train: Training set features
    :param x_test: Test set features
    :param y_train: Training set labels
    :param y_test: Test set labels
    :param lam: l2-regularization parameter
    :param nclasses: Number of classes in the datasets
    :param maxiter: Maximum number of iterations
    :param w: Initial parameter values for the classifier
    :param loss_name: Name of the loss function to use. Either 'square' or 'mnl'.
    :return: Tuple containing:

        * test_acc: Final accuracy on the test set
        * train_acc: Final accuracy on the training set
        * test_loss: Final loss on the test set
        * train_loss: Final loss on the training set
        * w: Trained parameters of the classifier
    """
    def obj(w):
        if w.__class__ == np.ndarray:
            w = w.reshape(x_train.shape[1] + 1, nclasses)
            w = torch.Tensor(w).to(defaults.device)
        yhat = torch.mm(x_train, w[1:, :]) + w[0, :]
        obj_val = loss(yhat, y_train) + lam * torch.norm(w[1:, :]) ** 2
        return obj_val.item()

    def grad(w):
        w = w.reshape(x_train.shape[1] + 1, nclasses)
        w = nn.Parameter(torch.Tensor(w).to(defaults.device))
        yhat = torch.mm(x_train, w[1:, :]) + w[0, :]
        obj_val = loss(yhat, y_train) + lam * torch.norm(w[1:, :]) ** 2
        obj_val.backward()
        grad = w.grad.data.detach().cpu().double().numpy().ravel()
        return grad

    if w is None:
        np.random.seed(0)
        w = np.random.normal(size=(np.size(x_train, 1) + 1, nclasses))

    if loss_name == 'square':
        y_train = one_hot_embedding(y_train, nclasses).type(torch.get_default_dtype()).to(defaults.device)
        if x_test is not None:
            y_test = one_hot_embedding(y_test, nclasses).type(torch.get_default_dtype()).to(defaults.device)
        loss = square_loss
    elif loss_name == 'mnl':
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    if maxiter > 0:
        opt = scipy.optimize.minimize(obj, w.flatten(), method='L-BFGS-B', jac=grad,
                                      options={'maxiter': maxiter, 'disp': False})
        w = opt['x'].reshape(*w.shape)

    torch.autograd.set_grad_enabled(False)
    w = nn.Parameter(torch.Tensor(w).to(defaults.device))

    train_accuracy, train_loss = compute_accuracy(x_train, y_train, w, loss)
    test_accuracy, test_loss = compute_accuracy(x_test, y_test, w, loss)

    torch.autograd.set_grad_enabled(True)

    return test_accuracy, train_accuracy, test_loss, train_loss, w


def compute_accuracy(x, y, w, loss):
    """
    Given generated features, labels, classifier parameters, and a loss function, compute the value of the given loss
    function and the accuracy.

    :param x: Features
    :param y: True labels
    :param w: Parameters of the classifier
    :param loss: Loss function to use
    """
    if x is not None:
        yhat = torch.mm(x, w[1:, :]) + w[0, :]
        loss_value = loss(yhat, y).item()
        yhat = torch.max(yhat, 1)[1]
        if hasattr(loss, '__name__') and loss.__name__ == 'square_loss':
            y = torch.max(y, 1)[1]
        accuracy = np.mean((yhat == y).cpu().data.numpy())
    else:
        accuracy = None
        loss_value = None

    return accuracy, loss_value


def square_loss(yhat, y):
    """
    Compute the square loss given the true and predicted labels.

    :param yhat: Predicted labels
    :param y: True labels
    :return Square loss
    """
    return 1/len(y)*torch.sum((y-yhat)**2)
