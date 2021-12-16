import torch

from src import default_params as defaults
from . import label_utils


def ulr_square_loss_y(features, y_one_hot, lam, penalize=True):
    """
    Compute the square loss objective given the features and labels.

    :param features: Features output by the network
    :param y_one_hot: Labels for the given features
    :param lam: l2-regularization parameter for the loss function
    :return: Tuple containing:

        * obj: The objective value with the given features and labels
        * w_last: Estimated coefficients of the output layer
        * b_last: Estimated biases of the output layer
    """
    n, d = features.shape
    xpiy = features.t().mm(label_utils.centering(y_one_hot))
    inv_term = features.t().mm(label_utils.centering(features)) + n*lam*torch.eye(d, device=defaults.device)
    wlast, _ = torch.solve(xpiy, inv_term)
    obj = 1 / n * (torch.sum((label_utils.centering(y_one_hot)) ** 2) - torch.trace(xpiy.t().mm(wlast)))
    w_last = wlast.detach()
    b_last = torch.mean(y_one_hot - features.mm(wlast), 0)

    return obj, w_last, b_last


def ulr_square_loss_m(features, M, lam):
    """
    Compute the square loss objective given the features and equivalence matrix M=YY^T.

    :param features: Features output by the network
    :param M: The equivalence matrix YY^T
    :param lam: l2-regularization parameter for the loss function
    :return: obj: The objective value with the given features and labels
    """
    n, d = features.shape
    pi = torch.eye(n, device=defaults.device) - 1.0 / n * torch.ones(n, n, device=defaults.device)
    PiX = label_utils.centering(features)
    G = PiX.mm(PiX.t())
    A = n*lam*torch.eye(n, device=defaults.device) + G
    A = torch.solve(pi, A)[0]
    A = lam*pi.mm(A)
    obj = torch.sum(M * A)

    return obj
