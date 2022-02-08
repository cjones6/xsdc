import torch
import torch.nn as nn


class RBF(nn.Module):
    """
    RBF kernel.

    :param sigma: Bandwidth of the kernel
    """
    def __init__(self, sigma):
        super(RBF, self).__init__()
        self.sigma = sigma
        self.name = 'rbf'

    def forward(self, x, y):
        """
        Evaluate the kernel on the provided inputs.

        :param x: First batch of inputs to the kernel
        :param y: Second batch of inputs to the kernel
        :return gram: Kernel evaluated on every pair of inputs in x and y
        """
        # norm = squared_l2_norm(x, y)
        norm = torch.cdist(x, y)**2
        gram = torch.exp(-norm*1/(2*self.sigma**2))

        return gram

    
def squared_l2_norm(x, y):
    """
    Compute ||x-y||^2 for every pair of rows in x and y.

    :param x: Batch of 2D inputs
    :param y: Another batch of 2D inputs
    :return ||x-y||^2 for every pair of rows in x and y
    """
    nx = x.size(0)
    ny = y.size(0)

    norm_x = torch.sum(x ** 2, 1).unsqueeze(0)
    norm_y = torch.sum(y ** 2, 1).unsqueeze(0)

    ones_x = torch.ones(nx, 1)
    ones_y = torch.ones(ny, 1)

    a = torch.mm(ones_y, norm_x)
    b = torch.mm(x, y.t())
    c = torch.mm(ones_x, norm_y)

    return a.t() - 2 * b + c


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

    return y_one_hot.type(torch.get_default_dtype())