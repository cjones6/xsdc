import torch
import torch.nn as nn

from src import default_params as defaults


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
        norm = squared_l2_norm(x, y)
        gram = torch.exp(-norm*1/(2*self.sigma**2))

        return gram


class RBFSphere(nn.Module):
    """
    RBF kernel on the sphere.

    :param sigma: Bandwidth of the kernel
    """
    def __init__(self, sigma):
        super(RBFSphere, self).__init__()
        self.gamma = 1.0/sigma
        self.name = 'rbf_sphere'

    def forward(self, x, y):
        """
        Evaluate the kernel on the provided inputs.

        :param x: First batch of inputs to the kernel
        :param y: Second batch of inputs to the kernel
        :return gram: Kernel evaluated on every pair of inputs in x and y
        """
        z = torch.mm(x, y.t())
        gram = torch.exp(-self.gamma**2 * (1-z))
        return gram


class Linear(nn.Module):
    """
    Linear kernel.
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.name = 'linear'

    def forward(self, x, y):
        """
        Evaluate the kernel on the provided inputs.

        :param x: First batch of inputs to the kernel
        :param y: Second batch of inputs to the kernel
        :return Kernel evaluated on every pair of inputs in x and y
        """
        return torch.mm(x, y.t())


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

    ones_x = torch.ones(nx, 1).to(defaults.device)
    ones_y = torch.ones(ny, 1).to(defaults.device)

    a = torch.mm(ones_y, norm_x)
    b = torch.mm(x, y.t())
    c = torch.mm(ones_x, norm_y)

    return a.t() - 2 * b + c
