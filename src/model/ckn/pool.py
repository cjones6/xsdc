import numpy as np
import torch
import torch.nn as nn

import src.default_params as defaults


class Pool(nn.Module):
    """
    Pooling for the CKN.

    :param method: Method to use for pooling. One of 'average', 'max', or 'rbf'.
    :param subsampling_factor: Factor by which the height and width should be subsampled after pooling
    :param pool_dim: Dimensions to use for pooling
    :param pool_sigma: Bandwidth to use in the pooling if doing RBF pooling
    """
    def __init__(self, method, subsampling_factor=1, pool_dim=(1, 1), pool_sigma=None):
        super(Pool, self).__init__()
        self.method = method
        self.subsampling_factor = subsampling_factor
        self.pool_dim = pool_dim
        self.pool_sigma = pool_sigma
        self.weights = self._get_weights()

    def _get_weights(self):
        """
        Compute the weights for the pooling.

        :return weights: Weights to use in the pooling
        """
        if self.method == 'average':
            weights = 1/np.prod(self.pool_dim)*np.ones(self.pool_dim)
        elif self.method == 'RBF':
            if self.pool_dim is None:
                self.pool_dim = (self.subsampling_factor, self.subsampling_factor)
            if self.pool_sigma is None:
                self.pool_sigma = self.subsampling_factor
            if self.pool_dim != 0:
                weights = np.array(
                    [np.exp(-x ** 2 / (self.pool_sigma ** 2)) for x in range(-int(np.ceil(self.pool_dim[0])),
                                                                             int(np.ceil(self.pool_dim[1])) + 1)])
                weights = np.outer(weights / sum(weights), weights / sum(weights))
            else:
                weights = np.array([[1]])
        elif self.method == 'max':
            weights = None
        else:
            raise NotImplementedError

        if weights is not None:
            weights = torch.Tensor(weights[np.newaxis, np.newaxis, :, :]).to(defaults.device)

        return weights

    def forward(self, x):
        """
        Perform pooling on the input x.

        :param x: Features on which to perform pooling
        :return Features after pooling and subsampling
        """
        batch_size = x.shape[0]
        if self.weights is None and self.method == 'max':
            outer_approx = nn.functional.max_pool2d(x, self.pool_dim, stride=1, padding=0)
        elif self.weights is None:
            raise NotImplementedError
        elif self.weights.nelement() != 1:
            if self.method == 'RBF':
                padding = int(self.weights.shape[2] / 2)
            else:
                padding = 0

            inner_approx = x.contiguous().view(-1, x.shape[2], x.shape[3]).unsqueeze(1)
            outer_approx = nn.functional.conv2d(inner_approx, self.weights, stride=1, padding=padding)
            outer_approx = outer_approx.contiguous().view(batch_size, -1, outer_approx.shape[2], outer_approx.shape[3])
        else:
            outer_approx = x
        offset = int(np.ceil(self.subsampling_factor / 2)) - 1

        return outer_approx[:, :, offset::self.subsampling_factor, offset::self.subsampling_factor]
