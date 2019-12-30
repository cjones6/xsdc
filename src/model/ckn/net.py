import numpy as np
import torch
import torch.nn as nn

import src.default_params as defaults
from src.model.ckn import utils


class CKN(nn.Module):
    """
    Convolutional kernel network.

    :param layers: List of CKN Layer objects
    :param input_spatial_dims: Spatial dimensions of the inputs (needed if the inputs are precomputed patches)
    :param featurize: Function to apply after loading inputs and before extracting patches
    """
    def __init__(self, layers, input_spatial_dims=None, featurize=None):
        super(CKN, self).__init__()
        self.layers = nn.Sequential(*layers)
        self.featurize = featurize
        self.layers[0].input_spatial_dims = input_spatial_dims

    def _sample_patches(self, data_loader, layer, patches_per_image, patches_per_layer):
        """
        Sample patches from the feature representations of images at the specified layer.

        :param data_loader: Data loader to use to obtain images from which patches will be extracted
        :param layer: Layer number of the CKN at which patches should be extracted
        :param patches_per_image: Maximum number of patches that should be extracted per image
        :param patches_per_layer: Number of patches that should be extracted at the given layer
        :return: Sampled patches
        """
        nimages = 0
        all_patches = []
        for (x, y, _) in data_loader:
            nimages += len(x)
            x = x.to(defaults.device)
            if self.featurize is not None:
                x = self.featurize(x)
            x = x.type(torch.get_default_dtype())

            if not (layer == 0 and self.layers[0].precomputed_patches):
                for layer_num in range(layer):
                    if layer_num == 0:
                        x = nn.Parameter(x)
                    x = self.layers[layer_num](x)
                patches = utils.extract_some_patches(x, self.layers[layer].patch_size, self.layers[layer].stride,
                                                     patches_per_image, whiten=self.layers[layer].whiten)
            else:
                patches = x.contiguous().view(-1, x.shape[-1])
            patches = patches.data.cpu()
            all_patches.append(patches)

            if nimages > patches_per_layer:
                break

        return torch.cat(all_patches)[:patches_per_layer]

    def _get_filters_dim(self, layer_num, data_loader):
        """
        Get the dimensions of the filters at the specified layer.

        :param layer_num: Layer number of the CKN where the filter dimensions are desired
        :param data_loader: Data loader from which one batch will be used to help determine the dimensions
        :return Tuple containing:

            * dim1: Dimension of each filter at the layer_num'th layer
            * dim2: Number of filters at the layer_num'th layer
        """
        if layer_num == 0:
            x, y = next(iter(data_loader))
            x = x.to(defaults.device)
            if self.featurize is not None:
                x = self.featurize(x)
            if not self.layers[layer_num].precomputed_patches:
                dim1 = x.shape[1]*np.prod(self.layers[layer_num].patch_size)
            else:
                dim1 = x.shape[2]*np.prod(self.layers[layer_num].patch_size)
        else:
            dim1 = self.layers[layer_num - 1].n_filters * self.layers[layer_num].patch_size[0] * \
                   self.layers[layer_num].patch_size[1]
        dim2 = self.layers[layer_num].n_filters

        return dim1, dim2

    def init(self, data_loader, patches_per_image=1, patches_per_layer=10000, layers=None):
        """
        Initialize the weights of the CKN at each specified layer.

        :param data_loader: Data loader containing observations to use to initialize the weights of the CKN
        :param patches_per_image: Maximum number of patches that should be extracted per image
        :param patches_per_layer: Number of patches that should be extracted at the given layer
        :param layers: Layers whose weights should be initialized. If layers is not specified, it initializes the
                       weights at every layer.
        """
        nlayers = len(self.layers)
        if layers is None:
            layers = range(nlayers)
        for layer_num in layers:
            print('Initializing layer', layer_num)
            if self.layers[layer_num].filters_init in ['spherical-k-means', 'k-means', 'random_sample']:
                patches = self._sample_patches(data_loader, layer_num, patches_per_image, patches_per_layer)
                self.layers[layer_num].initialize_W(patches)
            else:
                filters_dim = self._get_filters_dim(layer_num, data_loader)
                self.layers[layer_num].initialize_W(torch.zeros(filters_dim))

            if defaults.device.type == 'cuda':
                torch.cuda.empty_cache()

    def forward(self, x):
        """
        Run x through the CKN to generate its feature representation.

        :param x: Batch of inputs to the CKN
        :return Output from evaluating the CKN on the batch of inputs x
        """
        if self.featurize is not None:
            x = self.featurize(x).type(torch.get_default_dtype())

        return self.layers(x)
