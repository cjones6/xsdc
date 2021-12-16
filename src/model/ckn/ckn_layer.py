import torch
from torch import nn

import src.default_params as defaults
from src.model.ckn import utils, pool, kernels


class CKNLayer(nn.Module):
    """
    Single layer of a CKN.

    :param layer_num: Index of the layer
    :param patch_size: Array or tuple containing the height and width of the patches to be extracted
    :param patch_kernel: Kernel to use. One of 'rbf_sphere', 'rbf', 'linear'
    :param n_filters: Number of filters to use
    :param subsampling_factor: Factor by which the height and width should be subsampled after pooling
    :param padding: Amount of padding to apply before extracting patches
    :param stride: Stride (height, width) to use when extracting patches
    :param precomputed_patches: Whether the input to the layer will be precomputed patches, so patches do not need to be
                                extracted
    :param whiten: Whether to whiten the patches that are extracted
    :param filters_init: Method to use to initialize the filters. One of 'spherical-k-means', 'k-means',
                         'random_sample', 'identity', 'random_normal', 'precomputed'
    :param normalize: Whether to normalize the patches that are extracted so they have norm 1
    :param patch_sigma: Bandwidth for the kernel (if applicable)
    :param pool_method: Method to use for pooling. One of 'average', 'max', 'rbf'
    :param pool_dim: Dimensions to use for pooling
    :param store_normalization: Whether to store the normalization k(W^TW)^{-1/2}
    :param kww_reg: Multiple of the identity to be added to k(W^TW)^{-1/2}
    :param num_newton_iters: Number of intertwined Newton iterations to perform
    """
    def __init__(self, layer_num, patch_size, patch_kernel, n_filters, subsampling_factor, padding=0, stride=(1, 1),
                 precomputed_patches=False, whiten=False, filters_init='spherical-k-means', normalize=True,
                 patch_sigma=0.6, pool_method='average', pool_dim=(1, 1), store_normalization=False, kww_reg=0.001,
                 num_newton_iters=20):
        nn.Module.__init__(self)
        self.layer_num = layer_num
        self.patch_size = patch_size
        self.patch_kernel = self._initialize_patch_kernel(patch_kernel, patch_sigma=patch_sigma)
        self.n_filters = n_filters
        self.subsampling_factor = subsampling_factor
        self.pad = padding
        self.stride = stride
        self.precomputed_patches = precomputed_patches
        self.whiten = whiten
        self.filters_init = filters_init
        self.norm = normalize
        self.pooling = pool.Pool(pool_method, subsampling_factor, pool_dim)
        self.store_normalization = store_normalization
        self.kww_reg = kww_reg
        self.num_newton_iters = num_newton_iters

        self.W = None
        self.normalization = None
        self.input_spatial_dims = None

    @staticmethod
    def _initialize_patch_kernel(patch_kernel, **kwargs):
        """
        Set the patch kernel for this layer.

        :param patch_kernel: Kernel to use. One of 'rbf_sphere', 'rbf', 'linear'
        :param **kwargs: Keyword argument(s) specific to a kernel
        :return kernel: Kernel object
        """
        if patch_kernel.lower() == 'rbf_sphere':
            kernel = kernels.RBFSphere(kwargs['patch_sigma'])
        elif patch_kernel.lower() == 'rbf':
            kernel = kernels.RBF(kwargs['patch_sigma'])
        elif patch_kernel.lower() == 'linear':
            kernel = kernels.Linear()
        else:
            raise NotImplementedError

        return kernel

    def initialize_W(self, patches):
        """
        Initialize the filters W.

        :param patches: Patches to use in the initialization
        """
        if self.filters_init == 'spherical-k-means':
            W = utils.spherical_kmeans(patches, k=self.n_filters)
        elif self.filters_init == 'k-means':
            W = utils.kmeans(patches, k=self.n_filters)
        elif self.filters_init == 'random_sample':
            W = utils.random_sample(patches, k=self.n_filters)
        elif self.filters_init == 'identity':
            W = torch.eye(patches.shape[1], patches.shape[1], device=defaults.device)
        elif self.filters_init == 'random_normal':
            W = torch.randn(patches.shape[0], patches.shape[1], device=defaults.device)
            W /= torch.norm(W, 2, 0)
            W = W.t()
        elif self.filters_init == 'precomputed':
            W = patches
        else:
            raise NotImplementedError

        self.W = nn.Parameter(W.to(defaults.device), requires_grad=True)

        if self.store_normalization:
            self.normalization = self.compute_normalization().to(defaults.device)

    @staticmethod
    def _normalize_patches(patches, eps=1e-10):
        """
        Normalize the patches so they have norm 1.

        :param patches: Patches to be normalized
        :return Tuple containing:

            * patches: Normalized patches
            * norm_patches: Norms of the original patches
        """
        norm_patches = torch.clamp(torch.norm(patches, 2, 1, keepdim=True), min=eps)
        patches = patches / norm_patches

        return patches, norm_patches.squeeze()

    def compute_normalization(self):
        """
        Compute the normalization k(W^TW)^{-1/2}.

        :return k(W^TW)^{-1/2}
        """
        basis_gram = self.patch_kernel(self.W, self.W)
        identity = torch.eye(*basis_gram.shape, device=defaults.device)
        return utils.stable_newton_with_newton(basis_gram + self.kww_reg * identity, maxiter=self.num_newton_iters)

    def _project(self, patches, normalization):
        """
        Perform the projection onto the subspace.

        :param patches: Patches to use in the projection
        :param normalization: k(W^TW)^{-1/2}
        :return k(W^TW)^{-1/2}k(W^Tpatches)
        """
        embedded = self.patch_kernel(patches, self.W)
        embedded = torch.mm(embedded, normalization)

        return embedded

    def _next_dims(self, f):
        """
        Get the dimensions of the feature representation after applying this layer.

        :param f: Batch of features
        :return batch_size: Batch size based on f
        :return height_next: Height of the feature representation after applying this layer
        :return width_next: Width of the feature representation after applying this layer
        """
        batch_size = f.shape[0]
        if self.precomputed_patches is False and f.ndimension() == 4:
            height, width = f.shape[2:4]
        elif self.precomputed_patches is False and f.ndimension() == 3:
            height, width = f.shape[2:3], 1
        elif self.precomputed_patches is False and f.ndimension() == 2:
            height, width = 1, 1
        elif self.precomputed_patches is False and f.ndimension() > 4:
            raise NotImplementedError
        else:
            height = self.input_spatial_dims[0]
            width = self.input_spatial_dims[1]
        height_next = int((height + 2 * self.pad - (self.patch_size[0] - 1) - 1) / self.stride[0] + 1)
        width_next = int((width + 2 * self.pad - (self.patch_size[1] - 1) - 1) / self.stride[1] + 1)

        return batch_size, height_next, width_next

    def forward(self, f):
        """
        Apply this layer of the CKN to the provided input features.

        :param f: Input features
        :return f_next: Output features
        """
        if self.W is None:
            raise AssertionError('You must initialize W prior to running forward()')

        if self.precomputed_patches is False:
            patches_by_image = utils.images_to_patches(f, self.patch_size, self.stride, whiten=self.whiten,
                                                       padding=self.pad)
        else:
            patches_by_image = f
        patches = patches_by_image.contiguous().view(-1, patches_by_image.shape[-1])

        if self.norm:
            patches, norm_patches = self._normalize_patches(patches)
        if self.normalization is None or self.store_normalization is False:
            normalization = self.compute_normalization().to(defaults.device)
        else:
            normalization = self.normalization.to(defaults.device)

        cutoff = 113246208  # be careful with gpu memory
        if patches.shape[0]*self.W.shape[1] < cutoff:
            projected_patches = self._project(patches, normalization)
        else:
            bsize = int(cutoff/patches.shape[1])
            idx_num = 0
            projected_patches = []
            while idx_num < patches.shape[0]:
                subpatches = patches[idx_num:idx_num+bsize]
                projected_patches.append(self._project(subpatches, normalization))
                idx_num += bsize
            projected_patches = torch.cat(projected_patches)

        if self.norm:
            if norm_patches.ndimension() > 0:
                projected_patches = projected_patches*norm_patches.unsqueeze(1)
            else:
                projected_patches = projected_patches*norm_patches

        all_patches = utils.patches_to_images(projected_patches, self._next_dims(f))

        f_next = self.pooling(all_patches)

        return f_next
