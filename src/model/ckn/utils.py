import faiss
import numpy as np
import torch

from src import default_params as defaults


def pad(features, npad):
    """
    Pad the input features.

    :param npad: Amount of padding to add on each side
    :return padded features
    """
    padding = torch.nn.ZeroPad2d(npad)
    return padding(features)


def extract_some_patches(features, patch_dim, stride=(1, 1), patches_per_image=None, whiten=False, padding=0):
    """
    Sample patches with the specified dimension and stride from the input features. Skip that patches that are constant.

    :param features: Features from which patches should be extracted
    :param patch_dim: Array or tuple containing the height and width of the patches to be extracted
    :param stride: Stride (height, width) to use when extracting patches
    :param patches_per_image: Number of patches to extract from each image
    :param whiten: Whether to whiten the patches that are extracted
    :param padding: Amount of padding to add before extracting patches
    :return: all_patches: Patches sampled from each of the input features
    """
    if features.ndimension() == 1:
        features = features.unsqueeze(-1)
    elif features.ndimension() == 2:
        features = features.unsqueeze(-1).unsqueeze(-1)

    batch_size = features.shape[0]
    patches = images_to_patches(features, patch_dim, stride, whiten=whiten, padding=padding)
    if patches_per_image is not None:
        all_patches = []
        for i in range(batch_size):
            mean_subtracted_patches = patches[i, :] - torch.mean(patches[i, :], 1).unsqueeze(1)
            good_idxs = torch.nonzero(torch.norm(mean_subtracted_patches, 2, 1))
            if len(good_idxs) > 0:
                perm = torch.randperm(len(good_idxs)).to(defaults.device)
                shuffled_idxs = good_idxs[perm]
                sampled_patches = patches[i][shuffled_idxs[0:patches_per_image], :].squeeze()
                if sampled_patches.dim() == 1:
                    sampled_patches = sampled_patches.unsqueeze(0)
                all_patches.append(sampled_patches)
        all_patches = torch.cat(all_patches)
    else:
        all_patches = patches.contiguous().view(-1, patches.shape[-1])

    return all_patches


def images_to_patches(features, patch_dim, stride=[1, 1], padding=None, whiten=False):
    """
    Extract patches with the specified dimension and stride from the input features. Optionally, whiten the patches on
    a per-image basis.

    :param features: Features to extract patches from
    :param patch_dim: Array or tuple containing the height and width of the patches to be extracted
    :param stride: Stride (height, width) to use when extracting patches
    :param padding: Amount of padding to add before extracting patches
    :param whiten: Whether to whiten the patches that are extracted
    :return: patches: The patches from each of the input features
    """
    if features.ndimension() == 1:
        features = features.unsqueeze(-1)
    elif features.ndimension() == 2:
        features = features.unsqueeze(-1).unsqueeze(-1)

    kh, kw = patch_dim
    dh, dw = stride
    batch_size, nfilters = features.shape[0:2]
    patch_size = kh*kw*nfilters
    if padding:
        features = pad(features, padding)
    patches = features.unfold(2, int(kh), int(dh)).unfold(3, int(kw), int(dw))
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, -1, patch_size)

    if whiten:
        patches = patches.cpu()
        for i in range(features.shape[0]):
            img_patches = patches[i]
            # Center
            for j in range(nfilters):
                img_patches[:, j*kh*kw:(j+1)*kh*kw] = img_patches[:, j*kh*kw:(j+1)*kh*kw] - \
                                                      torch.mean(img_patches[:, j*kh*kw:(j+1)*kh*kw], 1).unsqueeze(1)
            # Whiten
            img_patches = zca_whiten(img_patches)
            patches[i] = img_patches
        patches = patches.to(defaults.device)

    return patches


def zca_whiten(X):
    """
    ZCA whiten the input.

    :param X: Input to be whitened
    :return X: Whitened input
    """
    X = X - torch.mean(X, 0)
    U, S, V = torch.svd(X.t())
    Smax = torch.max(S)
    sqrt_S = (S > Smax * 1e-8).type(torch.get_default_dtype())/torch.sqrt(S)
    Z = torch.mm(torch.mul(U, sqrt_S), U.t())
    X = torch.mm(X, Z)
    return X


def patches_to_images(features, dim):
    """
    Convert the resultant features from applying a layer of the CKN back to their 4D representation
    (batch size, # channels, height, width).

    :param features: 2D array of features to convert back to 4D
    :param dim: Batch size, height, and width of images
    :return features: Features in 4D image representation
    """
    batch_size, height, width = dim
    features = features.contiguous().view(batch_size, height, width, -1).permute(0, 3, 1, 2)
    return features


def spherical_kmeans(data, k, nrestarts=10, niters=100):
    """
    Run spherical k-means on the input data.

    :param data: Data to be clustered
    :param k: Number of clusters
    :param nrestarts: Number of restarts of spherical k-means to try
    :param niters: Number of iterations to perform
    :return: centroids from spherical k-means
    """
    data = np.ascontiguousarray(data.cpu().numpy()).astype('float32')
    data /= np.linalg.norm(data, axis=1)[:, np.newaxis]
    d = data.shape[1]

    clus = faiss.Clustering(d, k)
    clus.verbose = False
    clus.niter = niters
    clus.nredo = nrestarts
    clus.seed = defaults.seed
    clus.spherical = True
    index = faiss.IndexFlatIP(d)

    clus.train(data, index)
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(k, d)
    centroids = torch.Tensor(centroids).to(defaults.device)

    return centroids / torch.norm(centroids, 2, 1).unsqueeze(1)


def kmeans(data, k, nrestarts=10, niters=100):
    """
    Run k-means on the input data.

    :param data: Data to be clustered
    :param k: Number of clusters
    :param nrestarts: Number of restarts of k-means to try
    :param niters: Number of iterations to perform
    :return: centroids from k-means
    """
    data = np.ascontiguousarray(data.cpu().numpy()).astype('float32')
    d = data.shape[1]

    clus = faiss.Clustering(d, k)
    clus.verbose = False
    clus.niter = niters
    clus.nredo = nrestarts
    clus.seed = defaults.seed
    clus.spherical = False
    index = faiss.IndexFlatL2(d)

    clus.train(data, index)
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(k, d)
    centroids = torch.Tensor(centroids).to(defaults.device)

    return centroids


def random_sample(data, k):
    """
    Randomly sample k rows from the input data and normalize them.

    :param data: Data to randomly sample from
    :return data: Sampled rows, normalized so they have norm 1
    """
    idxs = np.random.choice(len(data), k, replace=False)
    landmarks = data[idxs]
    return landmarks/torch.norm(landmarks, 2, 1).unsqueeze(1)


def stable_newton_with_newton(S, maxiter=20):
    """
    Perform the intertwined Newton method to compute the matrix inverse square root S^{-1/2}.

    :param S: Matrix to compute the inverse square root of
    :return T: S^{-1/2}
    """
    T = torch.eye(*S.shape, device=defaults.device)
    Id = T.clone()
    S, scale_factor = scale_init_matrix(S)
    for i in range(maxiter):
        next = 0.5*(3*Id-T.mm(S))
        S = S.mm(next)
        T = next.mm(T)

    T = T*(scale_factor**(-1/2))
    return T


def scale_init_matrix(S):
    """
    Scale the input matrix for the intertwined Newton method.

    :param S: Matrix to be scaled
    :return Tuple containing:

        * S/nrm: Matrix divided by its Frobenius norm
        * nrm: Frobenius norm of S
    """
    nrm = torch.norm(S, 2)
    return S/nrm, nrm
