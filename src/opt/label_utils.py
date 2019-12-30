import faiss
import math
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
import torch

from src import default_params as defaults


def optimize_labels(X, k, lam, mask, known_values, nmin=None, nmax=None, use_cpu=False):
    """
    Given a matrix X of features and the number of classes, optimize over the labels of the unknown observations.

    :param X: 2D array of features
    :param k: Number of classes
    :param lam: Penalty on the l2 norm of the weights in the loss function
    :param mask: Binary matrix with value 0 in entry (i,j) if it is known whether i and j belong to the same class and 1
                 else
    :param known_values: Binary matrix with value 1 in entry (i,j) if it is known that i and j belong to the same class
                         and 0 else
    :param nmin: Minimum number of points in a class
    :param nmax: Maximum number of points in a class
    :param use_cpu: Whether to perform the computations on the CPU
    :return: M: Estimated equivalence matrix YY^T
    """
    n, d = X.shape
    if nmin is None or nmax is None:
        nmin = nmax = n/k
    if use_cpu:
        orig_device = defaults.device
        defaults.device = torch.device('cpu')
        X = X.cpu()

    PiX = centering(X)
    inv_term = X.t().mm(PiX) + n*lam*torch.eye(d, device=defaults.device)
    M = 1/n*(centering(torch.eye(n, device=defaults.device)) - PiX.mm(torch.solve(PiX.t(), inv_term)[0]))

    M0 = torch.ones(n, n, device=defaults.device)/k
    done = 0
    mu_factor = 1
    while not done:
        try:
            M = matrix_balancing(M, mask, known_values, nmin, nmax, M0, mu_factor=mu_factor)
            done = 1
        except:
            mu_factor *= 2
            if mu_factor > 2**10:
                raise ValueError

    if use_cpu:
        defaults.device = orig_device

    return M


def centering(X):
    """
    Center the matrix X. I.e., compute (I-11^T/n)X.

    :param X: Matrix to be centered
    :return: (I-11^T/n)X
    """
    PiX = X-torch.mean(X, dim=0, keepdim=True)
    return PiX


def matrix_balancing(Q, mask, known_values, nmin, nmax, Y0, mu_factor=1, num_iter=10):
    """
    Perform matrix balancing on Q.

    :param Q: Matrix of weights for M=YY^T in the matrix balancing algorithm
    :param mask: Binary matrix with value 0 in entry (i,j) if it is known whether i and j belong to the same class and 1
                 else
    :param known_values: Binary matrix with value 1 in entry (i,j) if it is known that i and j belong to the same class
                         and 0 else
    :param nmax: Maximum number of points in a class
    :param nmin: Minimum number of points in a class
    :param Y0: Initial guess for the solution
    :param mu_factor: Factor by which the median absolute entry in Q should be multiplied to obtain the value of the
                      entropic regularization parameter
    :param nmin: Minimum number of points in a class
    :param nmax: Maximum number of points in a class
    :param num_iter: Number of iterations to perform
    :return: M: Estimated equivalence matrix YY^T
    """
    n = Q.shape[0]
    mu = torch.median(torch.abs(Q))*mu_factor

    Q.div_(-1*mu).exp_().mul_(Y0)

    u = torch.ones(n, device=defaults.device)
    v = torch.ones(n, device=defaults.device)

    known_values = known_values.to(defaults.device)
    mask = mask.to(defaults.device)

    for t in range(num_iter):
        exp_minus_qtilde_lam = 1/u.unsqueeze(1)*1/v.unsqueeze(0)*known_values
        exp_minus_qtilde_lam[mask] = Q[mask]

        temp = (exp_minus_qtilde_lam).mv(v)
        u = (torch.clamp(temp, nmin, nmax))/temp
        temp = ((exp_minus_qtilde_lam.t()).mv(u))
        v = (torch.clamp(temp, nmin, nmax))/temp

    if math.isnan(torch.norm(u).item()) or math.isnan(torch.norm(v).item()):
        raise ValueError

    exp_minus_qtilde_lam = 1/u.unsqueeze(1)*1/v.unsqueeze(0)*known_values
    exp_minus_qtilde_lam[mask] = Q[mask]
    M = exp_minus_qtilde_lam*v
    M.mul_(u.unsqueeze(1))

    return M


def laplacian_eigenvectors(M, k, eps=1e-10):
    """
    Compute the eigenvectors corresponding to the smallest k eigenvalues of the normalized graph Laplacian
    L = I - D^{-1/2}MD^{-1/2} and then normalize the rows of the resulting matrix to have norm 1.

    :param M: Similarity matrix
    :param k: Number of eigenvectors to use
    :param eps: Minimum value for the norm of the rows
    :return: U: Matrix containing the resultant normalized eigenvectors
    """
    M = M.to(defaults.device)
    D_inv = torch.sqrt(1/M.sum(1))

    # Compute the normalized graph Laplacian L = I - D^{-1/2}MD^{-1/2}
    M.mul_(D_inv).mul_(-1*D_inv.unsqueeze(1))
    n = M.shape[0]
    ones = torch.ones(n, device=defaults.device)
    M.as_strided([n], [n + 1]).add_(ones)

    # Compute the eigenvectors corresponding to the smallest k eigenvalues and then normalize the rows
    v0 = np.zeros(len(M))
    v0[0] = 1  # Inputting v0 makes the output deterministic
    lam, U = scipy.sparse.linalg.eigsh(M.cpu().numpy(), v0=v0, k=k, which='SM')
    U = torch.Tensor(U).to(defaults.device)
    U = U/torch.clamp(torch.norm(U, 2, 1, keepdim=True), min=eps)

    return U


def kmeans(data, nclasses):
    """
    Run k-means on the input data and return the resultant class labels.

    :param data: Data to be clustered
    :param nclasses: Number of classes
    :return: labels: Class labels obtained from k-means
    """
    d = data.shape[1]
    clus = faiss.Clustering(d, nclasses)
    clus.verbose = False
    clus.niter = 100
    clus.nredo = 10
    clus.seed = defaults.seed
    clus.spherical = False

    if defaults.device.type == 'cuda':
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        index_flat = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        index.device = 0
    else:
        index = faiss.IndexFlatL2(d)

    clus.train(data, index)
    _, labels = index.search(data, 1)

    return labels


def get_estimated_labels(similarity_matrix, true_labels, nclasses):
    """
    Run spectral clustering on similarity_matrix to obtain class labels. Then relabel the class names based on the true
    labels in order to be able to assess the performance.

    :param similarity_matrix: Similarity matrix to use in spectral clustering
    :param true_labels: True labels corresponding to the rows of the similarity matrix
    :param nclasses: Number of classes in the dataset
    :return: labels_new: The estimated label for each row of the similarity matrix
    """
    similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.t())

    # Label observations
    eigs = laplacian_eigenvectors(similarity_matrix, k=nclasses)
    data = np.ascontiguousarray(eigs.cpu().numpy()).astype('float32')
    labels = kmeans(data, nclasses)

    # Relabel classes
    conf_matrix = confusion_matrix(true_labels.cpu(), labels)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-1 * conf_matrix)

    labels_new = -1 * np.ones_like(labels)
    for i in range(nclasses):
        idxs = np.where(labels == i)[0]
        label = np.where(col_ind == i)[0][0]
        labels_new[idxs] = int(label)
    labels_new = labels_new.flatten()

    return labels_new
