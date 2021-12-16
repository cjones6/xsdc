import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import scipy

from baseline.kernels_utils import one_hot_embedding


def predict_with_semi_sup_kmeans(data, max_iter=100):
    # Algo
    gram_train, gram_test = data['train']['gram'], data['test']['gram']
    y_train, y_test = data['train']['y'], data['test']['y']
    idx_labeled, n_class = data['idx_labeled'], data['n_class']
    Z = semi_sup_kmeans(gram_train, y_train, idx_labeled, n_class, max_iter)

    # Predict
    y_pred = predict(gram_test, gram_train, Z)
    y_pred = align_pred_test_lab(y_pred, y_test, n_class)
    acc_test = torch.sum(y_pred == y_test).float() / len(y_pred)
    # print('Exp done in time: {}'.format(time.time() - start_time))

    return acc_test


def semi_sup_kmeans(gram, labels, idx_labeled, n_clust, max_iter=50, verbose=True):
    # If we could define a center matrix the problem would be
    # min_{Z, C} ||X- ZC||^2
    # with X of size (n, d), Z of size (n, k) and C of size (k, d)
    # Z is the assignment matrix, the first n_lab columns are chosen to be the labeled ones (X must reflect that)
    # The distances of each point with respect to the clusters is given as a matrix of size (n, k)
    # diag(XX^T) 1_k^T - 2 X C^T + 1_n diag(CC^T)^T
    # (we ignore the constant term in the code)
    # Since we do not have access to a finite representation of the centers if we use a kernel,
    # we directly plug the current solution for the centers given as
    # C = (ZZ^T)^{-1}Z^T X
    # The computations follow using some properties of the assignment matrix such as (ZZ^T)^{-1} = diag(Z^T 1)^{-1}
    # To ensure that the known labels are well assigned, we fix them after each iteration.

    # Initialization
    k = n_clust
    n = gram.size(0)
    if len(idx_labeled) > 0:
        known_labels = one_hot_embedding(labels[idx_labeled], k)
        n_lab, k = known_labels.size(0), known_labels.size(1)
        Z = torch.zeros(n, k)
        Z[idx_labeled] = known_labels
    else:
        Z = kmeans_init(gram, k)

    # Repeat
    # first_time = True
    # start_time = time.time()
    for i in range(max_iter):
        if i % 10 == 5:
            prev_obj = compute_obj(gram, Z)

        Z_norm = Z*(1 / Z.sum(dim=0))[:, ]
        aux = gram.mm(Z_norm)
        dists = -2*aux + torch.ones(n).ger(torch.sum(Z_norm*aux, dim=0))
        Z = one_hot_embedding(dists.argmin(dim=1), k)
        if len(idx_labeled) > 0:
            Z[idx_labeled] = known_labels

        # if first_time and verbose:
        #     # print('Time for 1 iter: {}'.format(time.time()-start_time))
        #     first_time = False

        if i % 10 == 5:
            obj = compute_obj(gram, Z)
            if torch.abs(prev_obj-obj) < 1e-6:
                print('kmeans converged in {} iterations'.format(i))
                break
    return Z


def predict(new_gram, gram, Z):
    Z_norm = Z * (1 / Z.sum(dim=0))[:, ]
    aux = gram.mm(Z_norm)
    norm_centers = torch.sum(Z_norm * aux, dim=0)

    n_new = new_gram.size(0)
    Z_norm = Z * (1 / Z.sum(dim=0))[:, ]
    dists = -2*new_gram.mm(Z_norm) + torch.ones(n_new).ger(norm_centers)
    y_pred = dists.argmin(dim=1)
    return y_pred


def kmeans_init(gram, k):
    n = gram.size(0)

    Z = torch.zeros(n, 1)
    idx = torch.multinomial(torch.ones(n), 1)
    Z[idx, 0] = 1
    for i in range(1, k):
        aux = gram.mm(Z)
        dists = torch.diag(gram).ger(torch.ones(i)) - 2*aux + torch.ones(n).ger(torch.sum(Z*aux, dim=0))
        assert torch.sum(dists < 0) == 0
        probs = torch.min(dists, dim=1)[0]
        new_idx = torch.multinomial(probs, 1)
        new_Z = torch.zeros(n, 1)
        new_Z[new_idx] = 1
        Z = torch.cat([Z, new_Z], dim=1)
    return Z


def align_pred_test_lab(y_pred, y_test, k):
    # Relabel classes
    conf_matrix = confusion_matrix(y_test, y_pred)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-1 * conf_matrix)

    y_pred_aligned = -1 * np.ones_like(y_pred)
    for i in range(k):
        idxs = np.where(y_pred == i)[0]
        label = np.where(col_ind == i)[0][0]
        y_pred_aligned[idxs] = int(label)
    y_pred_aligned = torch.from_numpy(y_pred_aligned.flatten())
    return y_pred_aligned


def compute_obj(gram, Z):
    n = Z.size(0)
    norm_equiv = Z.mm(torch.diag(1 / Z.sum(dim=0)).mm(Z.t()))
    return torch.trace((torch.eye(n) - norm_equiv)*gram)


