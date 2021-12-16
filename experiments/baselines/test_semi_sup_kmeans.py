import seaborn as sns
import torch
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans

from baseline.kernels_utils import RBF
from baseline.get_data import make_synth_data, compute_gram
from baseline.semi_sup_kmeans import semi_sup_kmeans, predict, align_pred_test_lab, compute_obj


def visualize_semi_sup_kmeans(k=3, n_lab_per_grp=10, n_unlab_per_grp=20, n_test_per_grp=20,
                              dist_clust=2., radius=1., radius_unlab=1., d=2,
                              kernel='linear', visualize=False, compa_skl=False):
    # Data params
    n_lab = n_lab_per_grp * k
    n_unlab = n_unlab_per_grp * k
    n_test = n_test_per_grp * k
    if kernel == 'rbf':
        kernel = RBF(1.)

    # Generate data
    if n_lab > 0:
        X_lab, y_lab = make_synth_data(dist_clust, n_lab, k, d, radius)
        if visualize:
            plot_clusters(X_lab, y_lab)
    else:
        y_lab = []
    X_unlab, _ = make_synth_data(dist_clust, n_unlab, k, d, radius_unlab)
    X_test, y_test = make_synth_data(dist_clust, n_test, k, d, radius)
    X = torch.cat((X_lab, X_unlab)) if n_lab > 0 else X_unlab

    # Compute kmeans
    gram = compute_gram(X, X, kernel)
    Z = semi_sup_kmeans(gram, y_lab, np.arange(n_lab), k, max_iter=300)
    if visualize:
        plot_clusters(X, Z.argmax(dim=1))

    # Predict
    new_gram = compute_gram(X_test, X, kernel)
    y_pred = predict(new_gram, gram, Z)
    y_pred = align_pred_test_lab(y_pred, y_test, k)
    test_acc = 100 * torch.sum(y_pred == y_test).float() / y_pred.size(0)
    print('test accuracy: {:2.2f}%'.format(test_acc))

    if n_lab == 0 and compa_skl:
        # With sklearn
        X_skl = X.numpy()
        kmeans = KMeans(n_clusters=k, ).fit(X_skl)
        lab_skl = kmeans.labels_
        Z_skl = np.zeros(((lab_skl.size), lab_skl.max()+1))
        Z_skl[np.arange(lab_skl.size), lab_skl] = 1
        Z_skl = torch.from_numpy(Z_skl)
        if visualize:
            plot_clusters(X, lab_skl)

        # Compare
        y_pred = align_pred_test_lab(Z.argmax(dim=1), lab_skl, k)

        print('nb of different labels {}'.format(torch.sum(y_pred != torch.from_numpy(lab_skl).type(torch.long))))
        print('objective our implem: {0}\nobjective sklearn {1}'.format(compute_obj(gram, Z), compute_obj(gram, Z_skl)))
    return test_acc


def increasing_lab():
    n_labs = [5*i for i in range(50)]
    k, d, n_unlab, n_test = 3, 2, 100, 50
    dist_clust, radius, radius_unlab = 1., 0.5, 2.

    test_accs = []
    for n_lab in n_labs:
        test_acc = visualize_semi_sup_kmeans(k, n_lab, n_unlab, n_test, dist_clust, radius, radius_unlab, d)
        test_accs.append(test_acc)

    plt.figure()
    plt.plot(n_labs, test_accs)
    plt.show()


def plot_clusters(X, y):
    plt.figure()
    df = DataFrame(dict(x=X[:, 0].tolist(), y=X[:, 1].tolist(), lab=y))
    sns.scatterplot(x='x', y='y', hue='lab', data=df)
    plt.show()


if __name__ == '__main__':
    increasing_lab()
    # visualize_semi_sup_kmeans()



