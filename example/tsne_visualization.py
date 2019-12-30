from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from skimage.transform import resize
from sklearn.manifold import TSNE


def compute_embedding(features, perplexity=30.0, seed=0):
    """
    Compute a t-SNE embedding for the input features
    :param features: Input features on which to perform t-SNE
    :param perplexity: Perplexity parameter for the embedding
    :param seed: Seed for reproducibility
    :return: embedded: t-SNE embeddings
    """
    features = features.cpu().detach().numpy()
    embedded = TSNE(perplexity=perplexity, random_state=seed).fit_transform(features)
    return embedded


class MplColorHelper:
    """
    Given a colormap, get equally-spaced colors.
    Source: https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an

    :param cmap_name: Name of the colormap
    :param start_val: Starting value for normalization of the colors to [0,1]
    :param stop_val: Ending value for normalization of the colors to [0,1]
    """
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def display_color(x, images, labels, save_file=None):
    """
    Display color versions of the MNIST digits in a grid determined by the t-SNE projections. This code is translated
    and adapted from Andrej Karpathy's Matlab code: https://cs.stanford.edu/people/karpathy/cnnembed/

    :param x: t-SNE projections of images
    :param images: Images to be displayed at each location
    :param labels: Image labels
    :param save_file: File path where the resultant image should be saved. If None, it displays to the screen.
    """
    COL = MplColorHelper('nipy_spectral_r', 0, 10)

    N = len(images)
    S = 2000  # Size of final image
    G = np.ones((S, S, 3))  # Placeholder for RGB pixels in final image
    s = 50  # Size of every image thumbnail
    for i in range(N):
        a = np.ceil(x[i, 0] * (S - s) + 1)
        b = np.ceil(x[i, 1] * (S - s) + 1)
        a = a - ((a - 1) % s)
        b = b - ((b - 1) % s)
        a, b = int(a-1), int(b-1)
        if G[a, b, 0] != 1:  # Check if the spot is already filled
            continue
        I = images[i].permute(1, 2, 0).numpy()
        I = resize(I, (s, s))
        J = np.ones((s, s, 3))
        if I.shape[2] == 1:
            eps = 0.0001
            idx = np.where(I < eps)
            J[idx[0], idx[1], :] = [1, 1, 1]
            idx = np.where(I > eps)
            J[idx[0], idx[1], :] = COL.get_rgb(labels[i])[0:3]

        G[a:(a + s), b:(b + s)] = J

    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(G)

    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
