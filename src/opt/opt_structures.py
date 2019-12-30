import os
import pickle
import torch
import torch.nn as nn


class Data:
    """
    Class for storing the data loaders.

    :param train_labeled_loader: Dataloader for the labeled training set
    :param train_unlabeled_loader: Dataloader for the unlabeled training set
    :param valid_loader: Dataloader for the validation set
    :param train_valid_loader: Dataloader for the combined training+validation sets
    :param test_loader: Dataloader for the test set
    :param deepcluster_loader: Dataloader for the labeled+unlabeled training sets, for use with deep clustering
    """
    def __init__(self, train_labeled_loader, train_unlabeled_loader, valid_loader, train_valid_loader, test_loader,
                 deepcluster_loader=None):
        self.train_labeled_loader = train_labeled_loader
        self.train_unlabeled_loader = train_unlabeled_loader
        self.valid_loader = valid_loader
        self.train_valid_loader = train_valid_loader
        self.test_loader = test_loader
        if self.train_labeled_loader is not None:
            self.train_labeled_iter = iter(self.train_labeled_loader)
        else:
            self.train_labeled_iter = None
        if self.train_unlabeled_loader is not None:
            self.train_unlabeled_iter = iter(self.train_unlabeled_loader)
        else:
            self.train_unlabeled_iter = None
        self.deepcluster_loader = deepcluster_loader


class Model(nn.Module):
    """
    Class that stores the model, evaluates the model on inputs, saves the model, and loads the model.

    :param model: Model (object from a subclass of the nn.Module class)
    :param save_path: Where to save the model. If None, the model is not saved.
    """
    def __init__(self, model=None, save_path=None):
        super(Model, self).__init__()
        self.model = model
        self.save_path = save_path

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def forward(self, x):
        """
        Generate features for the provided inputs with the model.

        :param x: Inputs for which features should be generated
        :return: Features for inputs x
        """
        return self.model(x)

    def save(self, **kwargs):
        """
        Save the model to a file. It assumes that the extension is '.pickle'.

        :param kwargs: Additional keyword arguments to save along with the model
        """
        if self.save_path is not None:
            save_dict = {'model': self.model}
            for key, value in kwargs.items():
                save_dict[key] = value
            torch.save(save_dict, self.save_path[:-7] + '_' + str(kwargs['iteration']) + self.save_path[-7:])

    def load(self, path):
        """
        Load the model from a file.

        :param path: Filepath of the model
        :return: model_dict: Dictionary with the model and anything else stored with it
        """
        model_dict = torch.load(path)
        self.model = model_dict['model']
        return model_dict


class Params:
    """
    Class to store a variety of hyperparameter values and input options.

    :param nclasses: Number of classes in the dataset
    :param min_frac_points_class: Minimum fraction of points in a single class
    :param max_frac_points_class: Maximum fraction of points in a single class
    :param ckn: Whether the model is a CKN
    :param convnet: Whether the model is a ConvNet
    :param project: Whether to project the filters onto the unit sphere
    :param train_w_layers: List of layers to train. If None, it trains the parameters of all of the layers.
    :param lambda_filters: Value of the l2 penalty on the filters
    :param lambda_pix: Value of the l2 penalty on the centered features
    :param lam: Value of the l2 penalty on the parameters of the classifier
    :param standardize: Whether to standardize the features that are output from the network
    :param normalize: Whether to normalize the features that are output from the network
    :param add_constraints: Whether additional constraints related to the unlabeled training data should be added
    :param add_constraints_method: How to add the additional constraints. One of 'random' (random correct constraints
                                   that don't give away the true label) or 'specific' (constraints derived from
                                   knowledge of whether each observation belongs to some set of labels or not)
    :param add_constraints_frac: Fraction of unknown entries that should be made known in the 'random' constraints
                                 method mentioned above
    :param add_constraints_classes: Classes to use for the 'specific' constraints mentioned above
    :param nn: Number of nearest neighbors to use when labeling the unlabeled data prior to evaluating the performance
    :param balanced: Whether to use the balanced version of the matrix balancing algorithm
    :param only_sup: Whether to perform only supervised learning and ignore any unlabeled data
    :param w_last_init: Initial values of the coefficients of the output layer
    :param b_last_init: Initial values of the biases of the output layer
    :param labeling_method: Labeling method. One of 'matrix balancing', 'pseudo-labeling', or 'deep clustering'.
    :param deepcluster_k: Number of clusters to use when performing the clustering step of deep clustering
    :param deepcluster_update_clusters_every: Number of iterations between cluster updates in deep clustering
    :param labeling_burnin: Number of iterations for which learning on only the labeled data should be performed prior
                            to learning with the labeled+unlabeled data
    :param step_size_init_sup: Initial step size when training on the labeled data only
    :param step_size_init_semisup: Initial step size when training on the labeled+unlabeled data
    :param update_lambda: Whether to update lambda every 100 iterations
    :param momentum: Amount of momentum to use when optimizing a ConvNet
    :param maxiter_wlast: Maximum number of iterations when optimizing the classifier
    :param maxiter: Maximum number of iterations
    :param eval_test_every: How often to evaluate the performance (after how many iterations)
    :param save_every: How often to save the model, parameters, and results (after how many iterations)
    :param save_path: Location where the parameters should be saved. If None, they aren't saved
    """
    def __init__(self, nclasses=10, min_frac_points_class=None, max_frac_points_class=None, ckn=True, convnet=False,
                 project=False, train_w_layers=None, lambda_filters=0, lambda_pix=0, lam=None, standardize=False,
                 normalize=False, add_constraints=False, add_constraints_method='random', add_constraints_frac=None,
                 add_constraints_classes=0, nn=1, balanced=True, only_sup=False, w_last_init=None, b_last_init=None,
                 labeling_method='matrix balancing', deepcluster_k=None, deepcluster_update_clusters_every=None,
                 labeling_burnin=0, step_size_init_sup=None, step_size_init_semisup=None, update_lambda=True,
                 momentum=0, maxiter_wlast=1000, maxiter=1000, eval_test_every=10, save_every=100, save_path=None):

        self.nclasses = nclasses
        self.standardize = standardize
        self.normalize = normalize
        self.w_last_init = w_last_init
        self.b_last_init = b_last_init
        self.step_size_init_sup = step_size_init_sup
        self.step_size_init_semisup = step_size_init_semisup
        self.maxiter_wlast_full = maxiter_wlast
        self.maxiter_final = maxiter
        self.lam = lam
        self.save_path = save_path
        self.eval_test_every = eval_test_every
        self.save_every = save_every
        self.labeling_method = labeling_method
        self.deepcluster_k = deepcluster_k if deepcluster_k is not None else self.nclasses
        self.deepcluster_update_clusters_every = deepcluster_update_clusters_every
        self.train_w_layers = train_w_layers
        self.lambda_pix = lambda_pix
        self.lambda_filters = lambda_filters
        self.labeling_burnin = labeling_burnin
        self.project = project
        self.ckn = ckn
        self.convnet = convnet
        self.nn = nn
        self.update_lambda = update_lambda
        self.momentum = momentum
        self.balanced = balanced
        self.min_frac_points_class = min_frac_points_class
        self.max_frac_points_class = max_frac_points_class
        self.only_sup = only_sup
        self.add_constraints = add_constraints
        self.add_constraints_method = add_constraints_method
        self.add_constraints_classes = add_constraints_classes
        self.add_constraints_frac = add_constraints_frac

        assert self.labeling_method in ['matrix balancing', 'pseudo labeling', 'deep clustering'], \
            self.labeling_method + " is not a valid labeling method. It should be one of 'matrix balancing', 'pseudo " \
                                   "labeling', 'deep clustering'"

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def save(self):
        """
        Save the parameters to a file.
        """
        save_dir = os.path.join(*self.save_path.split(os.sep)[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def load(self, path):
        """
        Load the parameters from a file.

        :param path: Filepath of the parameters
        """
        params = pickle.load(open(path, 'rb'))
        for key, value in params.items():
            self.__dict__[key] = value


class Results:
    """
    Class to store the results from each iteration.

    :param save_path: Where to save the results. If None, they are not saved.
    """
    def __init__(self, save_path=None):
        self.train_accuracy = {}
        self.train_labeled_accuracy = {}
        self.train_unlabeled_accuracy = {}
        self.valid_accuracy = {}
        self.test_accuracy = {}
        self.train_loss = {}
        self.train_labeled_loss = {}
        self.train_unlabeled_loss = {}
        self.valid_loss = {}
        self.test_loss = {}
        self.epoch_time = {}
        self.step_size = {}
        self.save_path = save_path

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def update(self, iteration, **kwargs):
        """
        Update the dictionaries of results.

        :param iteration: Iteration number
        :param kwargs: Additional keyword arguments to save
        """
        for key, value in kwargs.items():
            self.__dict__[key][iteration] = value

    def save(self):
        """
        Save the current results.
        """
        if self.save_path is not None:
            pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def load(self, path):
        """
        Load results from a file.

        :param path: Filepath of the results
        """
        params = pickle.load(open(path, 'rb'))
        for key, value in params.items():
            self.__dict__[key] = value
