"""
Train a LeNet-5 CKN on the MNIST dataset
"""

import argparse
import numpy as np
import os
import random
import sys
import time
import torch

sys.path.append('../..')

from src.opt import opt_structures, train_xsdc
from src.model.ckn import parse_config, net
from src import default_params as defaults
import src.data_loaders.mnist as mnist

# Parameters for the model, data, and training
parser = argparse.ArgumentParser(description='LeNet-5 CKN training on the MNIST dataset')
parser.add_argument('--augment', default=0, type=int,
                    help='Whether to perform data augmentation on each batch')
parser.add_argument('--balanced', default=1, type=int,
                    help='Whether the unlabeled data should be balanced (1) or not (0)')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='Batch size for the validation and test data')
parser.add_argument('--batch_size_labeled', default=1024, type=int,
                    help='Batch size for the labeled training data')
parser.add_argument('--batch_size_unlabeled', default=1024, type=int,
                    help='Batch size for the unlabeled training data')
parser.add_argument('--constraints', default=None, type=str,
                    help="Type of constraints to add")
parser.add_argument('--data_path', default='../../data/mnist', type=str,
                    help='Location of the MNIST dataset')
parser.add_argument('--eval_test_every', default=10, type=int,
                    help='Number of iterations between evaluations of the performance on the test set')
parser.add_argument('--gpu', default='1', type=str,
                    help='Which GPU to use')
parser.add_argument('--imbalance', default=-1, type=float,
                    help='Imbalance of the unlabeled data. If -1, the unlabeled data will be balanced. Else, it should'
                         'be a value between 0 and 1 denoting the fraction of digits 0-4 in the unlabeled dataset.')
parser.add_argument('--labeling_burnin', default=100, type=int,
                    help='Number of iterations to perform on the labeled data prior to using the unlabeled data')
parser.add_argument('--labeling_method', default='matrix balancing', type=str,
                    help="Method to use for labeling unlabeled observations. One of 'matrix balancing',"
                         "'pseudo labeling', or 'deep clustering'.")
parser.add_argument('--lam', default=None, type=int,
                    help='log2(l2 penalty on classifier parameters)')
parser.add_argument('--lr_semisup', default=-4, type=int,
                    help='log2(Learning rate for the semi-supervised learning)')
parser.add_argument('--lr_sup_init', default=-4, type=int,
                    help='log2(Learning rate for the supervised initialization)')
parser.add_argument('--max_frac_points_class', default=0.1, type=float,
                    help='Maximum fraction of points per class')
parser.add_argument('--min_frac_points_class', default=None, type=float,
                    help='Minimum fraction of points per class')
parser.add_argument('--num_clusters', default=10, type=int,
                    help='Number of clusters to use in deep clustering')
parser.add_argument('--num_filters', default=32, type=int,
                    help='Number of filters per layer in the model')
parser.add_argument('--num_iters', default=500, type=int,
                    help='Number of total iterations to perform')
parser.add_argument('--num_labeled', default=50, type=int,
                    help='Fraction of data that is labeled')
parser.add_argument('--rounding', default=None, type=str,
                    help='Rounding to perform after the labeling method (for matrix balancing or eigendecomposition)')
parser.add_argument('--save_every', default=50, type=int,
                    help='Number of iterations between saves of the model and results')
parser.add_argument('--save_path', default='../../results/mnist_lenet5/temp', type=str,
                    help='File path prefix to use to save the model and results')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--update_clusters_every', default=50, type=int,
                    help='Number of iterations between cluster updates in deep clustering')
parser.add_argument('--update_lambda', default=1, type=int,
                    help='Whether the penalty on the classifier weights should be updated every 100 iterations (1) or '
                         'not (0)')

args = parser.parse_args()

# Set miscellaneous variables based on the inputs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.lr_sup_init = 2**args.lr_sup_init
args.lr_semisup = 2**args.lr_semisup
args.lam = 2**args.lam if args.lam is not None else None
if args.num_labeled == 0:
    args.balanced = False
print(args)

if args.imbalance > 0:
    balanced_version = False
    if args.min_frac_points_class is None:
        args.min_frac_points_class = 0.2-args.max_frac_points_class
    min_frac_points_class = args.min_frac_points_class
    max_frac_points_class = args.max_frac_points_class
    print('Bounds on the fraction of points per class:', min_frac_points_class, max_frac_points_class)
else:
    balanced_version = True
    min_frac_points_class = 1.0/args.num_clusters
    max_frac_points_class = 1.0/args.num_clusters

add_constraints_method = args.constraints
add_constraints = True if add_constraints_method is not None else False
if add_constraints_method == 'specific':
    add_constraints_classes = [4, 9]
    add_constraints_frac = 0
elif add_constraints_method == 'random':
    add_constraints_classes = []
    add_constraints_frac = 0.33
else:
    add_constraints_classes = []
    add_constraints_frac = 0

nclasses = 10
bw = 0.6
ckn = True

# Create the data loaders
train_loader, _, train_labeled_loader, train_unlabeled_loader, valid_loader, train_valid_loader, test_loader = \
    mnist.get_dataloaders(batch_size=args.batch_size, batch_size_labeled=args.batch_size_labeled,
                          batch_size_unlabeled=args.batch_size_unlabeled, num_labeled=args.num_labeled,
                          stratified_sampling=False, stratified_unlabeling=args.balanced,
                          unlabeled_imbalance=args.imbalance, data_path=args.data_path, num_workers=0, seed=args.seed)

save_dir = args.save_path
save_file = save_dir + str(args.num_labeled) + '_' + str(args.imbalance) + '_' + str(args.num_filters) + '_' + str(bw) \
            + '_' + str(args.lr_sup_init) + '_' + str(args.lr_semisup) + '_' + \
            '-'.join(str(args.labeling_method).split(' ')) + '_' + str(args.lam) + '_0_' + \
            str(args.max_frac_points_class) + '_' + str(args.min_frac_points_class) + '_' + str(args.num_clusters) + \
            '_' + str(args.update_clusters_every) + '_' + str(args.seed) + '_' + str(time.time())

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load and initialize the model
params = parse_config.load_config('../../cfg/lenet-5_ckn.cfg')
if args.num_filters > 0:
    nlayers = len(params['num_filters'])
    params['num_filters'] = [args.num_filters] * nlayers
    params['patch_sigma'] = [bw] * nlayers

layers = parse_config.create_layers(params)
model = net.CKN(layers).to(defaults.device)
model.init(train_loader)
print('Done with initialization')

# Set up the data, parameters, model, results, and optimizer objects
if args.num_labeled > 0:
    data = opt_structures.Data(train_labeled_loader, train_unlabeled_loader, valid_loader, train_valid_loader,
                               test_loader, deepcluster_loader=train_loader)
else:
    data = opt_structures.Data(None, train_unlabeled_loader, None, None, test_loader, deepcluster_loader=train_loader)
params = opt_structures.Params(nclasses=nclasses, min_frac_points_class=min_frac_points_class,
                               max_frac_points_class=max_frac_points_class, ckn=ckn, project=True,
                               train_w_layers=[0, 2, 4, 5], lam=args.lam, normalize=True, augment=args.augment,
                               add_constraints=add_constraints, add_constraints_method=add_constraints_method,
                               add_constraints_frac=add_constraints_frac,
                               add_constraints_classes=add_constraints_classes,
                               balanced=balanced_version, labeling_method=args.labeling_method, rounding=args.rounding,
                               deepcluster_k=args.num_clusters,
                               deepcluster_update_clusters_every=args.update_clusters_every,
                               labeling_burnin=args.labeling_burnin, step_size_init_sup=args.lr_sup_init,
                               step_size_init_semisup=args.lr_semisup, update_lambda=args.update_lambda,
                               maxiter=args.num_iters, eval_test_every=args.eval_test_every, save_every=args.save_every,
                               save_path=save_file + '_params.pickle',
                               )

model = opt_structures.Model(model, save_path=save_file + '_model.pickle')
results = opt_structures.Results(save_path=save_file + '_results.pickle')
optimizer = train_xsdc.TrainSupervised(data, model, params, results)

# Train the model with XSDC
optimizer.train()
