"""
Train a KN on the Gisette dataset
"""

import argparse
import numpy as np
import os
import random
import sklearn.metrics
import sys
import time
import torch

sys.path.append('../..')

from src.opt import opt_structures, train_xsdc
from src.model.ckn import parse_config, net
from src import default_params as defaults
import src.data_loaders.gisette as gisette

# Parameters for the model, data, and training
parser = argparse.ArgumentParser(description='KN training on the Gisette dataset')
parser.add_argument('--balanced', default=1, type=int,
                    help='Whether the unlabeled data should be balanced (1) or not (0)')
parser.add_argument('--batch_size', default=4096, type=int,
                    help='Batch size for the validation and test data')
parser.add_argument('--batch_size_labeled', default=4096, type=int,
                    help='Batch size for the labeled training data')
parser.add_argument('--batch_size_unlabeled', default=4096, type=int,
                    help='Batch size for the unlabeled training data')
parser.add_argument('--data_path', default='../../data/gisette_scale', type=str,
                    help='Location of the MAGIC dataset')
parser.add_argument('--eval_test_every', default=10, type=int,
                    help='Number of iterations between evaluations of the performance on the test set')
parser.add_argument('--gpu', default='0', type=str,
                    help='Which GPU to use')
parser.add_argument('--labeling_burnin', default=100, type=int,
                    help='Number of iterations to perform on the labeled data prior to using the unlabeled data')
parser.add_argument('--labeling_method', default='matrix balancing', type=str,
                    help="Method to use for labeling unlabeled observations. One of 'matrix balancing',"
                         "'pseudo labeling', or 'deep clustering'.")
parser.add_argument('--lam', default=None, type=int,
                    help='log2(l2 penalty on classifier parameters)')
parser.add_argument('--lambda_filters', default=-4, type=int,
                    help="log2(L2 penalty on the norm of the filters)")
parser.add_argument('--lr_semisup', default=-4, type=int,
                    help='log2(Learning rate for the semi-supervised learning)')
parser.add_argument('--lr_sup_init', default=-4, type=int,
                    help='log2(Learning rate for the supervised initialization)')
parser.add_argument('--num_clusters', default=2, type=int,
                    help='Number of clusters to use in deep clustering')
parser.add_argument('--num_filters', default=32, type=int,
                    help='Number of filters per layer in the model')
parser.add_argument('--num_iters', default=500, type=int,
                    help='Number of total iterations to perform')
parser.add_argument('--num_labeled', default=50, type=int,
                    help='Fraction of data that is labeled')
parser.add_argument('--save_every', default=50, type=int,
                    help='Number of iterations between saves of the model and results')
parser.add_argument('--save_path', default='../../results/gisette_kn/temp', type=str,
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
args.lambda_filters = 2**args.lambda_filters
args.lam = 2**args.lam if args.lam is not None else None
stratified_unlabeling = True if args.num_labeled > 0 else False
print(args)

balanced_version = True
min_frac_points_class = 1/args.num_clusters
max_frac_points_class = 1/args.num_clusters

nclasses = 2
ckn = True

# Create the data loaders
train_loader, _, train_labeled_loader, train_unlabeled_loader, valid_loader, train_valid_loader, test_loader = \
    gisette.get_dataloaders(batch_size=args.batch_size, batch_size_labeled=args.batch_size_labeled,
                            batch_size_unlabeled=args.batch_size_unlabeled, num_labeled=args.num_labeled,
                            stratified_unlabeling=stratified_unlabeling, data_path=args.data_path, num_workers=0)

bw = np.median(sklearn.metrics.pairwise.pairwise_distances(train_loader.dataset.dataset.images[0:1000]).reshape(-1))

save_dir = args.save_path
save_file = save_dir + str(args.num_labeled) + '_' + str(-1) + '_' + str(args.num_filters) + '_' + str(bw) + \
            '_' + str(args.lr_sup_init) + '_' + str(args.lr_semisup) + '_' + \
            '-'.join(str(args.labeling_method).split(' ')) + '_' + str(args.lam) + '_0_' + str(args.lambda_filters) + \
            '_' + str(max_frac_points_class) + '_' + str(args.num_clusters) + '_' + str(args.update_clusters_every) + \
            '_' + str(args.seed) + '_' + str(time.time())

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load and initialize the model
params = parse_config.load_config('../../cfg/kn.cfg')
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
                               max_frac_points_class=max_frac_points_class, ckn=ckn, project=False,
                               lambda_filters=args.lambda_filters, lam=args.lam, normalize=True,
                               balanced=balanced_version, labeling_method=args.labeling_method,
                               deepcluster_k=args.num_clusters,
                               deepcluster_update_clusters_every=args.update_clusters_every,
                               labeling_burnin=args.labeling_burnin, step_size_init_sup=args.lr_sup_init,
                               step_size_init_semisup=args.lr_semisup, maxiter=args.num_iters,
                               eval_test_every=args.eval_test_every, save_every=args.save_every,
                               save_path=save_file + '_params.pickle')

model = opt_structures.Model(model, save_path=save_file + '_model.pickle')
results = opt_structures.Results(save_path=save_file + '_results.pickle')
optimizer = train_xsdc.TrainSupervised(data, model, params, results)

# Train the model with XSDC
optimizer.train()
