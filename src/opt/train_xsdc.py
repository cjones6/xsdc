import numpy as np
import time
import torch
import torch.optim as optim

from . import train_classifier, ulr_utils, opt_utils, label_utils, deepcluster
from src import default_params as defaults


class TrainSupervised:
    """
    Class to optimize the filters of a kernel network, parameters of a classifier, and any unknown labels.

    :param data: Data object containing the training, validation, and testing set dataloaders
    :param model: Model object containing the deep architecture to be used in training
    :param params: Parameters object that contains the parameters related to training and evaluation
    :param results: Results object for storing the results
    """
    def __init__(self, data, model, params, results):
        self.data = data
        self.model = model
        self.params = params
        self.results = results

        self.iteration = 0
        self.w_last = params.w_last_init
        self.b_last = params.b_last_init
        self.step_size = params.step_size_init_sup
        self.params.only_unsup = self.data.train_labeled_loader is None

        if self.params.train_w_layers is None and not self.params.convnet:
            self.params.train_w_layers = range(len(self.model.model.layers))
        if self.params.convnet:
            self.optimizer = optim.SGD(self.model.model.parameters(), lr=self.params.step_size_init_sup,
                                       momentum=self.params.momentum, weight_decay=self.params.lambda_filters)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 500, gamma=0.25)

        if self.params.labeling_method == 'deep clustering':
            self.deepcluster = deepcluster.__dict__['Kmeans'](self.params.deepcluster_k)
            self.images = self.data.deepcluster_loader.dataset.dataset.images[self.data.deepcluster_loader.dataset.indices]

    def _get_step_size(self):
        """
        Get the step size to use at this iteration.

        :return Step size
        """
        if self.iteration < self.params.labeling_burnin:
            return self.params.step_size_init_sup
        else:
            return self.params.step_size_init_semisup

    def _update_deepcluster_labels(self):
        """
        Update the labels used in deep clustering
        """
        all_features = opt_utils.compute_all_features(self.data.train_labeled_loader,
                                                      self.data.train_unlabeled_loader, None, None, self.model,
                                                      normalize=True, standardize=False,
                                                      eps=1e-5, augment=self.params.augment)
        if len(all_features['train_labeled']['x']) > 0:
            all_features = torch.cat((all_features['train_labeled']['x'], all_features['train_unlabeled']['x']))
            batch_size = self.data.train_labeled_loader.batch_size
        else:
            all_features = all_features['train_unlabeled']['x']
            batch_size = self.data.train_unlabeled_loader.batch_size
        self.deepcluster.cluster(all_features.numpy(), verbose=False)
        train_dataset = deepcluster.cluster_assign(self.deepcluster.images_lists, self.images)

        sampler = deepcluster.UnifLabelSampler(int(self.params.deepcluster_update_clusters_every * batch_size),
                                               self.deepcluster.images_lists)
        try:
            train_dataset.transform = self.data.deepcluster_loader.dataset.dataset.transform
        except:
            train_dataset.transform = self.data.deepcluster_loader.dataset.transform
        self.data.deepcluster_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            sampler=sampler,
            pin_memory=True,
        )

    def _update_filters(self):
        """
        Take one step to optimize the filters at each layer.
        """
        self.step_size = self._get_step_size()

        # Update the deep clustering labels if applicable and necessary
        if self.params.labeling_method == 'deep clustering' \
                and self.iteration % self.params.deepcluster_update_clusters_every == 0 \
                and self.iteration >= self.params.labeling_burnin:
            self._update_deepcluster_labels()

        # Get a batch of labeled and/or unlabeled data
        if not self.params.only_unsup and (not self.params.labeling_method == 'deep clustering' \
                                           and self.iteration >= self.params.labeling_burnin):
            batch = opt_utils.get_batch(self.data, labeled=True, unlabeled=True, augment=self.params.augment)
            x_train_labeled, x_train_unlabeled, y_train_labeled, _, y_train_unlabeled_truth = batch
        elif self.iteration < self.params.labeling_burnin and not self.params.only_sup:
            x_train_labeled, y_train_labeled = opt_utils.get_batch(self.data, labeled=True, unlabeled=False,
                                                                   augment=self.params.augment)
            x_train_unlabeled, y_train_unlabeled_truth = None, None
        elif self.params.only_unsup:
            x_train_unlabeled, y_train_unlabeled, y_train_unlabeled_truth = opt_utils.get_batch(self.data,
                                                                                                labeled=False,
                                                                                                unlabeled=True,
                                                                                                augment=self.params.augment)
            x_train_labeled, y_train_labeled = None, None
        else:
            x_train_labeled, y_train_labeled = opt_utils.get_batch(self.data, labeled=False, unlabeled=True,
                                                                   deep_cluster=True, augment=self.params.augment)
            x_train_unlabeled, y_train_unlabeled_truth = None, None

        # Take a ULR-SGO step
        self.model.zero_grad()
        obj_value = self._ultimate_layer_reversal(x_train_labeled, y_train_labeled, x_train_unlabeled,
                                                  y_train_unlabeled_truth)
        obj_value.backward()

        if self.params.ckn:
            with torch.autograd.set_grad_enabled(False):
                for layer_num in self.params.train_w_layers:
                    grad = self.model.model.layers[layer_num].W.grad.data
                    W = self.model.model.layers[layer_num].W.data
                    if self.params.project:
                        W = W - self.step_size * grad / torch.norm(grad, 2, 1, keepdim=True)
                        W_proj = W / torch.norm(W, 2, 1, keepdim=True)
                        self.model.model.layers[layer_num].W.data = W_proj
                    else:
                        grad += 2*self.params.lambda_filters*self.model.model.layers[layer_num].W
                        self.model.model.layers[layer_num].W.data = W - self.step_size * grad
        else:
            self.optimizer.step()
            self.lr_scheduler.step()

    def _get_constraints(self, y_train_labeled, n, augment):
        """
        Get constraints for the equivalence matrix based on the labeled data.

        :param y_train_labeled: Labels for the labeled subset of the batch
        :param n: Batch size (labeled + unlabeled observations)
        :param augment: Number of augmented copies of each input example to use. If 0 then no augmentation is performed
        :return: mask: Binary matrix with value 0 in entry (i,j) if it is known whether i and j belong to the same class
                       and 1 else
        :return: known: Binary matrix with value 1 in entry (i,j) if it is known that i and j belong to the same class
                        and 0 else
        """
        if y_train_labeled is not None:
            nl = len(y_train_labeled)
            mask = (torch.BoolTensor(n, n).zero_() + 1).to(defaults.device)
            known = torch.zeros(n, n).to(defaults.device)
            y_labeled_one_hot = opt_utils.one_hot_embedding(y_train_labeled, self.params.nclasses)
            known[0:nl, 0:nl] = y_labeled_one_hot.mm(y_labeled_one_hot.t())
            torch.diagonal(known).fill_(1)
            mask[0:nl, 0:nl] = 0
            torch.diagonal(mask).fill_(0)
        else:
            mask = (torch.BoolTensor(n, n).zero_() + 1).to(defaults.device)
            known = torch.zeros(n, n).to(defaults.device)
            torch.diagonal(known).fill_(1)
            torch.diagonal(mask).fill_(0)
        if augment > 0:
            if y_train_labeled is not None:
                nu = n-nl
            else:
                nu = n
                nl = 0
            for i in range(int(nu//augment)):
                known[nl+i*augment:nl+(i+1)*augment, nl+i*augment:nl+(i+1)*augment] = 1
                mask[nl+i*augment:nl+(i+1)*augment, nl+i*augment:nl+(i+1)*augment] = 0

        return mask, known

    def _add_constraints(self, mask, known, y_train_labeled, y_train_unlabeled_truth):
        """
        Add additional constraints to the equivalence matrix.

        :param mask: Binary matrix with value 0 in entry (i,j) if it is known whether i and j belong to the same class
                     and 1 else
        :param known: Binary matrix with value 1 in entry (i,j) if it is known that i and j belong to the same class and
                      0 else
        :param y_train_labeled: Labels for the labeled subset of the batch
        :param y_train_unlabeled_truth: True (generally unknown) labels for the unlabeled subset of the batch
        :return: mask: Binary matrix with value 0 in entry (i,j) if it is known whether i and j belong to the same class
                       and 1 else
        :return: known: Binary matrix with value 1 in entry (i,j) if it is known that i and j belong to the same class
                        and 0 else
        """
        if self.params.add_constraints_method == 'random':
            n = len(mask)
            nl = len(y_train_labeled)
            mask = (torch.BoolTensor(n, n).zero_() + 1)
            y_labeled_one_hot = opt_utils.one_hot_embedding(y_train_labeled, self.params.nclasses)

            mask = mask.cpu().numpy()
            idxs = np.random.choice([0, 1], size=(n, n), p=[1 - self.params.add_constraints_frac,
                                                            self.params.add_constraints_frac])
            idxs = np.triu(idxs, k=1)
            idxs = idxs + idxs.T
            idxs = idxs.astype('bool')
            mask = mask * (~idxs)
            true_y = opt_utils.one_hot_embedding(torch.cat((y_train_labeled, y_train_unlabeled_truth.to(defaults.device))),
                                                 self.params.nclasses)
            true_m = true_y.mm(true_y.t())

            known = true_m * torch.Tensor(idxs).to(defaults.device)
            mask = torch.from_numpy(mask).to(defaults.device)

            known[:nl, :nl] = y_labeled_one_hot.mm(y_labeled_one_hot.t())
            torch.diagonal(known).fill_(1)
            mask[:nl, :nl] = 0
            torch.diagonal(mask).fill_(0)

            # Remove 1's among (labeled, unlabeled) pairs
            bad_idxs = known[:nl, nl:] == 1
            known[:nl, nl:][bad_idxs] = 0
            mask[:nl, nl:][bad_idxs] = 1

            bad_idxs = known[nl:, :nl] == 1
            known[nl:, :nl][bad_idxs] = 0
            mask[nl:, :nl][bad_idxs] = 1

        elif self.params.add_constraints_method == 'specific':
            mask = mask.cpu().numpy()
            nl = len(y_train_labeled)
            idxs_unlabeled = np.isin(y_train_unlabeled_truth.cpu(), self.params.add_constraints_classes)
            idxs_labeled = np.isin(y_train_labeled.cpu(), self.params.add_constraints_classes)
            mask[:nl, nl:][np.ix_(~idxs_labeled, idxs_unlabeled)] = 0
            mask[nl:, :nl][np.ix_(idxs_unlabeled, ~idxs_labeled)] = 0
            mask[:nl, nl:][np.ix_(idxs_labeled, ~idxs_unlabeled)] = 0
            mask[nl:, :nl][np.ix_(~idxs_unlabeled, idxs_labeled)] = 0
            mask[nl:, nl:][np.ix_(idxs_unlabeled, ~idxs_unlabeled)] = 0
            mask[nl:, nl:][np.ix_(~idxs_unlabeled, idxs_unlabeled)] = 0
            mask = torch.from_numpy(mask).to(defaults.device)

        return mask, known

    def _ultimate_layer_reversal(self, x_train_labeled, y_train_labeled, x_train_unlabeled, y_train_unlabeled_truth):
        """
        Compute the objective function of the ultimate layer reversal method and update W, b.

        :param x_train_labeled: Labeled training set features
        :param y_train_labeled: Labeled training set labels
        :param x_train_unlabeled: Unlabeled training set features
        :param y_train_unlabeled_truth: Unlabeled training set labels (only used for adding additional constraints, if
                                        specified)
        :return: obj: Objective value
        """
        if self.iteration < self.params.labeling_burnin or self.params.labeling_method == 'deep clustering':
            features = opt_utils.compute_features(x_train_labeled, self.model, normalize=self.params.normalize,
                                                  standardize=self.params.standardize)
        else:
            features_unlabeled = opt_utils.compute_features(x_train_unlabeled, self.model,
                                                            normalize=self.params.normalize,
                                                            standardize=self.params.standardize)
            if x_train_labeled is not None:
                features_labeled = opt_utils.compute_features(x_train_labeled, self.model,
                                                              normalize=self.params.normalize,
                                                              standardize=self.params.standardize)
                features = torch.cat((features_labeled, features_unlabeled))
            else:
                features = features_unlabeled

        if self.iteration >= self.params.labeling_burnin and not self.params.labeling_method == 'deep clustering':
            with torch.autograd.no_grad():
                n = len(features)
                mask, known = self._get_constraints(y_train_labeled, n, self.params.augment)
                if self.params.add_constraints:
                    mask, known = self._add_constraints(mask, known, y_train_labeled, y_train_unlabeled_truth)

            if self.params.labeling_method == 'matrix balancing':
                with torch.autograd.no_grad():
                    if self.iteration >= self.params.labeling_burnin:
                        k = self.params.deepcluster_k
                    else:
                        k = self.params.nclasses
                    M, eigenvalues = label_utils.optimize_labels(features, k, self.params.lam, mask=mask,
                                                                 known_values=known,
                                                                 nmin=self.params.min_frac_points_class*n,
                                                                 nmax=self.params.max_frac_points_class*n,
                                                                 eigenvalues=False)
                    M = M.type(torch.get_default_dtype())
                    self.results.update(self.iteration, **{'eigenvalues': eigenvalues})
                obj = ulr_utils.ulr_square_loss_m(features.to(defaults.device), M.to(defaults.device), self.params.lam)
            elif self.params.labeling_method == 'eigendecomposition':
                with torch.autograd.no_grad():
                    M, eigenvalues = label_utils.diffrac_relaxation_estimate_M(features, self.params.nclasses,
                                                                               self.params.lam, mask, known,
                                                                               nmin=self.params.min_frac_points_class*n,
                                                                               use_cpu=False,
                                                                               rounding=self.params.rounding)
                self.results.update(self.iteration, **{'eigenvalues': eigenvalues.cpu().numpy()})
                obj = ulr_utils.ulr_square_loss_m(features.to(defaults.device), M.to(defaults.device), self.params.lam)
            elif self.params.labeling_method == 'pseudo labeling':
                with torch.autograd.no_grad():
                    if y_train_labeled is not None:
                        y_labeled_one_hot = opt_utils.one_hot_embedding(y_train_labeled, self.params.nclasses)
                        obj, self.w_last, self.b_last = ulr_utils.ulr_square_loss_y(features_labeled, y_labeled_one_hot,
                                                                                    self.params.lam, 0)
                    yhat = features_unlabeled.mm(self.w_last) + self.b_last
                    y_pseudo = torch.argmax(yhat, 1)

                    if y_train_labeled is not None:
                        y = opt_utils.one_hot_embedding(torch.cat((y_train_labeled, y_pseudo)), self.params.nclasses)
                    else:
                        y = opt_utils.one_hot_embedding(y_pseudo, self.params.nclasses)
                    M = y.mm(y.t())
                obj = ulr_utils.ulr_square_loss_m(features.to(defaults.device), M.to(defaults.device), self.params.lam)
            else:
                raise NotImplementedError

        elif self.iteration >= self.params.labeling_burnin and self.params.labeling_method == 'deep clustering':
            y_one_hot = opt_utils.one_hot_embedding(y_train_labeled, self.params.deepcluster_k)
            M = y_one_hot.mm(y_one_hot.t()).to(defaults.device)
            obj = ulr_utils.ulr_square_loss_m(features.to(defaults.device), M.to(defaults.device),
                                              self.params.lam)

        else:
            y_one_hot = opt_utils.one_hot_embedding(y_train_labeled, self.params.nclasses)
            obj, w_last, b_last = ulr_utils.ulr_square_loss_y(features, y_one_hot, self.params.lam)
            self.w_last = w_last.detach()
            self.b_last = b_last.detach()

        return obj

    def _optimize_classifier_evaluate(self, only_labeled=False, only_unlabeled=False, update_lam=False):
        """
        Optimize the classifier on the full dataset and then evaluate the model.

        :param only_labeled: Whether to only use the labeled data
        :param only_unlabeled: Whether the data is only unlabeled
        :param update_lam: Whether to update the regularization parameter lambda
        """
        self.model.eval()
        if self.params.ckn:
            self.model.model = opt_utils.compute_normalizations(self.model.model)

        if only_labeled is False and only_unlabeled is False:
            all_features = opt_utils.compute_all_features(self.data.train_labeled_loader,
                                                          self.data.train_unlabeled_loader,
                                                          self.data.valid_loader,
                                                          self.data.test_loader,
                                                          self.model,
                                                          normalize=self.params.normalize,
                                                          standardize=self.params.standardize,
                                                          augment=self.params.augment)
            y_labeled_one_hot = opt_utils.one_hot_embedding(all_features['train_labeled']['y'],
                                                            self.params.nclasses).to(defaults.device)
            y_unlabeled = opt_utils.nearest_neighbor(all_features['train_labeled']['x'],
                                                     all_features['train_unlabeled']['x'],
                                                     all_features['train_labeled']['y'],
                                                     self.params.nn)
            y_unlabeled_one_hot = opt_utils.one_hot_embedding(y_unlabeled, self.params.nclasses)
            x_train = torch.cat((all_features['train_labeled']['x'],
                                 all_features['train_unlabeled']['x'])).to(defaults.device)
            if not update_lam:
                with torch.autograd.no_grad():
                    _, w_last, b_last = ulr_utils.ulr_square_loss_y(x_train,
                                                                    torch.cat((y_labeled_one_hot, y_unlabeled_one_hot)),
                                                                    self.params.lam)
            else:
                y_train = torch.argmax(torch.cat((y_labeled_one_hot, y_unlabeled_one_hot)), 1)
                test_acc, valid_acc, train_acc, test_loss, train_loss, w, best_lambda = train_classifier.train(
                                (x_train, y_train), (all_features['valid']['x'], all_features['valid']['y']),
                                (all_features['test']['x'], all_features['test']['y']), self.model,
                                self.params.nclasses, self.params.maxiter_wlast_full, w_init=None, normalize=True,
                                standardize=False, loss_name='square', lambdas=None, input_features=True)
                self.params.lam = best_lambda
                w_last = w[1:, :]
                b_last = w[0, :]
        elif only_unlabeled is False:
            all_features = opt_utils.compute_all_features(self.data.train_labeled_loader,
                                                          self.data.train_unlabeled_loader,
                                                          self.data.valid_loader,
                                                          self.data.test_loader,
                                                          self.model,
                                                          normalize=self.params.normalize,
                                                          standardize=self.params.standardize,
                                                          augment=self.params.augment)
            if self.iteration != 0 and not update_lam:
                y_labeled_one_hot = opt_utils.one_hot_embedding(all_features['train_labeled']['y'],
                                                                self.params.nclasses)
                _, w_last, b_last = ulr_utils.ulr_square_loss_y(all_features['train_labeled']['x'].to(defaults.device),
                                                                          y_labeled_one_hot, self.params.lam)
            else:
                test_acc, valid_acc, train_acc, test_loss, train_loss, w, best_lambda = train_classifier.train(
                    (all_features['train_labeled']['x'], all_features['train_labeled']['y']),
                    (all_features['valid']['x'], all_features['valid']['y']), (all_features['test']['x'],
                    all_features['test']['y']), self.model, self.params.nclasses, self.params.maxiter_wlast_full,
                    w_init=None, normalize=True, standardize=False, loss_name='square', lambdas=None,
                    input_features=True)
                self.params.lam = best_lambda
                self.w_last = w
                w_last = w[1:, :]
                b_last = w[0, :]
        else:
            all_features = opt_utils.compute_all_features(None,
                                                          None,
                                                          None,
                                                          self.data.test_loader,
                                                          self.model,
                                                          normalize=self.params.normalize,
                                                          standardize=self.params.standardize,
                                                          augment=self.params.augment)
            with torch.autograd.no_grad():
                n = len(all_features['test']['x'])
                mask = (torch.BoolTensor(n, n).zero_() + 1).to(defaults.device)
                known = torch.zeros(n, n).to(defaults.device)
                torch.diagonal(known).fill_(1)
                torch.diagonal(mask).fill_(0)

                x_test = all_features['test']['x'].to(defaults.device)
                M, eigengap = label_utils.optimize_labels(x_test, self.params.nclasses, self.params.lam, mask=mask,
                                                          known_values=known, nmin=self.params.min_frac_points_class*n,
                                                          nmax=self.params.max_frac_points_class*n, eigenvalues=False)
                M = M.type(torch.get_default_dtype())
                y_test = all_features['test']['y'].to(defaults.device)
                yhat_test = torch.LongTensor(label_utils.get_estimated_labels(M, y_test, self.params.nclasses))
                if self.params.labeling_method == 'pseudo labeling':
                    yhat_one_hot = opt_utils.one_hot_embedding(yhat_test, self.params.nclasses)
                    obj, self.w_last, self.b_last = ulr_utils.ulr_square_loss_y(x_test, yhat_one_hot,
                                                                                self.params.lam, 0)

                test_accuracy = torch.mean((y_test.cpu() == yhat_test).float())
                if self.iteration == 0:
                    print('Iteration \t Test accuracy')
                print(self.iteration, '\t\t', '{:06.4f}'.format(test_accuracy.item()))
                results = {'test_accuracy': test_accuracy}
                self.results.update(self.iteration, **results)

        if not only_unlabeled:
            results = opt_utils.evaluate_features(self.params, w_last, b_last, all_features)
            if not only_labeled:
                if self.iteration == 0:
                    opt_utils.print_results(self.iteration, results, header=True)
                else:
                    opt_utils.print_results(self.iteration, results, header=False)
            self.results.update(self.iteration, **results)

        if self.params.ckn:
            for layer_num in range(len(self.model.model.layers)):
                self.model.model.layers[layer_num].store_normalization = False
        self.model.train()

    def train(self):
        """
        Train the network and classifier using the ultimate layer reversal method and learn the unknown labels.
        """
        iter_since_last_eval = iter_since_last_save = 0
        if not self.params.only_unsup and self.params.update_lambda:
            self._optimize_classifier_evaluate(only_labeled=True)
            self._optimize_classifier_evaluate(update_lam=True)
        else:
            self._optimize_classifier_evaluate(only_labeled=False, only_unlabeled=True)

        if self.iteration >= self.params.labeling_burnin and self.params.convnet:
            self.optimizer = optim.SGD(self.model.model.parameters(), lr=self.params.step_size_init_semisup,
                                           momentum=self.params.momentum, weight_decay=self.params.lambda_filters)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 500, gamma=1)

        while self.iteration < self.params.maxiter_final:
            t1 = time.time()
            self._update_filters()
            t2 = time.time()
            self.iteration += 1
            self.results.update(self.iteration, epoch_time=t2-t1)
            iter_since_last_eval += 1
            iter_since_last_save += 1

            if iter_since_last_eval >= self.params.eval_test_every:
                if self.iteration % 100 == 0 and self.params.update_lambda:
                    self._optimize_classifier_evaluate(only_labeled=False, only_unlabeled=self.params.only_unsup,
                                                       update_lam=not self.params.only_unsup and not self.params.convnet)
                else:
                    self._optimize_classifier_evaluate(only_labeled=False, only_unlabeled=self.params.only_unsup)

                if iter_since_last_save >= self.params.save_every:
                    if self.params.convnet:
                        self.model.save(iteration=self.iteration, w_last=self.w_last, b_last=self.b_last,
                                        step_size=self.step_size, optimizer=self.optimizer)
                    else:
                        self.model.save(iteration=self.iteration, w_last=self.w_last, b_last=self.b_last,
                                        step_size=self.step_size)
                    self.results.save()
                    self.params.save()
                    iter_since_last_save = 0
                iter_since_last_eval = 0

            if self.iteration == self.params.labeling_burnin and self.params.convnet:
                self.optimizer = optim.SGD(self.model.model.parameters(), lr=self.params.step_size_init_semisup,
                                           momentum=self.params.momentum, weight_decay=self.params.lambda_filters)
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 500, gamma=1)

        print('Done training. Saving final results.')
        self._optimize_classifier_evaluate(only_labeled=False, only_unlabeled=self.params.only_unsup)
        self.results.save()
        self.params.save()
        if self.params.convnet:
            self.model.save(iteration=self.iteration, w_last=self.w_last, b_last=self.b_last, step_size=self.step_size,
                            optimizer=self.optimizer)
        else:
            self.model.save(iteration=self.iteration, w_last=self.w_last, b_last=self.b_last, step_size=self.step_size)
