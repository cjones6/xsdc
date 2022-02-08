import os

from matplotlib import pyplot as plt
from pandas import DataFrame
from matplotlib import cm
import numpy as np
import pickle
import seaborn as sns

nice_writing = dict(gisette='Gisette', magic='MAGIC', mnist='MNIST', cifar='CIFAR-10',
                    linear='k-means-ssl-lin', rbf='k-means-ssl-rbf', svd='k-means-ssl-corr', xsdc='XSDC',
                    kernel='Kernel')


def plot_exp(results, max_n_labels=None, add_xsdc_results=False):
    set_plt_params()
    datasets = results['dataset'].unique().tolist()
    fig, axs = plt.subplots(1, len(datasets), squeeze=False, figsize=(20*len(datasets), 10))

    if add_xsdc_results:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'xsdc_results.pickle')
        with open(file_path, 'rb') as file:
            xsdc_results = pickle.load(file)
        results = results.append(xsdc_results, ignore_index=True)
    if max_n_labels is not None:
        results = results[results['n_labels'] <= max_n_labels]

    for i, dataset in enumerate(datasets):
        sub_results = results[results['dataset'] == dataset]
        sns.lineplot(x='n_labels', y='test_acc', hue='kernel', style='kernel',
                     data=sub_results, ax=axs[0, i],
                     dashes=True, markers=True)

        axs[0, i].set_xlabel('# labeled observations')
        axs[0, i].set_title(nice_writing[dataset])
        axs[0, i].set_ylim(0, 1)
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    for i in range(len(labels)):
        labels[i] = nice_writing[labels[i]]

    for i in range(len(datasets)):
        axs[0, i].get_legend().remove()
        if i > 0:
            axs[0, i].set(ylabel=None)
        else:
            axs[0, i].set_ylabel('Test Accuracy')

    fig.legend(handles=handles, labels=labels,
               loc='upper center',
               ncol=len(labels),
               bbox_to_anchor=(0.5, 1.2)
               )
    fig.tight_layout()

    plt.show()
    # baseline_path = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(baseline_path, 'plots/baseline.pdf')
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # fig.savefig(file_path, format='pdf', bbox_inches='tight')


def nice_plots(results):
    datasets = ['gisette', 'magic', 'mnist', 'cifar']
    dataset_titles = ['Gisette', 'MAGIC', 'MNIST', 'CIFAR-10']
    methods = ['linear', 'svd', 'rbf'
        , 'xsdc'
               ]
    params = {'axes.labelsize': 44,
              'font.size': 44,
              'legend.fontsize': 44,
              'xtick.labelsize': 28,
              'ytick.labelsize': 28,
              'font.family': 'Times New Roman',
              # 'text.usetex': True,
              'lines.markersize': 8,
              'lines.linewidth': 3
              }

    plt.rcParams.update(params)

    cm_subsection = np.linspace(0.5, 1, 3)
    colors = [cm.Blues(x) for x in cm_subsection][::-1]
    red_colors = [cm.Reds(x) for x in cm_subsection][::-1]
    markers = ['s', 'd', 'o', 'p']
    # plt.figure(1)
    fig1, ax1 = plt.subplots(ncols=4, figsize=(32, 7))
    plot_type = 0

    for dataset_num, dataset in enumerate(datasets):
        results0 = results[results['dataset'] == dataset]
        summary_results = {}
        for method in methods:
            if method == 'xsdc':
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xsdc_results',
                                         dataset + '_comparison_results_nolambda.pickle')
                result_xsdc = pickle.load(open(file_path, 'rb'))
                result_xsdc = result_xsdc['xsdc']
                summary_results[method] = {}
                summary_results[method]['mean'] = []
                summary_results[method]['std'] = []
                summary_results[method]['n_labels'] = []
                for num_labeled in sorted(result_xsdc.keys()):
                    summary_results[method]['mean'].append(np.mean(result_xsdc[num_labeled]))
                    std = np.std(result_xsdc[num_labeled])
                    # std = 0
                    summary_results[method]['std'].append(std)
                    summary_results[method]['n_labels'].append(num_labeled)

            else:
                results1 = results0[results0['kernel'] == method]
                n_labels = results1['n_labels'].unique().tolist()
                mean = results1.groupby('n_labels')['test_acc'].mean().to_list()
                std = results1.groupby('n_labels')['test_acc'].std().to_list()
                summary_results[method] = {}
                summary_results[method]['mean'] = mean
                summary_results[method]['std'] = std
                # summary_results[method]['std'] = [0 for _ in range(len(n_labels))]
                summary_results[method]['n_labels'] = n_labels
                
        l1 = ax1[dataset_num].errorbar(summary_results['linear']['n_labels'], summary_results['linear']['mean'],
                                       yerr=summary_results['linear']['std'],
                                       label='k-means\textsubscript{l}',
                                       marker=markers[0], mfc='white', c=red_colors[0], zorder=1,
                                       capsize=8)
        l3 = ax1[dataset_num].errorbar(summary_results['svd']['n_labels'], summary_results['svd']['mean'],
                                       yerr=summary_results['svd']['std'],
                                       label='k-means\textsubscript{m}',
                                       marker=markers[2], mfc='white', c=red_colors[2],
                                       zorder=3, capsize=8)
        l2 = ax1[dataset_num].errorbar(summary_results['rbf']['n_labels'], summary_results['rbf']['mean'],
                                       yerr=summary_results['rbf']['std'],
                                       label='k-means\textsubscript{r}',
                                       marker=markers[1], mfc='white', c=red_colors[1],
                                       zorder=2, capsize=8)
        l4 = ax1[dataset_num].errorbar(summary_results['xsdc']['n_labels'], summary_results['xsdc']['mean'],
                                       yerr=summary_results['xsdc']['std'],
                                       label='xsdc',
                                       marker=markers[3], mfc='white', c=colors[0],
                                       zorder=4, capsize=8)

        if dataset_num == 0:
            ax1[dataset_num].set_ylim((0.45, 1))
        elif dataset_num == 1:
            ax1[dataset_num].set_ylim((0.45, 1.))
        elif dataset_num == 2:
            ax1[dataset_num].set_ylim((0.1, 1))
        else:
            ax1[dataset_num].set_ylim((0.05, 0.45))
        ax1[dataset_num].set_xlabel(r'\# labeled observations')
        if dataset_num == 0:
            ax1[dataset_num].set_ylabel('Test accuracy')
        ax1[dataset_num].set_title(dataset_titles[dataset_num], fontsize=44)

    fig1.legend(handles=[l1, l3, l2, l4],
                labels=['k-means-ssl-lin', 'k-means-ssl-corr', 'k-means-ssl-rbf', 'XSDC'],
                loc='lower center',
                bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=4, frameon=False)
    fig1.subplots_adjust(left=0.2, bottom=0.3, wspace=0.15, hspace=0.15)
    plt.show()
    baseline_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(baseline_path, 'plots/new_baseline')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fig1.savefig(file_path + '.pdf', format='pdf', bbox_inches='tight')
    pickle.dump(results, open(file_path + '.pickle', 'wb'))


def set_plt_params():
    params = {'axes.labelsize': 60,
              'font.size': 50,
              'legend.fontsize': 50,
              'xtick.labelsize': 60,
              'ytick.labelsize': 60,
              'lines.linewidth': 5,
              #'text.usetex': True,
              'lines.markersize': 30,
              # 'figure.figsize': (8, 6)
              }
    plt.rcParams.update(params)
