import sys
from pandas import DataFrame

sys.path.append('../..')

from experiments.baselines.baseline_pipeline import wrapped_exp_core
from experiments.baselines.plot_tools import plot_exp, nice_plots
from experiments.baselines.pipeline_utils import build_list_exp


def run_all(plot=False):
    datasets = [
        'gisette',
        # 'magic',
        # 'mnist',
        # 'cifar',
    ]

    results = DataFrame()
    for dataset in datasets:
        for kernel in [
                        'linear',
                        'rbf',
                        'svd',
                       ]:
            results_dataset = run_one(dataset, kernel)
            results = results.append(results_dataset, ignore_index=True)
    if plot:
        plot_exp(results, add_xsdc_results=True)


def run_one(dataset, kernel, plot=False):
    scaling = True
    data_cfg = dict(dataset=dataset, seed=[i for i in range(10)], scaling=scaling,
                    n_labels=[50 * i for i in range(0, 11)], kernel=kernel)
    optim_cfg = dict(max_iter=100)
    exp_cfgs = build_list_exp([dict(data_cfg=data_cfg, optim_cfg=optim_cfg)])

    results = DataFrame()
    for exp_cfg in exp_cfgs:
        result = wrapped_exp_core(exp_cfg['data_cfg'], exp_cfg['optim_cfg'])
        results = results.append(result, ignore_index=True)
    if plot:
        plot_exp(results)
    return results


if __name__ == '__main__':
    run_all(True)
