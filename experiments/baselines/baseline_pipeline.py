import os

from pandas import DataFrame

from baseline.get_data import get_preprocessed_data
from baseline.semi_sup_kmeans import predict_with_semi_sup_kmeans
from baseline.pipeline_utils import var_to_str, format_files, save, load


def exp_core(dataset, scaling=True, n_train=None, seed=0, n_labels=0,
             kernel='linear', sigma=None, reg=None, max_iter=100):
    data = get_preprocessed_data(dataset, scaling, n_train, seed, n_labels, kernel, reg, sigma)
    test_acc = predict_with_semi_sup_kmeans(data, max_iter)
    return test_acc


def wrapped_exp_core(data_cfg, optim_cfg):
    exp_cfg = dict(data_cfg=data_cfg, optim_cfg=optim_cfg)
    print(*['{0}:{1}'.format(key, value) for key, value in exp_cfg.items()], sep='\n')
    exp_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(exp_path, 'results', var_to_str(data_cfg), var_to_str(optim_cfg)) + format_files
    if os.path.exists(file_path):
        _, test_acc = load(open(file_path, 'rb'))
    else:
        test_acc = exp_core(**data_cfg, **optim_cfg)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save([exp_cfg, test_acc], open(file_path, 'wb'))

    result = {**data_cfg, **optim_cfg}
    result.update(test_acc=test_acc.item())
    result = DataFrame(result, index=[0])
    return result




