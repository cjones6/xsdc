import inspect
import os
import socket
import torch
from copy import deepcopy

if not torch.cuda.is_available() or socket.gethostname() == 'zh-ws1':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')

type_files = 'torch'
if type_files == 'pickle':
    import pickle
    format_files = '.pickle'
    save = pickle.dump
    load = pickle.load
elif type_files == 'torch':
    import torch
    format_files = '.pt'
    save = torch.save

    def load(file):
        if str(device) == 'cpu':
            loaded = torch.load(file, map_location=torch.device('cpu'))
        else:
            loaded = torch.load(file, map_location=torch.device('cuda:0'))
        return loaded

else:
    raise NotImplementedError


def get_exp_folder():
    return os.path.dirname(os.path.abspath(__file__))


def save_reload_comput(method, check_exp_done, reload_param, **kwargs):
    print(*['{0}:{1}'.format(key, value) for key, value in kwargs.items()], sep='\n')
    assert not any([isinstance(value, list) for cfg in kwargs.values() for value in cfg.values()])
    output, aux_vars, log = load_exp(kwargs)
    exp_done = output is not None
    if not exp_done:
        output, aux_vars, log, exp_done = re_load_exp(kwargs, check_exp_done, reload_param=reload_param)
    if not exp_done:
        output, aux_vars, log = method(**kwargs, input=output, aux_vars=aux_vars, log=log)
        save_exp(kwargs, output, aux_vars, log)
    return output, aux_vars, log


def load_exp(exp_cfg):
    results_folder = get_exp_folder() + '/results'
    file_path = get_path_exp(exp_cfg, results_folder) + format_files
    out = [None, None, None]
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            loaded = load(file)
            out = loaded[1:]
    return out


def re_load_exp(exp_cfg, check_exp_done, reload_param='max_iter'):
    exp_cfg_folder = get_exp_folder() + '/results'
    for cfg in list(exp_cfg.values())[:-1]:
        exp_cfg_folder += '/{0}'.format(var_to_str(cfg))
    paths = search_paths_similar_cfgs(exp_cfg, exp_cfg_folder, reload_param)

    max_param = search_param_in_exp_cfg(exp_cfg, reload_param)

    var_param_done = 0
    exp_cfg_to_load = None
    exp_done = False
    info_exp = aux_vars = output = None
    for path in paths:
        with open(path, 'rb') as file:
            exp_cfg_saved = load(file)[0]
            var_param_saved = search_param_in_exp_cfg(exp_cfg_saved, reload_param)
            if var_param_saved > var_param_done and var_param_saved<=max_param:
                var_param_done = var_param_saved
                exp_cfg_to_load = exp_cfg_saved

    if exp_cfg_to_load is not None:
        output, aux_vars, info_exp = load_exp(exp_cfg_to_load)
        if check_exp_done(info_exp):
            exp_done = True
    return output, aux_vars, info_exp, exp_done


def save_exp(exp_cfg, *args):
    results_folder = get_exp_folder() + '/results'
    file_path = get_path_exp(exp_cfg, results_folder, create_entry=True) + format_files
    with open(file_path, 'wb') as file:
        save([exp_cfg, *args], file)


def erase_exp(exp_cfg):
    results_folder = get_exp_folder() + '/results'
    file_path = get_path_exp(exp_cfg, results_folder) + format_files
    if os.path.exists(file_path):
        os.remove(file_path)


def get_path_exp(exp_cfg, source_dir, create_entry=False):
    path = source_dir
    for cfg in exp_cfg.values():
        path += '/{0}'.format(var_to_str(cfg))
    if create_entry:
        assert not os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_grid_search(exp_cfg, best_params):
    file_path = get_exp_folder() + '/results/grid_searches' + format_files
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    new_entry = dict(exp_cfg=exp_cfg, best_params=best_params)
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as file:
            save([new_entry], file)
    else:
        with open(file_path, 'rb') as file:
            entries = load(file)
        for entry in entries:
            assert entry['exp_cfg'] != new_entry['exp_cfg']
        entries.append(new_entry)
        with open(file_path, 'wb') as file:
            save(entries, file)


def load_grid_search(exp_cfg):
    best_params = None
    file_path = get_exp_folder() + '/results/grid_searches'+ format_files
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            entries = load(file)
        for entry in entries:
            if entry['exp_cfg'] == exp_cfg:
                assert best_params is None
                best_params = entry['best_params']
    return best_params


def erase_grid_search_entry(exp_cfg):
    file_path = get_exp_folder() + '/results/grid_searches'
    with open(file_path + format_files, 'rb') as file:
        entries = load(file)
    for i, entry in enumerate(entries):
        if entry['exp_cfg'] == exp_cfg:
            entries.pop(i)
    with open(file_path + format_files, 'wb') as file:
        save(entries, file)


def search_paths_similar_cfgs(exp_cfg, root_path='', variable_param=''):
    exp_cfg_main = deepcopy(exp_cfg)
    found_param = search_param_in_exp_cfg(exp_cfg_main, variable_param, del_param=True)
    assert found_param is not None
    all_files = get_list_of_files(root_path)

    paths = list()
    for file_path in all_files:
        with open(file_path, 'rb') as file:
            exp_cfg_saved = load(file)[0]
            search_param_in_exp_cfg(exp_cfg_saved, variable_param, del_param=True)
            # if equal_exp_cfg(exp_cfg_main, exp_cfg_saved):
            if exp_cfg_main == exp_cfg_saved:
                paths.append(file_path)
    return paths


def search_param_in_exp_cfg(exp_cfg, param_to_search, del_param=False):
    found_param = None
    already_seen = False
    for cfg in exp_cfg.values():
        if param_to_search in cfg.keys():
            assert not already_seen
            found_param = deepcopy(cfg[param_to_search])
            already_seen = True
            if del_param:
                del cfg[param_to_search]
    return found_param


def var_to_str(var):
    translate_table = {ord(c): None for c in ',()[]'}
    translate_table.update({ord(' '): '_'})

    if type(var) == dict:
        sortedkeys = sorted(var.keys(), key=lambda x: x.lower())
        var_str = [key + '_' + var_to_str(var[key]) for key in sortedkeys if var[key] is not None]
        var_str = '_'.join(var_str)
    elif inspect.isclass(var):
        raise NotImplementedError('Do not give classes as items in cfg inputs')
    elif type(var) in [list, set, frozenset, tuple]:
        value_list_str = [var_to_str(item) for item in var]
        var_str = '_'.join(value_list_str)
    elif isinstance(var, float):
        var_str = '{0:1.2e}'.format(var)
    elif isinstance(var, int):
        var_str = str(var)
    elif isinstance(var, str):
        var_str = var
    elif var is None:
        var_str = str(var)
    elif isinstance(var, torch.Tensor):
        # todo: use norm of the tensor as its signature for saving it, avoid if possible
        var_str = '{0:.6e}'.format(torch.norm(var).item())
    else:
        print(type(var))
        raise NotImplementedError
    return var_str


def get_list_of_files(dir_name):
    all_files = list()
    if os.path.exists(dir_name):
        list_of_files = os.listdir(dir_name)
        for entry in list_of_files:
            full_path = os.path.join(dir_name, entry)
            if os.path.isdir(full_path):
                all_files = all_files + get_list_of_files(full_path)
            else:
                all_files.append(full_path)
    return all_files


def summarize_folder_by_dict(folder):
    summary_file_path = folder + '/summary.txt'
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'w') as file:
            file.write('')
    all_files = get_list_of_files(folder)
    if len(all_files) > 0:
        for file_path in all_files:
            with open(file_path, 'rb') as file:
                file_contents = load(file)

            with open(summary_file_path, 'a') as file:
                file.write(os.path.split(file_path)[1] + '\n')
                for cfg in file_contents[0]:
                    file.write(str(cfg) + '\n')
                for file_content in file_contents[1:]:
                    if isinstance(file_content, dict):
                        file.write(str(file_content) + '\n')
                file.write('\n')


def build_list_exp(exp_cfgs):
    """
    Exp_cfgs is a list of exp_cfg = [data_cfg, model_cfg, optim_cfg]
    Each data_cfg, model_cfg, optim_cfg can have parameters that are list (as in grid_search)
    Therefore for each exp_cfg corresponds all combinations of these parameters
    But the combinations of the parameters of two different exp_cfg (in the list exp_cfgs) won't be combined
    Can be useful if for example one wants to run different algorithms with different parameter names
    At the end, it builds a total list of experiments that can be run on a cluster in parallel
    by assigning different portions of the list to each node

    The function is only used to run experiments on a cluster, all results are saved and then used by other functions
    """
    exp_cfgs_list = list()
    for exp_cfg in exp_cfgs:
        params_grid = dict()
        for cfg in exp_cfg.values():
            for key, value in cfg.items():
                if isinstance(value, list):
                    params_grid.update({key: value})
        params_list = build_list_from_grid(params_grid)
        for params in params_list:
            exp_cfg = {key: set_cfg(cfg, params) for key, cfg in exp_cfg.items()}
            exp_cfgs_list.append(exp_cfg)
    return exp_cfgs_list


def build_list_from_grid(params_grid):
    """
    Create a list of parameters from a grid of any size
    :param params_grid: (dict) dictionary containing parameters name and their range on which the grid search is done.
                    e.g. params_grid = dict(step_size = [1,2,3], line_search=['armijo', 'wolfe'])
    :return:
        params_list: (list) list of all possible configurations of the parameters given in the grid,
                    e.g. params_list[0] = dict(step_size=1, line_search='armijo')
    """
    param_sample0 = {key: None for key in params_grid.keys()}
    params_list = [param_sample0]
    for param_name, param_range in params_grid.items():
        new_params_list = []
        for param_sample in params_list:
            for param in param_range:
                new_param_sample = deepcopy(param_sample)
                new_param_sample[param_name] = param
                new_params_list.append(new_param_sample)
        params_list = deepcopy(new_params_list)
    return params_list


def build_grid_from_cfg(exp_cfg):
    params_grid = dict()
    for cfg in exp_cfg.values():
        for key, value in cfg.items():
            if isinstance(value, list):
                params_grid.update({key: value})
    return params_grid


def set_cfg(default_cfg, given_params):
    """
    Set cfg with the given_params
    :param default_cfg: (dict) one of data_cfg, model_cfg, optim_cfg dictionaries
    :param given_params: (dict) params to include in the default_cfg
                        e.g. given_params = dict(step_size=0, line_search='wolfe')
    :return:
        cfg_to_test: (dict) updated cfg with the given params
    """
    cfg_to_test = deepcopy(default_cfg)
    for param_key in default_cfg.keys():
        if param_key in given_params.keys():
            cfg_to_test[param_key] = given_params[param_key]
    return cfg_to_test