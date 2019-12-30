import configparser

from src.model.ckn import ckn_layer


def load_config(path):
    """
    Load and parse the CKN configuration from the specified filepath.

    :param path: Filepath where the configuration is stored
    :return params: Dictionary of parameters formed from the config file
    """
    default_param_dict = {'pad': '0',
                          'patch_size': '(3, 3)',
                          'stride': '[1, 1]',
                          'precomputed_patches': 'False',
                          'whiten': 'False',
                          'patch_kernel': 'rbf_sphere',
                          'filters_init': 'spherical-k-means',
                          'normalize': 'True',
                          'patch_sigma': '0.6',
                          'num_filters': '128',
                          'pool_method': 'average',
                          'pool_dim': '(1, 1)',
                          'subsample_factor': '(1, 1)',
                          'store_normalization': 'False',
                          'kww_reg': '0.001',
                          'num_newton_iters': '20',
                          }

    config = configparser.ConfigParser(default_param_dict)
    config.read_file(open(path))

    int_args = ['pad', 'num_filters', 'num_newton_iters']
    float_args = ['patch_sigma', 'kww_reg']
    str_args = ['patch_kernel', 'filters_init', 'pool_method']
    bool_args = ['precomputed_patches', 'whiten', 'normalize', 'store_normalization']
    list_int_args = ['patch_size', 'stride', 'pool_dim', 'subsample_factor']

    params = {}
    for arg_list, key_type in zip([int_args, float_args, str_args], [int, float, str]):
        for key in arg_list:
            params[key] = list(map(key_type, [config.get(section, key) for section in config.sections()]))

    for key in bool_args:
        values = [config.get(section, key) for section in config.sections()]
        params[key] = [values[i].lower() == 'true' for i in range(len(values))]

    for key in list_int_args:
        values = [config.get(section, key) for section in config.sections()]
        params[key] = [eval(values[i]) for i in range(len(values))]

    return params


def create_layers(params):
    """
    Create the layers of a CKN based on the input parameters.

    :param params: Parameters of the CKN from a config file
    :return layers: List of layers to use to create a CKN
    """
    n_layers = len(params['num_filters'])
    layers = []
    for layer_num in range(n_layers):
        layer = ckn_layer.CKNLayer(layer_num,
                                   params['patch_size'][layer_num],
                                   params['patch_kernel'][layer_num],
                                   params['num_filters'][layer_num],
                                   params['subsample_factor'][layer_num],
                                   padding=params['pad'][layer_num],
                                   stride=params['stride'][layer_num],
                                   precomputed_patches=params['precomputed_patches'][layer_num],
                                   whiten=params['whiten'][layer_num],
                                   filters_init=params['filters_init'][layer_num],
                                   normalize=params['normalize'][layer_num],
                                   patch_sigma=params['patch_sigma'][layer_num],
                                   pool_method=params['pool_method'][layer_num],
                                   pool_dim=params['pool_dim'][layer_num],
                                   store_normalization=params['store_normalization'][layer_num],
                                   kww_reg=params['kww_reg'][layer_num],
                                   num_newton_iters=params['num_newton_iters'][layer_num],
                                   )
        layers.append(layer)

    return layers
