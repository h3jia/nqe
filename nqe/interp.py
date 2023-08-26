import numpy as np
import scipy
from ._interp import _get_configs, _pdf_1_n, _pdf_n_n, _cdf_1_n, _cdf_n_n, _ppf_1_n, _ppf_n_n


__all__ = ['get_configs', 'pdf', 'cdf', 'ppf', 'sample', 'Interp1D']


# type table
UNDEFINED = 0
NORMAL_CUBIC = 1
LEFT_END_CUBIC = 2
RIGHT_END_CUBIC = 3
LINEAR = 4
DOUBLE_EXP = 5
LEFT_END_EXP = 6
RIGHT_END_EXP = 7
# MERGE_EXP = 8


# config index table
I_KNOTS = 0
I_QUANTILES = 1
I_SPLIT_FACTORS = 2
I_SPLIT_FACTORS_2 = 3
I_TYPES = 4
I_DYDXS = 5
I_DPDXS = 6
I_EXPAS_0 = 7
I_EXPAS_1 = 8
N_CONFIG_INDICES = 9


def get_configs(knots, quantiles, split_threshold=1e-2):
    knots = np.atleast_2d(knots).astype(np.float64)
    quantiles = np.atleast_2d(quantiles).astype(np.float64)
    assert knots.shape[1] == quantiles.shape[1] >= 4
    assert knots.ndim == quantiles.ndim == 2
    if knots.shape[0] == quantiles.shape[0]:
        pass
    elif knots.shape[0] == 1 and quantiles.shape[0] > 1:
        knots = np.repeat(knots, quantiles.shape[0], axis=0)
    elif quantiles.shape[0] == 1 and knots.shape[0] > 1:
        quantiles = np.repeat(quantiles, knots.shape[0], axis=0)
    else:
        raise ValueError('the shapes of knots and quantiles do not match.')
    configs = np.full((knots.shape[0], N_CONFIG_INDICES, knots.shape[1]), np.nan, dtype=np.float64)
    _get_configs(knots, quantiles, configs, knots.shape[0], knots.shape[1], split_threshold)
    return configs


def _check_configs(configs):
    configs = np.asarray(configs, dtype=np.float64)
    if configs.ndim == 2:
        configs = configs[np.newaxis]
    assert configs.ndim == 3 and configs.shape[1] == N_CONFIG_INDICES
    return configs


def _check_input(configs, x):
    configs = _check_configs(configs)
    x = np.atleast_1d(x).astype(np.float64)
    assert x.ndim == 1
    return configs, x, np.empty_like(x)


def pdf(configs, x):
    configs, x, y = _check_input(configs, x)
    if configs.shape[0] == 1:
        _pdf_1_n(configs, x, y, x.shape[0], configs.shape[2])
    elif configs.shape[0] == x.shape[0]:
        _pdf_n_n(configs, x, y, x.shape[0], configs.shape[2])
    else:
        raise ValueError('the shapes of configs and x do not match.')
    return y


def cdf(configs, x):
    configs, x, y = _check_input(configs, x)
    if configs.shape[0] == 1:
        _cdf_1_n(configs, x, y, x.shape[0], configs.shape[2])
    elif configs.shape[0] == x.shape[0]:
        _cdf_n_n(configs, x, y, x.shape[0], configs.shape[2])
    else:
        raise ValueError('the shapes of configs and x do not match.')
    return y


def ppf(configs, y):
    configs, y, x = _check_input(configs, y)
    if configs.shape[0] == 1:
        _ppf_1_n(configs, x, y, x.shape[0], configs.shape[2])
    elif configs.shape[0] == y.shape[0]:
        _ppf_n_n(configs, x, y, x.shape[0], configs.shape[2])
    else:
        raise ValueError('the shapes of configs and y do not match.')
    return x


def sample(configs, n=1, random_seed=None, sobol=True, i=None, d=None):
    configs = _check_configs(configs)
    n = int(n)
    i = 0 if i is None else int(i)
    d = 1 if d is None else int(d)
    if configs.shape[0] == 1 or configs.shape[0] == n:
        if sobol:
            return ppf(configs, scipy.stats.qmc.Sobol(
                d, scramble=True, seed=random_seed, bits=64).random(n)[:, i])
        else:
            return ppf(configs, np.random.default_rng(random_seed).uniform(size=n).reshape(-1))
    else:
        raise NotImplementedError('currently only supports configs.shape[0] == 1 or '
                                  'configs.shape[0] == n.')


class Interp1D:
    def __init__(self, knots=None, quantiles=None, split_threshold=1e-2, configs=None):
        if configs is not None:
            self.configs = _check_configs(configs)
        else:
            self.configs = get_configs(knots, quantiles, split_threshold)

    def pdf(self, x):
        return pdf(self.configs, x)

    def cdf(self, x):
        return cdf(self.configs, x)

    def ppf(self, y):
        return ppf(self.configs, y)

    def sample(self, n=1, x=None, theta=None, random_seed=None, sobol=True, i=None, d=None,
               batch_size=None, device='cpu'):
        return sample(self.configs, n, random_seed, sobol, i, d)
