import numpy as np
import scipy
from ._interp import _get_configs, _pdf_1_n, _pdf_n_n, _cdf_1_n, _cdf_n_n, _ppf_1_n, _ppf_n_n
from ._interp import _cdf_1_n_local, _cdf_n_n_local, _broaden_configs


__all__ = ['get_configs', 'pdf', 'cdf', 'ppf', 'sample', 'broaden', 'Interp1D']


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
I_CDFS = 1
I_SPLIT_FACTORS = 2
I_SPLIT_FACTORS_2 = 3
I_TYPES = 4
I_DYDXS = 5
I_DPDXS = 6
I_EXPAS_0 = 7
I_EXPAS_1 = 8
N_CONFIG_INDICES = 9


def get_configs(knots, cdfs, p_tail_limit=0.6, split_threshold=1e-2):
    knots = np.ascontiguousarray(np.atleast_2d(knots), dtype=np.float64)
    cdfs = np.ascontiguousarray(np.atleast_2d(cdfs), dtype=np.float64)
    # assert knots.shape[1] == cdfs.shape[1] >= 4
    assert (np.min(np.sum(np.isfinite(knots), axis=1)) ==
            np.min(np.sum(np.isfinite(cdfs), axis=1)) >= 2)
    assert knots.ndim == cdfs.ndim == 2
    if knots.shape[0] == cdfs.shape[0]:
        pass
    elif knots.shape[0] == 1 and cdfs.shape[0] > 1:
        knots = np.repeat(knots, cdfs.shape[0], axis=0)
    elif cdfs.shape[0] == 1 and knots.shape[0] > 1:
        cdfs = np.repeat(cdfs, knots.shape[0], axis=0)
    else:
        raise ValueError('the shapes of knots and cdfs do not match.')
    configs = np.full((knots.shape[0], N_CONFIG_INDICES, knots.shape[1]), np.nan, dtype=np.float64)
    _get_configs(knots, cdfs, configs, knots.shape[0], knots.shape[1], float(p_tail_limit),
                 float(split_threshold))
    return configs


def _check_configs(configs):
    configs = np.ascontiguousarray(configs, dtype=np.float64)
    if configs.ndim == 2:
        configs = configs[np.newaxis]
    assert configs.ndim == 3 and configs.shape[1] == N_CONFIG_INDICES
    return configs


def _check_input(configs, x):
    configs = _check_configs(configs)
    x = np.atleast_1d(np.ascontiguousarray(x, dtype=np.float64))
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


def cdf(configs, x, local=False):
    configs, x, y = _check_input(configs, x)
    if configs.shape[0] == 1:
        _cdf_1_n(configs, x, y, x.shape[0], configs.shape[2])
        if local:
            _cdf_1_n_local(configs, y, y.shape[0], configs.shape[2])
    elif configs.shape[0] == x.shape[0]:
        _cdf_n_n(configs, x, y, x.shape[0], configs.shape[2])
        if local:
            _cdf_n_n_local(configs, y, y.shape[0], configs.shape[2])
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


def broaden(configs, broadening_factor=1.1, p_tail_limit=0.6, split_threshold=1e-2):
    flagged_cache = np.full((configs.shape[0], 3, configs.shape[2]), np.nan, dtype=np.float64)
    _broaden_configs(configs, configs.shape[0], configs.shape[2], flagged_cache, broadening_factor)
    return get_configs(knots=flagged_cache[:, 0, :], cdfs=flagged_cache[:, 1, :],
                       p_tail_limit=p_tail_limit, split_threshold=split_threshold)


class Interp1D:
    def __init__(self, knots=None, cdfs=None, p_tail_limit=0.6, split_threshold=1e-2, configs=None):
        if configs is not None:
            self.configs = _check_configs(configs)
        else:
            self.configs = get_configs(knots, cdfs, p_tail_limit, split_threshold)
        self.p_tail_limit = float(p_tail_limit)
        self.split_threshold = float(split_threshold)
        self._ok = True # need this for nqe._QuantileInterp1D

    def pdf(self, x, broadening_factor=None):
        if self._ok:
            if broadening_factor is not None and broadening_factor != 1.:
                return self.broaden(broadening_factor).pdf(x=x)
            else:
                return pdf(self.configs, x)
        else:
            raise RuntimeError('this Interp1D has not been fitted.')

    def cdf(self, x, local=False, broadening_factor=None):
        if self._ok:
            if broadening_factor is not None and broadening_factor != 1.:
                return self.broaden(broadening_factor).cdf(x=x, local=local)
            else:
                return cdf(self.configs, x, local)
        else:
            raise RuntimeError('this Interp1D has not been fitted.')

    def ppf(self, y, broadening_factor=None):
        if self._ok:
            if broadening_factor is not None and broadening_factor != 1.:
                return self.broaden(broadening_factor).ppf(y=y)
            else:
                return ppf(self.configs, y)
        else:
            raise RuntimeError('this Interp1D has not been fitted.')

    def sample(self, n=1, random_seed=None, sobol=True, i=None, d=None, broadening_factor=None):
        if self._ok:
            if broadening_factor is not None and broadening_factor != 1.:
                return self.broaden(broadening_factor).sample(n=n, random_seed=random_seed,
                                                              sobol=sobol, i=i, d=d)
            else:
                return sample(self.configs, n, random_seed, sobol, i, d)
        else:
            raise RuntimeError('this Interp1D has not been fitted.')

    def broaden(self, broadening_factor=1.1):
        return Interp1D(configs=broaden(self.configs, broadening_factor, self.p_tail_limit,
                                        self.split_threshold))
