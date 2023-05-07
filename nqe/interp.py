import numpy as np
from ._interp import get_split_factors, get_types, get_dydxs, get_exps, get_pdf, get_cdf


__all__ = ['Interp1D']


# type table
UNDEFINED = 0
NORMAL_CUBIC = 1
LEFT_END_CUBIC = 2
RIGHT_END_CUBIC = 3
LINEAR = 4
DOUBLE_EXP = 5
LEFT_END_EXP = 6
RIGHT_END_EXP = 7
MERGE_EXP = 8


class Interp1D:
    def __init__(self, knots, quantiles, split_threshold=1e-4):
        self.knots = np.asarray(knots, dtype=float).reshape(-1)
        self.quantiles = np.asarray(quantiles, dtype=float).reshape(-1)
        assert self.knots.shape == self.quantiles.shape
        assert self.knots.size >= 4
        assert self.quantiles[0] == 0.
        assert self.quantiles[-1] == 1.
        self.n_interval = self.knots.size - 1
        assert self.n_interval >= 4
        self._knots = np.copy(self.knots)         # make a copy since some intervals
        self._quantiles = np.copy(self.quantiles) # can be merged
        self.split_factors = np.full(self.n_interval, np.nan)
        self.split_factors[:2] = np.inf
        self.split_factors[-2:] = np.inf
        self.split_factors_2 = np.full(self.n_interval - 1, np.nan)
        self.split_factors_2[:2] = np.inf
        self.split_factors_2[-2:] = np.inf
        self.types = np.full(self.n_interval, UNDEFINED, dtype=np.int32)
        # self.types[0] = LEFT_END_EXP
        # self.types[-1] = RIGHT_END_EXP
        self.dydxs = np.full(self.n_interval + 1, np.nan)
        self.dpdxs = np.full(self.n_interval + 1, np.nan)
        self.expas = np.full((self.n_interval, 2), np.nan)
        self.split_threshold = float(split_threshold)

    def set_interval(self, i_start, i_end):
        assert i_start >= 0 and i_end <= self.n_interval and i_start < i_end
        get_split_factors(self.split_factors, self.split_factors_2, self.knots, self.quantiles,
                          self.n_interval, max(i_start - 4, 2), min(i_end + 4, self.n_interval - 2))
        get_types(self.types, self.split_factors, self.split_factors_2, self.n_interval,
                  i_start - 2, i_end + 2, self.split_threshold)
        i_merge = np.where(self.types == MERGE_EXP)[0]
        if i_merge.size > 0:
            self.knots = np.delete(self.knots, i_merge)
            self.quantiles = np.delete(self.quantiles, i_merge)
            self.n_interval -= i_merge.size
            self.types = np.delete(self.types, i_merge)
            self.dydxs = np.delete(self.dydxs, i_merge)
            self.dpdxs = np.delete(self.dpdxs, i_merge)
            self.expas = np.delete(self.expas, i_merge, axis=0)
        get_dydxs(self.dydxs, self.knots, self.quantiles, self.types, self.n_interval, i_start,
                  i_end)
        get_exps(self.expas, self.dpdxs, self.knots, self.quantiles, self.dydxs, self.types,
                 self.n_interval, i_start, i_end)

    def set_all(self):
        self.set_interval(0, self.n_interval)

    def check(self, x):
        xmin = np.min(x)
        xmax = np.max(x)
        # assert xmin >= self.knots[0] and xmax <= self.knots[-1]
        imin = np.searchsorted(self.knots, xmin)
        imax = np.searchsorted(self.knots, xmax)
        if not np.all(self.types[imin:(imax + 1)]):
            self.set_interval(imin, imax + 1)

    def pdf(self, x, check=True):
        x = np.asarray(x, dtype=float)
        if x.ndim <= 1 and x.size > 0:
            if check:
                self.check(x)
            out = np.empty_like(np.atleast_1d(x))
            get_pdf(np.atleast_1d(x), out, self.knots, self.quantiles, self.dydxs, self.dpdxs,
                    self.expas, self.types, out.size, self.n_interval)
            return out if x.ndim == 1 else float(out)
        else:
            raise NotImplementedError

    def cdf(self, x, check=True):
        x = np.asarray(x, dtype=float)
        if x.ndim <= 1 and x.size > 0:
            if check:
                self.check(x)
            out = np.empty_like(np.atleast_1d(x))
            get_cdf(np.atleast_1d(x), out, self.knots, self.quantiles, self.dydxs, self.dpdxs,
                    self.expas, self.types, out.size, self.n_interval)
            return out if x.ndim == 1 else float(out)
        else:
            raise NotImplementedError

    def solve(self, quantile):
        assert 0. <= quantile <= 1.
