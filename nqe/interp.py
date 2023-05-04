import numpy as np

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


class Interp1D:
    def __init__(self, knots, quantiles):
        self.knots = np.asarray(knots, dtype=float).reshape(-1)
        self.quantiles = np.asarray(quantiles, dtype=float).reshape(-1)
        assert self.knots.shape == self.quantiles.shape
        assert self.quantiles[0] == 0.
        assert self.quantiles[1] == 1.
        self.n_interval = self.knots.shape - 1
        assert self.n_interval >= 4
        self.dydx = np.full(self.n_interval + 1, np.nan)
        self.split_factors = np.full(self.n_interval, -np.inf)
        self.types = np.full(self.n_interval, UNDEFINED)
        self.types[0] = LEFT_END_EXP
        self.types[-1] = RIGHT_END_EXP
        self.ready = np.full(self.n_interval, 0)
        self.exp_as = np.zeros((self.n_interval, 2))

    def set_all(self):
        raise NotImplementedError

    def solve(self, quantile):
        assert 0. <= quantile <= 1.
