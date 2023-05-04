import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from scipy.special.cython_special cimport erf, erfi
from libc.math cimport exp, sqrt, pi, fabs, fmax
from scipy.optimize.cython_optimize cimport brentq
cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"


__all__ = ['find_interval', 'get_split_factors']


ctypedef struct int_p_dx_params:
    double h
    double p0
    double dpdx0
    double mass

cdef double XTOL = 1e-8, RTOL = 1e-8
cdef int MITR = 200


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_split_factors(double* split_factors, const double* knots, const double* quantiles,
                               int n_interval):
    cdef size_t i
    for i in prange(n_interval, nogil=True, schedule='static'):
        split_factors[i] = _get_split_factor(&knots[i], &quantiles[i])


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_split_factor(const double* knots, const double* quantiles) nogil:
    cdef double y0 = quantiles[1], y1 = quantiles[2], y2 = quantiles[3], y3 = quantiles[4]
    cdef double h0 = knots[1] - knots[0], h1 = knots[2] - knots[1], h2 = knots[3] - knots[2]
    cdef double h3 = knots[4] - knots[3], h4 = knots[5] - knots[4]
    cdef double m0 = (quantiles[1] - quantiles[0]) / h0, m1 = (quantiles[2] - quantiles[1]) / h1
    cdef double m2 = (quantiles[3] - quantiles[2]) / h2, m3 = (quantiles[4] - quantiles[3]) / h3
    cdef double m4 = (quantiles[5] - quantiles[4]) / h4
    cdef double dydx0 = _get_dydx_2(h0, h1, m0, m1)
    cdef double dydx1 = _get_dydx_1(h1, h0, m1, m0)
    cdef double dydx2 = _get_dydx_1(h3, h4, m3, m4)
    cdef double dydx3 = _get_dydx_2(h3, h4, m3, m4)
    cdef double dpdx1 = 6. * y0 + 2. * dydx0 - 6. * y1 + 4. * dydx1
    cdef double dpdx2 = -1. * (-6. * y2 - 4. * dydx2 + 6. * y3 - 2. * dydx3)
    cdef double a1 = _solve_single_exp(h2, dydx1, dpdx1, 0.5 * (y2 - y1))
    cdef double a2 = _solve_single_exp(h2, dydx2, dpdx2, 0.5 * (y2 - y1))
    cdef double p1a = dydx2 * exp(a2 * h2 * h2 + dpdx2 / dydx2 * h2)
    cdef double p2a = dydx1 * exp(a1 * h2 * h2 + dpdx1 / dydx1 * h2)
    return fmax(p1a / dydx1, p2a / dydx2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _solve_single_exp(double h, double p0, double dpdx0, double mass) nogil:
    cdef int_p_dx_params myargs = {'h': h, 'p0': p0, 'dpdx0': dpdx0, 'mass': mass}
    cdef double a0, a1, tmp
    cdef int i = 0
    tmp = _int_p_dx(0., h, p0, dpdx0, mass)
    if tmp > 0.:
        a1 = 0.
        a0 = -1. / h / h
        tmp = _int_p_dx(a0, h, p0, dpdx0, mass)
        while tmp > 0.:
            a0 *= 3.
            i += 1
            tmp = _int_p_dx(a0, h, p0, dpdx0, mass)
            if i > 200:
                return nan
    elif tmp < 0.:
        a0 = 0.
        a1 = 1. / h / h
        tmp = _int_p_dx(a1, h, p0, dpdx0, mass)
        while tmp < 0.:
            a1 *= 3.
            i += 1
            tmp = _int_p_dx(a1, h, p0, dpdx0, mass)
            if i > 200:
                return nan
    elif tmp == 0.:
        return 0.
    else:
        return nan
    return brentq(_int_p_dx_args, a0, a1, <int_p_dx_params *> &myargs, XTOL, RTOL, MITR, NULL)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _int_p_dx(double a, double h, double p0, double dpdx0, double mass) nogil:
    cdef double absa = fabs(a)
    if dpdx0 == 0:
        if a > 0:
            return sqrt(pi) * p0 / 2. / sqrt(absa) * erfi(sqrt(absa) * h) - mass
        elif a == 0:
            return p0 * h - mass
        elif a < 0:
            return sqrt(pi) * p0 / 2. / sqrt(absa) * erf(sqrt(absa) * h) - mass
        else:
            return nan
    else:
        if a > 0:
            return (
                sqrt(pi) * p0 / 2. / sqrt(absa) * exp(-dpdx0 * dpdx0 / 4. / absa / p0 / p0) * (
                    erfi((2 * absa * p0 * h + dpdx0) / 2. / sqrt(absa) / p0) -
                    erfi(dpdx0 / 2. / sqrt(absa) / p0)
                ) - mass
            )
        elif a == 0:
            return p0 * p0 / dpdx0 * (exp(dpdx0 / p0 * h) - 1) - mass
        elif a < 0:
            return (
                sqrt(pi) * p0 / 2. / sqrt(absa) * exp(dpdx0 * dpdx0 / 4. / absa / p0 / p0) * (
                    erf((2 * absa * p0 * h - dpdx0) / 2. / sqrt(absa) / p0) +
                    erf(dpdx0 / 2. / sqrt(absa) / p0)
                ) - mass
            )
        else:
            return nan


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _int_p_dx_args(double a, void *args):
    cdef int_p_dx_params *myargs = <int_p_dx_params *> args
    return _int_p_dx(a, myargs.h, myargs.p0, myargs.dpdx0, myargs.mass)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_dydx_1(double h0, double h1, double m0, double m1) nogil:
    cdef d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if d * m0 <= 0.:
        return 0.
    elif m0 * m1 <= 0. and fabs(d) > 3. * fabs(m0):
        return 3. * m0
    return d


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_dydx_2(double h0, double h1, double m0, double m1) nogil:
    cdef double alpha
    if m0 * m1 <= 0.:
        return 0.
    else:
        alpha = (h0 + 2. * h1) / 3. / (h0 + h1)
        return m0 * m1 / (alpha * m1 + (1 - alpha) * m0)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_split_factors(double[::1] split_factors, const double[::1] knots,
                      const double[::1] quantiles, int n_interval, int i_start, int i_end):
    if i_start < 2:
        i_start = 2
    if i_end > n_interval - 2:
        i_end = n_interval - 2
    _get_split_factors(&split_factors[i_start], &knots[i_start - 2], &quantiles[i_start - 2],
                       i_end - i_start)
