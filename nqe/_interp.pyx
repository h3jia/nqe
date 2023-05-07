import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.parallel cimport prange
from scipy.special.cython_special cimport erf, erfi, erfcx, dawsn
from libc.stdlib cimport malloc, free
from libc.math cimport exp, sqrt, pi, fabs, fmax, fmin
from scipy.optimize.cython_optimize cimport brentq
cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"
    double inf "NPY_INFINITY"


__all__ = ['get_split_factors', 'get_types', 'get_dydxs', 'get_exps', 'get_pdf', 'get_cdf']


ctypedef struct int_p_dx_params_expa:
    double h
    double p0
    double dpdx0
    double mass

ctypedef struct int_p_dx_params_dpdx:
    double h
    double p0
    double mass

cdef double XTOL = 1e-10, RTOL = 1e-10
cdef int MITR = 300
cdef int UNDEFINED = 0, NORMAL_CUBIC = 1, LEFT_END_CUBIC = 2, RIGHT_END_CUBIC = 3, LINEAR = 4
cdef int DOUBLE_EXP = 5, LEFT_END_EXP = 6, RIGHT_END_EXP = 7, MERGE_EXP = 8


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline int _max(int a, int b) nogil:
    return a if a > b else b


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline int _min(int a, int b) nogil:
    return a if a < b else b


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int find_interval(const double* x, int m, double xval, int prev_interval=-1) nogil:
    """
    Find an interval such that x[interval - 1] <= xval < x[interval].

    Assumeing that x is sorted in the ascending order. If xval < x[0], then interval = 0, if xval >
    x[-1] then interval = m.

    Parameters
    ----------
    x : ndarray of double, shape (m,)
        Piecewise polynomial breakpoints sorted in ascending order.
    m : int
        Shape of x.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.

    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.
    """
    cdef int high, low, mid, interval
    cdef double a, b

    a = x[0]
    b = x[m - 1]

    interval = prev_interval
    if interval < 0 or interval > m:
        interval = m // 2

    if not (a <= xval < b):
        if xval < a:
            # below
            interval = 0
        elif xval >= b:
            # above
            interval = m
        else:
            # nan
            interval = -1
    else:
        # Find the interval the coordinate is in (binary search with locality)
        if xval >= x[interval - 1]:
            low = interval
            high = m - 1
        else:
            low = 1
            high = interval - 1

        if xval < x[low]:
            high = low

        while low < high:
            mid = (high + low) // 2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 2
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid + 1
                break

        interval = low

    return interval


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_split_factors(double* split_factors, const double* knots, const double* quantiles,
                               int n_interval, int central_width=1):
    cdef size_t i
    for i in prange(n_interval, nogil=True, schedule='static'):
        split_factors[i] = _get_split_factor(&knots[i], &quantiles[i], central_width)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_split_factor(const double* knots, const double* quantiles, int central_width=1) nogil:
    cdef int offset = central_width - 1
    cdef double y0 = quantiles[1], y1 = quantiles[2]
    cdef double y2 = quantiles[3 + offset], y3 = quantiles[4 + offset]
    cdef double h0 = knots[1] - knots[0], h1 = knots[2] - knots[1]
    cdef double h2 = knots[3 + offset] - knots[2], h3 = knots[4 + offset] - knots[3 + offset]
    cdef double h4 = knots[5 + offset] - knots[4 + offset]
    cdef double m0 = (quantiles[1] - quantiles[0]) / h0, m1 = (quantiles[2] - quantiles[1]) / h1
    cdef double m2 = (quantiles[3 + offset] - quantiles[2]) / h2
    cdef double m3 = (quantiles[4 + offset] - quantiles[3 + offset]) / h3
    cdef double m4 = (quantiles[5 + offset] - quantiles[4 + offset]) / h4
    cdef double dydx0 = _get_dydx_2(h0, h1, m0, m1)
    cdef double dydx1 = _get_dydx_1(h1, h0, m1, m0)
    cdef double dydx2 = _get_dydx_1(h3, h4, m3, m4)
    cdef double dydx3 = _get_dydx_2(h3, h4, m3, m4)
    cdef double dpdx1, dpdx2, a1, a2, p1a, p2a
    if dydx1 > 0. and dydx2 > 0.:
        dpdx1 = _get_right_end_dpdx(h1, y0, y1, dydx0, dydx1)
        a1 = _solve_single_expa(h2, dydx1, dpdx1, 0.5 * (y2 - y1))
        if dpdx1 * a1 < 0.:
            a1 = 0.
            dpdx1 = _solve_single_dpdx(h2, dydx1, dpdx1, 0.5 * (y2 - y1))
        dpdx2 = _get_left_end_dpdx(h3, y2, y3, dydx2, dydx3)
        a2 = _solve_single_expa(h2, dydx2, -dpdx2, 0.5 * (y2 - y1))
        if dpdx2 * a2 > 0.:
            a2 = 0.
            dpdx2 = -_solve_single_dpdx(h2, dydx2, -dpdx2, 0.5 * (y2 - y1))
        p1a = dydx2 * exp(a2 * h2 * h2 - dpdx2 / dydx2 * h2)
        p2a = dydx1 * exp(a1 * h2 * h2 + dpdx1 / dydx1 * h2)
        return fmax(p1a / dydx1, p2a / dydx2)
    else:
        return inf


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _get_left_end_dpdx(double h, double y0, double y1, double dydx0,
                                      double dydx1) nogil:
    return (-6. * y0 / h - 4. * dydx0 + 6. * y1 / h - 2. * dydx1) / h


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _get_right_end_dpdx(double h, double y0, double y1, double dydx0,
                                       double dydx1) nogil:
    return (6. * y0 / h + 2. * dydx0 - 6. * y1 / h + 4. * dydx1) / h


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _solve_single_expa(double h, double p0, double dpdx0, double mass) nogil:
    cdef int_p_dx_params_expa myargs = {'h': h, 'p0': p0, 'dpdx0': dpdx0, 'mass': mass}
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
            if i > MITR:
                return nan
    elif tmp < 0.:
        a0 = 0.
        a1 = 1. / h / h
        tmp = _int_p_dx(a1, h, p0, dpdx0, mass)
        while tmp < 0.:
            a1 *= 3.
            i += 1
            tmp = _int_p_dx(a1, h, p0, dpdx0, mass)
            if i > MITR:
                return nan
    elif tmp == 0.:
        return 0.
    else:
        return nan
    return brentq(_int_p_dx_args_expa, a0, a1, <int_p_dx_params_expa *> &myargs, XTOL, RTOL, MITR,
                  NULL)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _solve_single_dpdx(double h, double p0, double dpdx0, double mass) nogil:
    cdef int_p_dx_params_dpdx myargs = {'h': h, 'p0': p0, 'mass': mass}
    cdef double dpdx1, dpdx2
    cdef size_t i
    if dpdx0 == 0.:
        dpdx1 = -1.
        dpdx2 = 1.
    else:
        dpdx1 = -3. * fabs(dpdx0)
        dpdx2 = 3. * fabs(dpdx0)
    for i in range(MITR + 1):
        if _int_p_dx(0., h, p0, dpdx1, mass) < 0.:
            break
        dpdx1 *= 3.
        if i == MITR:
            return nan
    for i in range(MITR + 1):
        if _int_p_dx(0., h, p0, dpdx2, mass) > 0.:
            break
        dpdx2 *= 3.
        if i == MITR:
            return nan
    return brentq(_int_p_dx_args_dpdx, dpdx1, dpdx2, <int_p_dx_params_dpdx *> &myargs, XTOL, RTOL,
                  MITR, NULL)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _int_p_dx(double a, double h, double p0, double dpdx0, double mass) nogil:
    cdef double absa = fabs(a)
    if fabs(dpdx0 / p0 * h) < 1e-8:
        if a > 0:
            return sqrt(pi) * p0 / 2. / sqrt(absa) * erfi(sqrt(absa) * h) - mass
        elif a == 0:
            return p0 * h - mass
        elif a < 0:
            return sqrt(pi) * p0 / 2. / sqrt(absa) * erf(sqrt(absa) * h) - mass
        else:
            return nan
    else:
        if sqrt(absa) < fabs(dpdx0 / 2. / p0) * 1e-10:
            return p0 * p0 / dpdx0 * (exp(dpdx0 / p0 * h) - 1) - mass
        elif a > 0:
            return (
                p0 / sqrt(absa) * (
                    exp(absa * h * h + dpdx0 * h / p0) *
                    dawsn((2. * absa * p0 * h + dpdx0) / 2. / sqrt(absa) / p0) -
                    dawsn(dpdx0 / 2. / sqrt(absa) / p0)
                ) - mass
            )
        elif a < 0:
            # erfcx likes positive inputs
            if dpdx0 > 0.:
                return (
                    sqrt(pi) * p0 / 2. / sqrt(absa) * (
                        exp(-absa * h * h + dpdx0 * h / p0) *
                        erfcx((-2 * absa * p0 * h + dpdx0) / 2. / sqrt(absa) / p0) -
                        erfcx(dpdx0 / 2. / sqrt(absa) / p0)
                    ) - mass
                )
            else:
                return (
                    sqrt(pi) * p0 / 2. / sqrt(absa) * (
                        -exp(-absa * h * h + dpdx0 * h / p0) *
                        erfcx((2 * absa * p0 * h - dpdx0) / 2. / sqrt(absa) / p0) +
                        erfcx(-dpdx0 / 2. / sqrt(absa) / p0)
                    ) - mass
                )
        else:
            return nan


"""@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _int_p_dx(double a, double h, double p0, double dpdx0, double mass) nogil:
    cdef double absa = fabs(a)
    if fabs(dpdx0 / p0 * h) < 1e-8:
        if a > 0:
            return sqrt(pi) * p0 / 2. / sqrt(absa) * erfi(sqrt(absa) * h) - mass
        elif a == 0:
            return p0 * h - mass
        elif a < 0:
            return sqrt(pi) * p0 / 2. / sqrt(absa) * erf(sqrt(absa) * h) - mass
        else:
            return nan
    else:
        if absa < fabs(dpdx0 * dpdx0 / 4. / p0 / p0) / 600.: # avoid computing exp(>600)
            return p0 * p0 / dpdx0 * (exp(dpdx0 / p0 * h) - 1) - mass
        elif a > 0:
            return (
                sqrt(pi) * p0 / 2. / sqrt(absa) * exp(-dpdx0 * dpdx0 / 4. / absa / p0 / p0) * (
                    erfi((2 * absa * p0 * h + dpdx0) / 2. / sqrt(absa) / p0) -
                    erfi(dpdx0 / 2. / sqrt(absa) / p0)
                ) - mass
            )
        elif a < 0:
            return (
                sqrt(pi) * p0 / 2. / sqrt(absa) * exp(dpdx0 * dpdx0 / 4. / absa / p0 / p0) * (
                    erf((2 * absa * p0 * h - dpdx0) / 2. / sqrt(absa) / p0) +
                    erf(dpdx0 / 2. / sqrt(absa) / p0)
                ) - mass
            )
        else:
            return nan"""


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _int_p_dx_args_expa(double a, void *args):
    cdef int_p_dx_params_expa *myargs = <int_p_dx_params_expa *> args
    return _int_p_dx(a, myargs.h, myargs.p0, myargs.dpdx0, myargs.mass)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _int_p_dx_args_dpdx(double dpdx, void *args):
    cdef int_p_dx_params_dpdx *myargs = <int_p_dx_params_dpdx *> args
    return _int_p_dx(0., myargs.h, myargs.p0, dpdx, myargs.mass)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_dydx_1(double h0, double h1, double m0, double m1) nogil:
    cdef double d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
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
cdef void _get_types(int* types):
    if types[1] == UNDEFINED:
        if types[0] not in (DOUBLE_EXP, MERGE_EXP) and types[2] not in (DOUBLE_EXP, MERGE_EXP):
            types[1] = NORMAL_CUBIC
        elif types[0] in (DOUBLE_EXP, MERGE_EXP) and types[2] not in (DOUBLE_EXP, MERGE_EXP):
            types[1] = LEFT_END_CUBIC
        elif types[0] not in (DOUBLE_EXP, MERGE_EXP) and types[2] in (DOUBLE_EXP, MERGE_EXP):
            types[1] = RIGHT_END_CUBIC
        else:
            types[1] = LINEAR


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_split_factors(double[::1] split_factors, double[::1] split_factors_2,
                      const double[::1] knots, const double[::1] quantiles, int n_interval,
                      int i_start, int i_end):
    if i_start < 2:
        i_start = 2
    if i_end > n_interval - 2:
        i_end = n_interval - 2
    _get_split_factors(&split_factors[i_start], &knots[i_start - 2], &quantiles[i_start - 2],
                       i_end - i_start)
    _get_split_factors(&split_factors_2[i_start], &knots[i_start - 2], &quantiles[i_start - 2],
                       i_end - i_start - 1, 2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_types(int[::1] types, const double[::1] split_factors, const double[::1] split_factors_2,
              int n_interval, int i_start, int i_end, double split_threshold):
    cdef size_t i
    for i in range(_max(i_start, 2), _min(i_end, n_interval - 2)):
        if split_factors[i] < fmin(split_threshold, fmin(split_factors[i - 1],
                                                         split_factors[i + 1])):
            types[i] = DOUBLE_EXP
    for i in range(_max(i_start, 2), _min(i_end, n_interval - 2)):
        if split_factors_2[i] < fmin(split_threshold,
                                     fmin(fmin(split_factors_2[i - 2], split_factors_2[i - 1]),
                                          fmin(split_factors_2[i + 1], split_factors_2[i + 2]))):
            types[i] = DOUBLE_EXP
            types[i + 1] = MERGE_EXP
    for i in range(_max(i_start, 2), _min(i_end, n_interval - 2)):
        _get_types(&types[i - 1])
    if i_start < 2:
        types[0] = LEFT_END_EXP
        types[1] = LEFT_END_CUBIC if types[2] not in (DOUBLE_EXP, RIGHT_END_EXP) else LINEAR
    if i_end > n_interval - 2:
        types[n_interval - 1] = RIGHT_END_EXP
        types[n_interval - 2] = (RIGHT_END_CUBIC if types[n_interval - 3] not in
                                 (DOUBLE_EXP, LEFT_END_EXP, MERGE_EXP) else LINEAR)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_dydxs(double[::1] dydxs, const double[::1] knots, const double[::1] quantiles,
              const int[::1] types, int n_interval, int i_start, int i_end):
    cdef size_t i
    cdef double h0, h1
    for i in prange(_max(i_start, 1), _min(i_end + 1, n_interval), nogil=True,
                    schedule='static'):
        if types[i] == LINEAR:
            dydxs[i] = (quantiles[i + 1] - quantiles[i]) / (knots[i + 1] - knots[i])
        elif types[i - 1] == LINEAR:
            dydxs[i] = (quantiles[i] - quantiles[i - 1]) / (knots[i] - knots[i - 1])
        elif types[i] == NORMAL_CUBIC or types[i] == RIGHT_END_CUBIC:
            h0 = knots[i] - knots[i - 1]
            h1 = knots[i + 1] - knots[i]
            dydxs[i] = _get_dydx_2(h0, h1, (quantiles[i] - quantiles[i - 1]) / h0,
                                   (quantiles[i + 1] - quantiles[i]) / h1)
        elif types[i] == LEFT_END_CUBIC:
            h0 = knots[i + 1] - knots[i]
            h1 = knots[i + 2] - knots[i + 1]
            dydxs[i] = _get_dydx_1(h0, h1, (quantiles[i + 1] - quantiles[i]) / h0,
                                   (quantiles[i + 2] - quantiles[i + 1]) / h1)
        elif types[i] == DOUBLE_EXP or types[i] == RIGHT_END_EXP:
            h0 = knots[i] - knots[i - 1]
            h1 = knots[i - 1] - knots[i - 2]
            dydxs[i] = _get_dydx_1(h0, h1, (quantiles[i] - quantiles[i - 1]) / h0,
                                   (quantiles[i - 1] - quantiles[i - 2]) / h1)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_exps(double[:, ::1] expas, double[::1] dpdxs, const double[::1] knots,
             const double[::1] quantiles, const double[::1] dydxs, const int[::1] types,
             int n_interval, int i_start, int i_end):
    cdef size_t i
    for i in prange(i_start, i_end, nogil=True, schedule='static'):
        if types[i] == DOUBLE_EXP or types[i] == RIGHT_END_EXP:
            dpdxs[i] = _get_right_end_dpdx(knots[i] - knots[i - 1], quantiles[i - 1], quantiles[i],
                                           dydxs[i - 1], dydxs[i])
            expas[i, 0] = _solve_single_expa(
                knots[i + 1] - knots[i], dydxs[i], dpdxs[i],
                (0.5 if types[i] == DOUBLE_EXP else 1.) * (quantiles[i + 1] - quantiles[i])
            )
            if dpdxs[i] * expas[i, 0] < 0.:
                expas[i, 0] = 0.
                dpdxs[i] = _solve_single_dpdx(
                    knots[i + 1] - knots[i], dydxs[i], dpdxs[i],
                    (0.5 if types[i] == DOUBLE_EXP else 1.) * (quantiles[i + 1] - quantiles[i])
                )
        if types[i] == DOUBLE_EXP or types[i] == LEFT_END_EXP:
            dpdxs[i + 1] = _get_left_end_dpdx(knots[i + 2] - knots[i + 1], quantiles[i + 1],
                                              quantiles[i + 2], dydxs[i + 1], dydxs[i + 2])
            expas[i, 1] = _solve_single_expa(
                knots[i + 1] - knots[i], dydxs[i + 1], -dpdxs[i + 1],
                (0.5 if types[i] == DOUBLE_EXP else 1.) * (quantiles[i + 1] - quantiles[i])
            )
            if dpdxs[i + 1] * expas[i, 1] > 0.:
                expas[i, 1] = 0.
                dpdxs[i + 1] = -_solve_single_dpdx(
                    knots[i + 1] - knots[i], dydxs[i + 1], -dpdxs[i + 1],
                    (0.5 if types[i] == DOUBLE_EXP else 1.) * (quantiles[i + 1] - quantiles[i])
                )


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_pdf(const double[::1] xs, double[::1] ys, const double[::1] knots,
            const double[::1] quantiles, const double[::1] dydxs, const double[::1] dpdxs,
            const double[:, ::1] expas, const int[::1] types, int n_point, int n_interval):
    cdef size_t i
    cdef int *j = <int *> malloc(n_point * sizeof(int))
    cdef double *h = <double *> malloc(n_point * sizeof(double))
    cdef double *t = <double *> malloc(n_point * sizeof(double))
    if not j or not h or not t:
        raise MemoryError('cannot malloc required array in get_pdf.')
    try:
        for i in range(n_point): # prange(n_point, nogil=True, schedule='static'):
            j[i] = find_interval(&knots[0], n_interval + 1, xs[i]) - 1
            if j[i] >= 0 and j[i] <= n_interval - 1:
                if (types[j[i]] == NORMAL_CUBIC or types[j[i]] == LEFT_END_CUBIC or
                    types[j[i]] == RIGHT_END_CUBIC or types[j[i]] == LINEAR):
                    h[i] = knots[j[i] + 1] - knots[j[i]]
                    t[i] = (xs[i] - knots[j[i]]) / h[i]
                    ys[i] = (
                        (6. * t[i] * t[i] - 6. * t[i]) / h[i] * quantiles[j[i]] +
                        (3. * t[i] * t[i] - 4. * t[i] + 1.) * dydxs[j[i]] +
                        (-6. * t[i] * t[i] + 6. * t[i]) / h[i] * quantiles[j[i] + 1] +
                        (3. * t[i] * t[i] - 2. * t[i]) * dydxs[j[i] + 1]
                    )
                elif types[j[i]] == DOUBLE_EXP:
                    t[i] = xs[i] - knots[j[i]]
                    ys[i] = dydxs[j[i]] * exp(expas[j[i], 0] * t[i] * t[i] +
                                              dpdxs[j[i]] / dydxs[j[i]] * t[i])
                    t[i] = knots[j[i] + 1] - xs[i]
                    ys[i] += dydxs[j[i] + 1] * exp(expas[j[i], 1] * t[i] * t[i] +
                                                   -dpdxs[j[i] + 1] / dydxs[j[i] + 1] * t[i])
                elif types[j[i]] == LEFT_END_EXP:
                    t[i] = knots[j[i] + 1] - xs[i]
                    ys[i] = dydxs[j[i] + 1] * exp(expas[j[i], 1] * t[i] * t[i] +
                                                  -dpdxs[j[i] + 1] / dydxs[j[i] + 1] * t[i])
                elif types[j[i]] == RIGHT_END_EXP:
                    t[i] = xs[i] - knots[j[i]]
                    ys[i] = dydxs[j[i]] * exp(expas[j[i], 0] * t[i] * t[i] +
                                              dpdxs[j[i]] / dydxs[j[i]] * t[i])
                else:
                    ys[i] = nan
            elif j[i] == -1 or j[i] == n_interval:
                ys[i] = 0.
            else:
                ys[i] = nan
    finally:
        free(j)
        free(h)
        free(t)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_cdf(const double[::1] xs, double[::1] ys, const double[::1] knots,
            const double[::1] quantiles, const double[::1] dydxs, const double[::1] dpdxs,
            const double[:, ::1] expas, const int[::1] types, int n_point, int n_interval):
    cdef size_t i
    cdef int *j = <int *> malloc(n_point * sizeof(int))
    cdef double *h = <double *> malloc(n_point * sizeof(double))
    cdef double *t = <double *> malloc(n_point * sizeof(double))
    if not j or not h or not t:
        raise MemoryError('cannot malloc required array in get_pdf.')
    try:
        for i in range(n_point): # prange(n_point, nogil=True, schedule='static'):
            j[i] = find_interval(&knots[0], n_interval + 1, xs[i]) - 1
            if j[i] >= 0 and j[i] <= n_interval - 1:
                if (types[j[i]] == NORMAL_CUBIC or types[j[i]] == LEFT_END_CUBIC or
                    types[j[i]] == RIGHT_END_CUBIC or types[j[i]] == LINEAR):
                    h[i] = knots[j[i] + 1] - knots[j[i]]
                    t[i] = (xs[i] - knots[j[i]]) / h[i]
                    ys[i] = (
                        (2. * t[i] * t[i] * t[i] - 3. * t[i] * t[i] + 1.) * quantiles[j[i]] +
                        (t[i] * t[i] * t[i] - 2. * t[i] * t[i] + t[i]) * h[i] * dydxs[j[i]] +
                        (-2. * t[i] * t[i] * t[i] + 3. * t[i] * t[i]) * quantiles[j[i] + 1] +
                        (t[i] * t[i] * t[i] - t[i] * t[i]) * h[i] * dydxs[j[i] + 1]
                    )
                elif types[j[i]] == DOUBLE_EXP:
                    ys[i] = 0.5 * (quantiles[j[i]] + quantiles[j[i] + 1])
                    ys[i] += _int_p_dx(expas[j[i], 0], xs[i] - knots[j[i]], dydxs[j[i]],
                                       dpdxs[j[i]], 0.)
                    ys[i] -= _int_p_dx(expas[j[i], 1], knots[j[i] + 1] - xs[i], dydxs[j[i] + 1],
                                       -dpdxs[j[i] + 1], 0.)
                elif types[j[i]] == LEFT_END_EXP:
                    ys[i] = quantiles[j[i] + 1]
                    ys[i] -= _int_p_dx(expas[j[i], 1], knots[j[i] + 1] - xs[i], dydxs[j[i] + 1],
                                       -dpdxs[j[i] + 1], 0.)
                elif types[j[i]] == RIGHT_END_EXP:
                    ys[i] = quantiles[j[i]]
                    ys[i] += _int_p_dx(expas[j[i], 0], xs[i] - knots[j[i]], dydxs[j[i]],
                                       dpdxs[j[i]], 0.)
                else:
                    ys[i] = nan
                if ys[i] < 0.:
                    ys[i] = 0.
                elif ys[i] > 1.:
                    ys[i] = 1.
            elif j[i] == -1:
                ys[i] = 0.
            elif j[i] == n_interval:
                ys[i] = 1.
            else:
                ys[i] = nan

    finally:
        free(j)
        free(h)
        free(t)


"""@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_ppf(double[::1] xs, const double[::1] ys, const double[::1] knots,
            const double[::1] quantiles, const double[::1] dydxs, const double[::1] dpdxs,
            const double[:, ::1] expas, const int[::1] types, int n_point, int n_interval):
    cdef size_t i
    cdef int *j = <int *> malloc(n_point * sizeof(int))
    cdef double *h = <double *> malloc(n_point * sizeof(double))
    cdef double *t = <double *> malloc(n_point * sizeof(double))
    if not j or not h or not t:
        raise MemoryError('cannot malloc required array in get_pdf.')
    try:
        for i in range(n_point): # prange(n_point, nogil=True, schedule='static'):
            if ys[i] == 0.:
                xs[i] = knots[0]
            elif ys[i] == 1:
                xs[i] = knots[n_interval]
            else:
                j[i] = find_interval(&knots[0], n_interval + 1, xs[i]) - 1
                if j[i] >= 0 and j[i] <= n_interval - 1:
                    if (types[j[i]] == NORMAL_CUBIC or types[j[i]] == LEFT_END_CUBIC or
                        types[j[i]] == RIGHT_END_CUBIC or types[j[i]] == LINEAR):
                        h[i] = knots[j[i] + 1] - knots[j[i]]
                        t[i] = (xs[i] - knots[j[i]]) / h[i]
                        ys[i] = (
                            (2. * t[i] * t[i] * t[i] - 3. * t[i] * t[i] + 1.) * quantiles[j[i]] +
                            (t[i] * t[i] * t[i] - 2. * t[i] * t[i] + t[i]) * h[i] * dydxs[j[i]] +
                            (-2. * t[i] * t[i] * t[i] + 3. * t[i] * t[i]) * quantiles[j[i] + 1] +
                            (t[i] * t[i] * t[i] - t[i] * t[i]) * h[i] * dydxs[j[i] + 1]
                        )
                    elif types[j[i]] == DOUBLE_EXP:
                        ys[i] = 0.5 * (quantiles[j[i]] + quantiles[j[i] + 1])
                        ys[i] += _int_p_dx(expas[j[i], 0], xs[i] - knots[j[i]], dydxs[j[i]],
                                           dpdxs[j[i]], 0.)
                        ys[i] -= _int_p_dx(expas[j[i], 1], knots[j[i] + 1] - xs[i], dydxs[j[i] + 1],
                                           -dpdxs[j[i] + 1], 0.)
                    elif types[j[i]] == LEFT_END_EXP:
                        ys[i] = quantiles[j[i] + 1]
                        ys[i] -= _int_p_dx(expas[j[i], 1], knots[j[i] + 1] - xs[i], dydxs[j[i] + 1],
                                           -dpdxs[j[i] + 1], 0.)
                    elif types[j[i]] == RIGHT_END_EXP:
                        ys[i] = quantiles[j[i]]
                        ys[i] += _int_p_dx(expas[j[i], 0], xs[i] - knots[j[i]], dydxs[j[i]],
                                           dpdxs[j[i]], 0.)
                    else:
                        ys[i] = nan
                else:
                    xs[i] = nan

    finally:
        free(j)
        free(h)
        free(t)"""
