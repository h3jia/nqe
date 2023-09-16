import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.parallel cimport prange
from scipy.special.cython_special cimport erf, erfi, erfcx, dawsn
# from libc.stdlib cimport malloc, free
from libc.math cimport exp, sqrt, pi, fabs, fmax, fmin, lround, isnan
from scipy.optimize.cython_optimize cimport brentq
cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"
    double inf "NPY_INFINITY"


__all__ = ['_get_configs', '_pdf_1_n', '_pdf_n_n', '_cdf_1_n', '_cdf_n_n', '_ppf_1_n', '_ppf_n_n',
           '_cdf_1_n_local', '_cdf_n_n_local']


ctypedef struct int_p_dx_params_expa:
    double h
    double p0
    double dpdx0
    double mass

ctypedef struct int_p_dx_params_dpdx:
    double h
    double p0
    double mass

ctypedef struct cdf_params:
    double y
    double h
    double y0
    double y1
    double dydx0
    double dydx1
    double dpdx0
    double dpdx1
    double expa0
    double expa1

cdef double XTOL = 1e-10, RTOL = 1e-10
cdef int MITR = 300
cdef int UNDEFINED = 0, NORMAL_CUBIC = 1, LEFT_END_CUBIC = 2, RIGHT_END_CUBIC = 3, LINEAR = 4
cdef int DOUBLE_EXP = 5, LEFT_END_EXP = 6, RIGHT_END_EXP = 7 #, MERGE_EXP = 8
cdef int I_KNOTS = 0, I_QUANTILES = 1, I_SPLIT_FACTORS = 2, I_SPLIT_FACTORS_2 = 3, I_TYPES = 4
cdef int I_DYDXS = 5, I_DPDXS = 6, I_EXPAS_0 = 7, I_EXPAS_1 = 8, N_CONFIG_INDICES = 9


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline int iround(double x) nogil:
    return int(lround(x))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _find_interval(const double* x, int m, double xval, int prev_interval=-1) nogil:
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


# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.cdivision(True)
# cdef double _int_p_dx(double a, double h, double p0, double dpdx0, double mass) nogil:
#     cdef double absa = fabs(a)
#     if fabs(dpdx0 / p0 * h) < 1e-8:
#         if a > 0:
#             return sqrt(pi) * p0 / 2. / sqrt(absa) * erfi(sqrt(absa) * h) - mass
#         elif a == 0:
#             return p0 * h - mass
#         elif a < 0:
#             return sqrt(pi) * p0 / 2. / sqrt(absa) * erf(sqrt(absa) * h) - mass
#         else:
#             return nan
#     else:
#         if absa < fabs(dpdx0 * dpdx0 / 4. / p0 / p0) / 600.: # avoid computing exp(>600)
#             return p0 * p0 / dpdx0 * (exp(dpdx0 / p0 * h) - 1) - mass
#         elif a > 0:
#             return (
#                 sqrt(pi) * p0 / 2. / sqrt(absa) * exp(-dpdx0 * dpdx0 / 4. / absa / p0 / p0) * (
#                     erfi((2 * absa * p0 * h + dpdx0) / 2. / sqrt(absa) / p0) -
#                     erfi(dpdx0 / 2. / sqrt(absa) / p0)
#                 ) - mass
#             )
#         elif a < 0:
#             return (
#                 sqrt(pi) * p0 / 2. / sqrt(absa) * exp(dpdx0 * dpdx0 / 4. / absa / p0 / p0) * (
#                     erf((2 * absa * p0 * h - dpdx0) / 2. / sqrt(absa) / p0) +
#                     erf(dpdx0 / 2. / sqrt(absa) / p0)
#                 ) - mass
#             )
#         else:
#             return nan


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _int_p_dx_args_expa(double a, void *args) noexcept nogil:
    cdef int_p_dx_params_expa *myargs = <int_p_dx_params_expa *> args
    return _int_p_dx(a, myargs.h, myargs.p0, myargs.dpdx0, myargs.mass)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _int_p_dx_args_dpdx(double dpdx, void *args) noexcept nogil:
    cdef int_p_dx_params_dpdx *myargs = <int_p_dx_params_dpdx *> args
    return _int_p_dx(0., myargs.h, myargs.p0, dpdx, myargs.mass)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_dydx_1(double h0, double h1, double m0, double m1, double p_tail_limit=0.75) nogil:
    cdef double d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    # if d * m0 <= 0.:
    #     return 0.5 * m0
    # if d * m0 <= 0. or fabs(d) < 0.5 * fabs(m0):
    #     return 0.5 * m0
    # elif m0 * m1 <= 0. and fabs(d) > 3. * fabs(m0):
    #     return 3. * m0
    # return d
    if m0 > 0.:
        return fmin(fmax(d, p_tail_limit * m0), 3. * m0)
    elif m0 < 0.:
        return fmin(fmax(d, 3. * m0), p_tail_limit * m0)
    else:
        return 0.


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
cdef void _get_split_factors(double* split_factors, const double* knots, const double* quantiles,
                             int n_interval, int central_width=1, double p_tail_limit=0.75) nogil:
    cdef size_t i
    for i in range(n_interval): # prange(n_interval, nogil=True, schedule='static'):
        split_factors[i] = _get_split_factor(&knots[i], &quantiles[i], central_width, p_tail_limit)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_split_factor(const double* knots, const double* quantiles,
                              int central_width=1, double p_tail_limit=0.75) nogil:
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
    cdef double dydx1 = _get_dydx_1(h1, h0, m1, m0, p_tail_limit)
    cdef double dydx2 = _get_dydx_1(h3, h4, m3, m4, p_tail_limit)
    cdef double dydx3 = _get_dydx_2(h3, h4, m3, m4)
    cdef double dpdx1, dpdx2, a1, a2, p1a, p2a
    if dydx1 > 0. and dydx2 > 0.: # should be always true for cdf interp here
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
cdef void _get_exp_types(double* types, const double* split_factors, const double* split_factors_2,
                         int n_2, double split_threshold) nogil:
    cdef size_t i
    for i in range(2, n_2 - 3):
        if split_factors[i] < fmin(split_threshold, fmin(split_factors[i - 1],
                                                         split_factors[i + 1])):
            types[i] = DOUBLE_EXP
    for i in range(2, n_2 - 4):
        if split_factors_2[i] < fmin(split_threshold,
                                     fmin(fmin(split_factors_2[i - 2], split_factors_2[i - 1]),
                                          fmin(split_factors_2[i + 1], split_factors_2[i + 2]))):
            types[i] = DOUBLE_EXP
            types[i + 1] = DOUBLE_EXP


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _merge_exp_types(double[:, ::1] configs, int n_2) nogil:
    cdef size_t i, j
    for i in range(2, n_2 - 4):
        if iround(configs[I_TYPES, i]) == DOUBLE_EXP:
            j = 0
            while i + j + 1 <= n_2 - 4 and iround(configs[I_TYPES, i + j + 1]) == DOUBLE_EXP:
                j += 1
            if j > 0:
                _remove_merged(configs, i, j, n_2)
    return _get_n_m(configs, n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _get_n_m(const double[:, ::1] configs, int n_2) nogil:
    cdef size_t i
    for i in range(n_2 - 1, -1, -1):
        if not isnan(configs[I_KNOTS, i]):
            return i + 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _remove_merged(double[:, ::1] configs, int i, int j, int n_2) nogil:
    cdef size_t k, l
    cdef size_t[3] k_all = [I_KNOTS, I_QUANTILES, I_TYPES]
    for k in k_all:
        for l in range(i + 1, n_2 - j):
            configs[k, l] = configs[k, l + j]
        for l in range(n_2 - j, n_2):
            configs[k, l] = nan


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _get_cubic_types(double* types, int n_m) nogil:
    cdef size_t i
    types[1] = LEFT_END_CUBIC if iround(types[2]) not in (DOUBLE_EXP, RIGHT_END_EXP) else LINEAR
    types[n_m - 3] = (RIGHT_END_CUBIC if iround(types[n_m - 4]) not in (DOUBLE_EXP, LEFT_END_EXP)
                      else LINEAR)
    for i in range(2, n_m - 3):
        if iround(types[i]) == UNDEFINED:
            if iround(types[i - 1]) != DOUBLE_EXP and iround(types[i + 1]) != DOUBLE_EXP:
                types[i] = NORMAL_CUBIC
            elif iround(types[i - 1]) == DOUBLE_EXP and iround(types[i + 1]) != DOUBLE_EXP:
                types[i] = LEFT_END_CUBIC
            elif iround(types[i - 1]) != DOUBLE_EXP and iround(types[i + 1]) == DOUBLE_EXP:
                types[i] = RIGHT_END_CUBIC
            else:
                types[i] = LINEAR


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _get_dydxs(double* dydxs, const double* knots, const double* quantiles,
                     const double* types, int n_interval, double p_tail_limit=0.75) nogil:
    cdef size_t i
    cdef double h0, h1
    # for i in prange(_max(i_start, 1), _min(i_end + 1, n_interval), nogil=True,
    #                 schedule='static'):
    for i in range(1, n_interval):
        if iround(types[i]) == LINEAR:
            dydxs[i] = (quantiles[i + 1] - quantiles[i]) / (knots[i + 1] - knots[i])
        elif iround(types[i - 1]) == LINEAR:
            dydxs[i] = (quantiles[i] - quantiles[i - 1]) / (knots[i] - knots[i - 1])
        elif iround(types[i]) == NORMAL_CUBIC or iround(types[i]) == RIGHT_END_CUBIC:
            h0 = knots[i] - knots[i - 1]
            h1 = knots[i + 1] - knots[i]
            dydxs[i] = _get_dydx_2(h0, h1, (quantiles[i] - quantiles[i - 1]) / h0,
                                   (quantiles[i + 1] - quantiles[i]) / h1)
        elif iround(types[i]) == LEFT_END_CUBIC:
            h0 = knots[i + 1] - knots[i]
            h1 = knots[i + 2] - knots[i + 1]
            dydxs[i] = _get_dydx_1(h0, h1, (quantiles[i + 1] - quantiles[i]) / h0,
                                   (quantiles[i + 2] - quantiles[i + 1]) / h1, p_tail_limit)
        elif iround(types[i]) == DOUBLE_EXP or iround(types[i]) == RIGHT_END_EXP:
            h0 = knots[i] - knots[i - 1]
            h1 = knots[i - 1] - knots[i - 2]
            dydxs[i] = _get_dydx_1(h0, h1, (quantiles[i] - quantiles[i - 1]) / h0,
                                   (quantiles[i - 1] - quantiles[i - 2]) / h1, p_tail_limit)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _get_exps(double* expas_0, double* expas_1, double* dpdxs, const double* knots,
                    const double* quantiles, const double* dydxs, const double* types,
                    int n_interval) nogil:
    cdef size_t i
    # for i in prange(i_start, i_end, nogil=True, schedule='static'):
    for i in range(n_interval):
        if iround(types[i]) == DOUBLE_EXP or iround(types[i]) == RIGHT_END_EXP:
            dpdxs[i] = _get_right_end_dpdx(knots[i] - knots[i - 1], quantiles[i - 1], quantiles[i],
                                           dydxs[i - 1], dydxs[i])
            expas_0[i] = _solve_single_expa(
                knots[i + 1] - knots[i], dydxs[i], dpdxs[i],
                (0.5 if iround(types[i]) == DOUBLE_EXP else 1.) * (quantiles[i + 1] - quantiles[i])
            )
            if dpdxs[i] * expas_0[i] < 0.:
                expas_0[i] = 0.
                dpdxs[i] = _solve_single_dpdx(
                    knots[i + 1] - knots[i], dydxs[i], dpdxs[i],
                    (0.5 if iround(types[i]) == DOUBLE_EXP else 1.) *
                    (quantiles[i + 1] - quantiles[i])
                )
        if iround(types[i]) == DOUBLE_EXP or iround(types[i]) == LEFT_END_EXP:
            dpdxs[i + 1] = _get_left_end_dpdx(knots[i + 2] - knots[i + 1], quantiles[i + 1],
                                              quantiles[i + 2], dydxs[i + 1], dydxs[i + 2])
            expas_1[i] = _solve_single_expa(
                knots[i + 1] - knots[i], dydxs[i + 1], -dpdxs[i + 1],
                (0.5 if iround(types[i]) == DOUBLE_EXP else 1.) * (quantiles[i + 1] - quantiles[i])
            )
            if dpdxs[i + 1] * expas_1[i] > 0.:
                expas_1[i] = 0.
                dpdxs[i + 1] = -_solve_single_dpdx(
                    knots[i + 1] - knots[i], dydxs[i + 1], -dpdxs[i + 1],
                    (0.5 if iround(types[i]) == DOUBLE_EXP else 1.) *
                    (quantiles[i + 1] - quantiles[i])
                )


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_pdf(double x, const double[:, ::1] configs, int n_2) nogil:
    cdef int n_m = _get_n_m(configs, n_2)
    cdef double h, t, y
    cdef int j = _find_interval(&configs[I_KNOTS, 0], n_m, x) - 1
    cdef int t_j = iround(configs[I_TYPES, j])
    if j >= 0 and j <= n_m - 2:
        if t_j in (NORMAL_CUBIC, LEFT_END_CUBIC, RIGHT_END_CUBIC, LINEAR):
            h = configs[I_KNOTS, j + 1] - configs[I_KNOTS, j]
            t = (x - configs[I_KNOTS, j]) / h
            y = (
                (6. * t * t - 6. * t) / h * configs[I_QUANTILES, j] +
                (3. * t * t - 4. * t + 1.) * configs[I_DYDXS, j] +
                (-6. * t * t + 6. * t) / h * configs[I_QUANTILES, j + 1] +
                (3. * t * t - 2. * t) * configs[I_DYDXS, j + 1]
            )
        elif t_j == DOUBLE_EXP:
            t = configs[I_KNOTS, j + 1] - x
            y = configs[I_DYDXS, j + 1] * exp(configs[I_EXPAS_1, j] * t * t -
                                              configs[I_DPDXS, j + 1] / configs[I_DYDXS, j + 1] * t)
            t = x - configs[I_KNOTS, j]
            y += configs[I_DYDXS, j] * exp(configs[I_EXPAS_0, j] * t * t +
                                           configs[I_DPDXS, j] / configs[I_DYDXS, j] * t)
        elif t_j == LEFT_END_EXP:
            t = configs[I_KNOTS, j + 1] - x
            y = configs[I_DYDXS, j + 1] * exp(configs[I_EXPAS_1, j] * t * t -
                                              configs[I_DPDXS, j + 1] / configs[I_DYDXS, j + 1] * t)
        elif t_j == RIGHT_END_EXP:
            t = x - configs[I_KNOTS, j]
            y = configs[I_DYDXS, j] * exp(configs[I_EXPAS_0, j] * t * t +
                                          configs[I_DPDXS, j] / configs[I_DYDXS, j] * t)
        else:
            y = nan
    elif j == -1 or j == n_m - 1:
        y = 0.
    else:
        y = nan
    return y


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _pdf_1_n(const double[:, :, ::1] configs, const double[::1] xs, double[::1] ys, int n_point,
             int n_2):
    cdef size_t i
    for i in prange(n_point, nogil=True, schedule='static'):
        ys[i] = _get_pdf(xs[i], configs[0], n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _pdf_n_n(const double[:, :, ::1] configs, const double[::1] xs, double[::1] ys, int n_point,
             int n_2):
    cdef size_t i
    for i in prange(n_point, nogil=True, schedule='static'):
        ys[i] = _get_pdf(xs[i], configs[i], n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _cdf_cubic(double t, double h, double y0, double y1, double dydx0,
                              double dydx1) nogil:
    t /= h
    return (
        (2. * t * t * t - 3. * t * t + 1.) * y0 +
        (t * t * t - 2. * t * t + t) * h * dydx0 +
        (-2. * t * t * t + 3. * t * t) * y1 +
        (t * t * t - t * t) * h * dydx1
    )


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _cdf_cubic_args(double t, void *args) noexcept nogil:
    cdef cdf_params *myargs = <cdf_params *> args
    return _cdf_cubic(t, myargs.h, myargs.y0, myargs.y1, myargs.dydx0, myargs.dydx1) - myargs.y


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _cdf_double_exp(double t, double h, double y0, double y1, double dydx0,
                                   double dydx1, double dpdx0, double dpdx1, double expa0,
                                   double expa1) nogil:
    return (
        0.5 * (y0 + y1) +
        _int_p_dx(expa0, t, dydx0, dpdx0, 0.) -
        _int_p_dx(expa1, h - t, dydx1, -dpdx1, 0.)
    )


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _cdf_double_exp_args(double t, void *args) noexcept nogil:
    cdef cdf_params *myargs = <cdf_params *> args
    return _cdf_double_exp(t, myargs.h, myargs.y0, myargs.y1, myargs.dydx0, myargs.dydx1,
                           myargs.dpdx0, myargs.dpdx1, myargs.expa0, myargs.expa1) - myargs.y


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _cdf_left_exp(double t, double y, double dydx, double dpdx, double expa) nogil:
    # note the different def of t
    return y - _int_p_dx(expa, t, dydx, -dpdx, 0.)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _cdf_left_exp_args(double t, void *args) noexcept nogil:
    cdef cdf_params *myargs = <cdf_params *> args
    return _cdf_left_exp(t, myargs.y1, myargs.dydx1, myargs.dpdx1, myargs.expa1) - myargs.y


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _cdf_right_exp(double t, double y, double dydx, double dpdx, double expa) nogil:
    return y + _int_p_dx(expa, t, dydx, dpdx, 0.)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _cdf_right_exp_args(double t, void *args) noexcept nogil:
    cdef cdf_params *myargs = <cdf_params *> args
    return _cdf_right_exp(t, myargs.y0, myargs.dydx0, myargs.dpdx0, myargs.expa0) - myargs.y


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_cdf(double x, const double[:, ::1] configs, int n_2) nogil:
    cdef int n_m = _get_n_m(configs, n_2)
    cdef double y
    cdef int j = _find_interval(&configs[I_KNOTS, 0], n_m, x) - 1
    cdef int t_j = iround(configs[I_TYPES, j])
    if j >= 0 and j <= n_m - 2:
        if t_j in (NORMAL_CUBIC, LEFT_END_CUBIC, RIGHT_END_CUBIC, LINEAR):
            y = _cdf_cubic(
                x - configs[I_KNOTS, j], configs[I_KNOTS, j + 1] - configs[I_KNOTS, j],
                configs[I_QUANTILES, j], configs[I_QUANTILES, j + 1], configs[I_DYDXS, j],
                configs[I_DYDXS, j + 1]
            )
        elif t_j == DOUBLE_EXP:
            y = _cdf_double_exp(
                x - configs[I_KNOTS, j], configs[I_KNOTS, j + 1] - configs[I_KNOTS, j],
                configs[I_QUANTILES, j], configs[I_QUANTILES, j + 1], configs[I_DYDXS, j],
                configs[I_DYDXS, j + 1], configs[I_DPDXS, j], configs[I_DPDXS, j + 1],
                configs[I_EXPAS_0, j], configs[I_EXPAS_1, j]
            )
        elif t_j == LEFT_END_EXP:
            y = _cdf_left_exp(configs[I_KNOTS, j + 1] - x, configs[I_QUANTILES, j + 1],
                              configs[I_DYDXS, j + 1], configs[I_DPDXS, j + 1],
                              configs[I_EXPAS_1, j])
        elif t_j == RIGHT_END_EXP:
            y = _cdf_right_exp(x - configs[I_KNOTS, j], configs[I_QUANTILES, j],
                               configs[I_DYDXS, j], configs[I_DPDXS, j], configs[I_EXPAS_0, j])
        else:
            y = nan
        if y < 0.:
            y = 0.
        elif y > 1.:
            y = 1.
    elif j == -1:
        y = 0.
    elif j == n_m - 1:
        y = 1.
    else:
        y = nan
    return y


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _cdf_1_n(const double[:, :, ::1] configs, const double[::1] xs, double[::1] ys, int n_point,
             int n_2):
    cdef size_t i
    for i in prange(n_point, nogil=True, schedule='static'):
        ys[i] = _get_cdf(xs[i], configs[0], n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _cdf_n_n(const double[:, :, ::1] configs, const double[::1] xs, double[::1] ys, int n_point,
             int n_2):
    cdef size_t i
    for i in prange(n_point, nogil=True, schedule='static'):
        ys[i] = _get_cdf(xs[i], configs[i], n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_cdf_local(double y, const double[:, ::1] configs, int n_2) nogil:
    cdef int n_m = _get_n_m(configs, n_2)
    cdef int j = _find_interval(&configs[I_QUANTILES, 0], n_m, y) - 1, dj
    cdef double y0 = nan, y1 = nan
    if y == 0.:
        return 0.
    elif y == 1.:
        return 1.
    else:
        for dj in range(0, j + 1):
            if (iround(configs[I_TYPES, j - dj]) == DOUBLE_EXP and
                y >= 0.5 * (configs[I_QUANTILES, j - dj] + configs[I_QUANTILES, j - dj + 1])):
                y0 = 0.5 * (configs[I_QUANTILES, j - dj] + configs[I_QUANTILES, j - dj + 1])
                break
            if iround(configs[I_TYPES, j - dj]) == LEFT_END_EXP:
                y0 = configs[I_QUANTILES, j - dj]
                break
        for dj in range(0, n_m - j - 1):
            if (iround(configs[I_TYPES, j + dj]) == DOUBLE_EXP and
                y < 0.5 * (configs[I_QUANTILES, j + dj] + configs[I_QUANTILES, j + dj + 1])):
                y1 = 0.5 * (configs[I_QUANTILES, j + dj] + configs[I_QUANTILES, j + dj + 1])
                break
            if iround(configs[I_TYPES, j + dj]) == RIGHT_END_EXP:
                y1 = configs[I_QUANTILES, j + dj + 1]
                break
        return (y - y0) / (y1 - y0)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _cdf_1_n_local(const double[:, :, ::1] configs, double[::1] ys, int n_point, int n_2):
    cdef size_t i
    for i in prange(n_point, nogil=True, schedule='static'):
        ys[i] = _get_cdf_local(ys[i], configs[0], n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _cdf_n_n_local(const double[:, :, ::1] configs, double[::1] ys, int n_point, int n_2):
    cdef size_t i
    for i in prange(n_point, nogil=True, schedule='static'):
        ys[i] = _get_cdf_local(ys[i], configs[i], n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _get_ppf(double y, const double[:, ::1] configs, int n_2) nogil:
    cdef int n_m = _get_n_m(configs, n_2)
    cdef double x
    cdef int j = _find_interval(&configs[I_QUANTILES, 0], n_m, y) - 1
    cdef int t_j = iround(configs[I_TYPES, j])
    cdef cdf_params p = {'y': y, 'h': configs[I_KNOTS, j + 1] - configs[I_KNOTS, j],
                         'y0': configs[I_QUANTILES, j], 'y1': configs[I_QUANTILES, j + 1],
                         'dydx0': configs[I_DYDXS, j], 'dydx1': configs[I_DYDXS, j + 1],
                         'dpdx0': configs[I_DPDXS, j], 'dpdx1': configs[I_DPDXS, j + 1],
                         'expa0': configs[I_EXPAS_0, j], 'expa1': configs[I_EXPAS_1, j]}
    if j >= 0 and j <= n_m - 2:
        if t_j in (NORMAL_CUBIC, LEFT_END_CUBIC, RIGHT_END_CUBIC, LINEAR):
            x = configs[I_KNOTS, j] + brentq(
                _cdf_cubic_args, 0., configs[I_KNOTS, j + 1] - configs[I_KNOTS, j], &p, XTOL, RTOL,
                MITR, NULL
            )
        elif t_j == DOUBLE_EXP:
            x = configs[I_KNOTS, j] + brentq(
                _cdf_double_exp_args, 0., configs[I_KNOTS, j + 1] - configs[I_KNOTS, j], &p, XTOL,
                RTOL, MITR, NULL
            )
        elif t_j == LEFT_END_EXP:
            x = configs[I_KNOTS, j + 1] - brentq(
                _cdf_left_exp_args, 0., configs[I_KNOTS, j + 1] - configs[I_KNOTS, j], &p, XTOL,
                RTOL, MITR, NULL
            )
        elif t_j == RIGHT_END_EXP:
            x = configs[I_KNOTS, j] + brentq(
                _cdf_right_exp_args, 0., configs[I_KNOTS, j + 1] - configs[I_KNOTS, j], &p, XTOL,
                RTOL, MITR, NULL
            )
        else:
            x = nan
    elif j == -1:
        x = configs[I_KNOTS, 0]
    elif j == n_m - 1:
        x = configs[I_KNOTS, n_m - 1]
    else:
        x = nan
    return x


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _ppf_1_n(const double[:, :, ::1] configs, double[::1] xs, const double[::1] ys, int n_point,
             int n_2):
    cdef size_t i
    for i in prange(n_point, nogil=True, schedule='static'):
        xs[i] = _get_ppf(ys[i], configs[0], n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _ppf_n_n(const double[:, :, ::1] configs, double[::1] xs, const double[::1] ys, int n_point,
             int n_2):
    cdef size_t i
    for i in prange(n_point, nogil=True, schedule='static'):
        xs[i] = _get_ppf(ys[i], configs[i], n_2)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _get_config(const double[::1] knots, const double[::1] quantiles, double[:, ::1] configs,
                      int n_2, double split_threshold=1e-2, double p_tail_limit=0.75) nogil:
    cdef size_t i
    cdef int n_m
    cdef size_t[4] i_all = [0, 1, n_2 - 3, n_2 - 2]
    cdef size_t[4] i_all_2 = [0, 1, n_2 - 4, n_2 - 3]
    configs[I_KNOTS] = knots
    configs[I_QUANTILES] = quantiles
    configs[I_QUANTILES, 0] = 0.
    configs[I_QUANTILES, n_2 - 1] = 1.
    for i in i_all:
        configs[I_SPLIT_FACTORS, i] = inf
    for i in i_all_2:
        configs[I_SPLIT_FACTORS_2, i] = inf
    for i in range(n_2 - 1):
        configs[I_TYPES, i] = UNDEFINED
    configs[I_TYPES, 0] = LEFT_END_EXP
    configs[I_TYPES, n_2 - 2] = RIGHT_END_EXP

    _get_split_factors(&configs[I_SPLIT_FACTORS, 2], &configs[I_KNOTS, 0], &configs[I_QUANTILES, 0],
                       n_2 - 4, 1, p_tail_limit)
    _get_split_factors(&configs[I_SPLIT_FACTORS_2, 2], &configs[I_KNOTS, 0],
                       &configs[I_QUANTILES, 0], n_2 - 5, 2, p_tail_limit)

    _get_exp_types(&configs[I_TYPES, 0], &configs[I_SPLIT_FACTORS, 0],
                   &configs[I_SPLIT_FACTORS_2, 0], n_2, split_threshold)
    n_m = _merge_exp_types(configs, n_2)
    _get_cubic_types(&configs[I_TYPES, 0], n_m)
    _get_dydxs(&configs[I_DYDXS, 0], &configs[I_KNOTS, 0], &configs[I_QUANTILES, 0],
               &configs[I_TYPES, 0], n_m - 1, p_tail_limit)
    _get_exps(&configs[I_EXPAS_0, 0], &configs[I_EXPAS_1, 0], &configs[I_DPDXS, 0],
              &configs[I_KNOTS, 0], &configs[I_QUANTILES, 0], &configs[I_DYDXS, 0],
              &configs[I_TYPES, 0], n_m - 1)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _get_configs(const double[:, ::1] knots, const double[:, ::1] quantiles,
                 double[:, :, ::1] configs, int n_0, int n_2, double split_threshold=1e-2,
                 double p_tail_limit=0.75):
    cdef size_t i
    for i in prange(n_0, nogil=True, schedule='static'):
        _get_config(knots[i], quantiles[i], configs[i], n_2, split_threshold, p_tail_limit)
