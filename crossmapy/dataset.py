# -*- coding: utf-8 -*-
import warnings

from numba import njit
import numpy as np


@njit
def simu_logistic_2v(b_xy, b_yx, a_x=3.7, a_y=3.7, n_iter=300, x0=.2, y0=.4, noise=.0, seed=None):
    """
    perform the simulation of 2-variable logistic system.

    Parameters
    ----------
    b_xy: float
        coupling parameter beta_xy
    b_yx: float
        coupling parameter beta_yx
    a_x: float
        parameter alpha_x
    a_y: float
        parameter alpha_y
    n_iter: int
        maximal number of iteration
    x0: float
        initial x
    y0: float
        initial y
    noise: float
        noise intensity
    seed:
        random number seed

    Returns
    -------
    XY: 2d array
        simulated output array
    """
    XY = np.zeros((n_iter + 1, 2))
    XY[0] = [x0, y0]

    if seed is not None:
        np.random.seed(seed)

    x_last, y_last = x0, y0
    for i in range(1, n_iter + 1):
        x_cur = x_last * (a_x - a_x * x_last - b_xy * y_last) + np.random.normal(0, noise)
        y_cur = y_last * (a_y - a_y * y_last - b_yx * x_last) + np.random.normal(0, noise)
        XY[i] = [x_cur, y_cur]
        x_last, y_last = x_cur, y_cur

    return XY


def mul_logistic_2v(b_xy, b_yx, n_trail=100, n_iter=300, seed=None, n_max_fail=10, **kwargs):
    """
    perform multiple simulations of 2-variable logistic system.

    Parameters
    ----------
    b_xy: float
        coupling parameter beta_xy
    b_yx: float
        coupling parameter beta_yx
    n_trail: int
        maximal number of trails
    n_iter: int
        maximal number of iterations for each trail
    seed: int
        random number seed
    kwargs:
        other keyword arguments are passed through to simu_logistic_2v()

    Returns
    -------
    mul_XY: 3d array
        simulated output array
    """
    mul_XY = np.zeros((n_trail, n_iter + 1, 2))
    count_fail = 0
    for n in range(n_trail):
        while True:
            XY = simu_logistic_2v(b_xy, b_yx, n_iter=n_iter, seed=seed, **kwargs)
            if (~np.isnan(XY)).all() and (~np.isinf(XY)).all():
                mul_XY[n] = XY
                break
            else:
                warnings.warn('Warning! Invalid data, regenerating ...')
                count_fail += 1
                if count_fail > n_max_fail:
                    raise RuntimeError('Bad parameters, adjust them and try again.')
                if seed is not None:
                    seed += 1
                continue

        if seed is not None:
            seed += 1

    return mul_XY



@njit
def simu_logistic_3v(b_xy,
                     b_yx,
                     b_yz,
                     b_zy,
                     b_xz,
                     b_zx,
                     a_x=3.6,
                     a_y=3.72,
                     a_z=3.68,
                     n_iter=300,
                     x0=.2,
                     y0=.4,
                     z0=.2,
                     noise=.0,
                     seed=None):
    """
    perform the simulation of 3-variable logistic system.

    Parameters
    ----------
    b_xy: float
        coupling parameter beta_xy
    b_yx: float
        coupling parameter beta_yx
    b_yz: float
        coupling parameter beta_yz
    b_zy: float
        coupling parameter beta_zy
    b_xz: float
        coupling parameter beta_xz
    b_zx: float
        coupling parameter beta_zx
    a_x: float
        parameter alpha_x
    a_y: float
        parameter alpha_y
    a_z: float
        parameter alpha_z
    n_iter: int
        maximal number of iteration
    x0: float
        initial x
    y0: float
        initial y
    z0: float
        initial z
    noise: float
        noise intensity
    seed: int
        random number seed

    Returns
    -------
    XYZ: 3d array
        simulated output array
    """
    XYZ = np.zeros((n_iter + 1, 3))
    XYZ[0] = [x0, y0, z0]

    if seed is not None:
        np.random.seed(seed)

    x_last, y_last, z_last = x0, y0, z0
    for i in range(1, n_iter + 1):
        x_cur = x_last * (a_x - a_x * x_last - b_xy * y_last - b_xz * z_last) + np.random.normal(0, noise)
        y_cur = y_last * (a_y - a_y * y_last - b_yx * x_last - b_yz * z_last) + np.random.normal(0, noise)
        z_cur = z_last * (a_z - a_z * z_last - b_zx * x_last - b_zy * y_last) + np.random.normal(0, noise)
        XYZ[i] = [x_cur, y_cur, z_cur]
        x_last, y_last, z_last = x_cur, y_cur, z_cur

    return XYZ


def mul_logistic_3v(b_xy,
                    b_yx,
                    b_yz,
                    b_zy,
                    b_xz,
                    b_zx,
                    n_trail=100,
                    n_iter=300,
                    seed=None,
                    n_max_fail=10,
                    **kwargs):
    """
    perform multiple simulations of 3-variable logitcal system.

    Parameters
    ----------
    b_xy: float
        coupling parameter beta_xy
    b_yx: float
        coupling parameter beta_yx
    b_yz: float
        coupling parameter beta_yz
    b_zy: float
        coupling parameter beta_zy
    b_xz: float
        coupling parameter beta_xz
    b_zx: float
        coupling parameter beta_zx
    n_trail: int
        maximal number of trails
    n_iter: int
        maximal number of iterations for each trail
    seed: int
        random number seed
    kwargs:
        other keyword arguments are passed through to simu_logistic_3v()

    Returns
    -------
    mul_XYZ: 3d array
        simulated output array
    """
    mul_XYZ = np.zeros((n_trail, n_iter + 1, 3))
    count_fail = 0
    for n in range(n_trail):
        while True:
            XYZ = simu_logistic_3v(b_xy, b_yx, b_yz, b_zy, b_xz, b_zx, n_iter=n_iter, seed=seed, **kwargs)

            if (~np.isnan(XYZ)).all() and (~np.isinf(XYZ)).all():
                mul_XYZ[n] = XYZ
                break
            else:
                warnings.warn('Warning! Invalid data, regenerating ...')
                count_fail += 1
                if count_fail > n_max_fail:
                    raise RuntimeError('Bad parameters, adjust them and try again.')
                if seed is not None:
                    seed += 1
                continue

        if seed is not None:
            seed += 1

    return mul_XYZ


@njit
def simu_henon_9v(xs0=None, xs1=None, alpha=0.2, n_iter=1000, noise=.0):
    """
    perform the simulation of 9-variable Henon system.

    Parameters
    ----------
    xs0: vector or 1d array
        nine initial x0
    xs1: vector or 1d array
        nine initial x1
    alpha: float
        parameter of henon system
    n_iter: int
        maximal number of iteration
    noise: float
        noise intensity

    Returns
    -------
    sim_array: 2d array
        simulated output array
    """
    n_known = 2
    n_feature = 9

    if xs0 is None:
        xs0 = np.arange(0.1, 1., 0.1)
    x1_0, x2_0, x3_0, x4_0, x5_0, x6_0, x7_0, x8_0, x9_0 = xs0

    if xs1 is None:
        xs1 = np.arange(0.9, 0., -0.1)
    x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1 = xs1

    sim_array = np.zeros((n_iter + n_known, n_feature))
    sim_array[0] = xs0
    sim_array[1] = xs1
    for i in range(n_known, n_iter + n_known):
        x1 = 1.4 - x1_1 ** 2 + 0.3 * x1_0 + np.random.normal(0, noise)
        x9 = 1.4 - x9_1 ** 2 + 0.3 * x9_0 + np.random.normal(0, noise)
        x2 = 1.4 - (0.5 * alpha * (x1_1 + x3_1) + (1.0 - alpha) * x2_1) ** 2 + \
             0.3 * x2_0 + np.random.normal(0, noise)
        x3 = 1.4 - (0.5 * alpha * (x2_1 + x4_1) + (1.0 - alpha) * x3_1) ** 2 + \
             0.3 * x3_0 + np.random.normal(0, noise)
        x4 = 1.4 - (0.5 * alpha * (x3_1 + x5_1) + (1.0 - alpha) * x4_1) ** 2 + \
             0.3 * x4_0 + np.random.normal(0, noise)
        x5 = 1.4 - (0.5 * alpha * (x4_1 + x6_1) + (1.0 - alpha) * x5_1) ** 2 + \
             0.3 * x5_0 + np.random.normal(0, noise)
        x6 = 1.4 - (0.5 * alpha * (x5_1 + x7_1) + (1.0 - alpha) * x6_1) ** 2 + \
             0.3 * x6_0 + np.random.normal(0, noise)
        x7 = 1.4 - (0.5 * alpha * (x6_1 + x8_1) + (1.0 - alpha) * x7_1) ** 2 + \
             0.3 * x7_0 + np.random.normal(0, noise)
        x8 = 1.4 - (0.5 * alpha * (x7_1 + x9_1) + (1.0 - alpha) * x8_1) ** 2 + \
             0.3 * x8_0 + np.random.normal(0, noise)
        sim_array[i] = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
        x1_0, x2_0, x3_0, x4_0, x5_0, x6_0, x7_0, x8_0, x9_0 = \
            x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1
        x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1 = \
            x1, x2, x3, x4, x5, x6, x7, x8, x9
        
    return sim_array


def mul_henon_9v(n_trail=10, n_iter=600, **kwargs):
    """
    perform multiple simulations of 9-variable Henon system.

    Parameters
    ----------
    n_trail: int
        maximal number of trails
    n_iter: int
        maximal number of iterations for each trail
    kwargs:
        other keyword arguments are passed through to simu_henon_9v()

    Returns
    -------
    mul_array: 3d array
        simulated output array
    """
    mul_array = np.zeros((n_trail, n_iter + 2, 9))
    for n in range(n_trail):
        while True:
            sim_array = simu_henon_9v(**kwargs)
            if np.isnan(sim_array).any() or np.isinf(sim_array).any():
                warnings.warn('Warning! Invalid data, regenerating ...')
                continue
            else:
                mul_array[n] = sim_array
                break

    return mul_array
