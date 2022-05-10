# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.spatial.distance import cdist
from scipy.special import psi

from . import utils


def cal_mi_from_knn(x, y, n_neighbors=3, norm=True, norm_mode='min'):
    """Compute mutual information between two continuous variables.
    Parameters
    ----------
    x, y : ndarray, shape (n_samples, n_features)
        Samples of two continuous random variables, must have an identical
        shape.
    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information. If it turned out to be negative it is
        replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    n_samples = len(x)
    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric='chebyshev')
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric='chebyshev')
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    psi_sp = psi(n_samples)
    psi_nb = psi(n_neighbors)
    psi_x = np.mean(psi(nx + 1))
    psi_y = np.mean(psi(ny + 1))

    # MI(X, Y) = psi(n_neighbors) + psi(n_samples) - < psi(nX + 1) + psi(nY + 1) >
    mi = max(0, psi_nb + psi_sp - psi_x - psi_y)
    if norm:
        return _norm_mi(mi, psi_sp, psi_x, psi_y, norm_mode)
    else:
        return mi


def cal_cmi_from_knn(x, y, z, n_neighbors=3, norm=True, norm_mode='min'):
    """Compute conditional mutual information between two continuous variables on
    the third variable.

    Parameters
    ----------
    x, y, z : ndarray, shape (n_samples, n_features)
        Samples of two continuous random variables, must have an identical
        shape.
    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information. If it turned out to be negative
        it is replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    yz = np.hstack((y, z))
    xz = np.hstack((x, z))
    xyz = np.hstack((x, y, z))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

    nn.fit(xyz)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(z, metric='chebyshev')
    nz = kd.query_radius(z, radius, count_only=True, return_distance=False)
    nz = np.array(nz) - 1.0

    kd = KDTree(xz, metric='chebyshev')
    nxz = kd.query_radius(xz, radius, count_only=True, return_distance=False)
    nxz = np.array(nxz) - 1.0

    kd = KDTree(yz, metric='chebyshev')
    nyz = kd.query_radius(yz, radius, count_only=True, return_distance=False)
    nyz = np.array(nyz) - 1.0

    psi_nb = psi(n_neighbors)
    psi_xz = np.mean(psi(nxz + 1))
    psi_yz = np.mean(psi(nyz + 1))
    psi_z = np.mean(psi(nz + 1))
    # CMI(X, Y | Z) = psi(n_neighbors) - < psi(nXZ + 1) + psi(nYZ + 1) - psi(nZ + 1) >
    # cmi = (psi(n_neighbors) + np.mean(psi(nz + 1)) -
    #        np.mean(psi(nxz + 1)) - np.mean(psi(nyz + 1)))
    cmi = max(0, psi_nb + psi_z - psi_xz - psi_yz)

    if norm:
        kd = KDTree(x, metric='chebyshev')
        nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
        nx = np.array(nx) - 1.0

        kd = KDTree(y, metric='chebyshev')
        ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
        ny = np.array(ny) - 1.0

        psi_sp = psi(len(x))
        psi_x = np.mean(psi(nx + 1))
        psi_y = np.mean(psi(ny + 1))
        return _norm_mi(cmi, psi_sp, psi_x, psi_y, norm_mode)
    else:
        return cmi


def _norm_mi(mi, psi_sp, psi_x, psi_y, norm_mode='min'):
    Hx = psi_sp - psi_x
    Hy = psi_sp - psi_y
    if norm_mode == 'min':
        entropy = min(Hx, Hy)
    elif norm_mode == 'target':
        entropy = Hy
    elif norm_mode == 'source':
        entropy = Hx
    else:
        return mi

    if entropy <= 0:
        return 0
    else:
        return mi / entropy


def cal_mi_from_knn_old(X, Y, n_neighbor=3, n_excluded=0):
    """
    estimate the mutual information based on the following paper.
    "Kraskov et., Estimating Mutual Information, 2004"

    Parameters
    ----------
    X: 2d array
        manifold X
    Y: 2d array
        manifold Y
    n_neighbor: int
        the number of neighbors
    n_excluded: int
        the number of excluded neighbors

    Returns
    -------
    mi_estimated: float
        estimated mutual information value
    """

    assert X.ndim == 2 and Y.ndim == 2, 'X and Y must be 2d array.'
    assert X.shape[0] == Y.shape[0], 'the numbers of row of X and Y must be same.'

    n_row = X.shape[0]

    XY = np.hstack((X, Y))
    n_X = np.zeros(n_row)
    n_Y = np.zeros(n_row)
    for i in range(n_row):
        # Theiler correction. Points around i are excluded.
        excluded_idx = utils.exclude_range(n_row, i, n_excluded)

        X_tmp = X[excluded_idx, :]
        Y_tmp = Y[excluded_idx, :]
        XY_tmp = XY[excluded_idx, :]
        neigh = NearestNeighbors(n_neighbors=n_neighbor, metric='chebyshev')
        neigh.fit(XY_tmp)

        target_XY = XY[i:i + 1, :]
        dist_k, _ = neigh.kneighbors(target_XY)
        half_epsilon = dist_k[0, -1]
        n_X[i] = np.sum(cdist(X_tmp, X[i:i + 1], metric='chebyshev') < half_epsilon)
        n_Y[i] = np.sum(cdist(Y_tmp, Y[i:i + 1], metric='chebyshev') < half_epsilon)

    idx = np.where((n_X != 0) & (n_Y != 0))
    mi_estimated = psi(n_neighbor) - np.mean(psi(n_X[idx] + 1)) - \
                   np.mean(psi(n_Y[idx] + 1)) + psi(n_row)

    return max(0, mi_estimated)


def cal_cmi_from_knn_old(X, Y, Z, n_neighbor=3, n_excluded=0, metric='chebyshev'):
    """
    This function aims to estimate the MI of two variables condition on the third
    one, namely, Mx and My condition on Mz.

    Parameters
    ----------
    X: 2d array
        manifold X
    Y: 2d array
        manifold Y
    Z: 2d array
        manifold Z
    n_neighbor: int
        the number of neighbors
    n_excluded: int
        the number of excluded neighbors
    metric: str
        distance metric

    Returns
    -------
    cmi_estimated: float
        estimated conditional mutual information value
    """
    assert X.ndim == 2 and Y.ndim == 2 and Z.ndim == 2, 'X, Y and Z must be 2d array.'
    assert X.shape[0] == Y.shape[0] and X.shape[0] == Z.shape[0], \
        'the numbers of row of X and Y must be same.'

    n_row = X.shape[0]

    # n_XY = np.zeros(n_row)
    n_YZ = np.zeros(n_row)
    n_XZ = np.zeros(n_row)
    n_Z = np.zeros(n_row)

    # XY = np.hstack((X, Y))
    YZ = np.hstack((Y, Z))
    XZ = np.hstack((X, Z))
    XYZ = np.hstack((X, Y, Z))
    for i in range(n_row):
        excluded_idx = utils.exclude_range(n_row, i, n_excluded)
        Z_tmp = Z[excluded_idx, :]
        # XY_tmp = XY[excluded_idx, :]
        YZ_tmp = YZ[excluded_idx, :]
        XZ_tmp = XZ[excluded_idx, :]
        XYZ_tmp = XYZ[excluded_idx, :]
        neigh = NearestNeighbors(n_neighbors=n_neighbor, metric=metric)
        neigh.fit(XYZ_tmp)

        target_XYZ = XYZ[i:i + 1, :]
        dist_k, _ = neigh.kneighbors(target_XYZ)
        half_epsilon_XYZ = dist_k[0, -1]
        
        # n_XY[i] = np.sum(cdist(XY_tmp, XY[i:i + 1], metric=metric) < half_epsilon_XYZ)
        n_YZ[i] = np.sum(cdist(YZ_tmp, YZ[i:i + 1], metric=metric) < half_epsilon_XYZ)
        n_XZ[i] = np.sum(cdist(XZ_tmp, XZ[i:i + 1], metric=metric) < half_epsilon_XYZ)
        n_Z[i] = np.sum(cdist(Z_tmp, Z[i:i + 1], metric=metric) < half_epsilon_XYZ)

    idx = np.where((n_YZ != 0) & (n_XZ != 0) & (n_Z != 0))

    # CMI(X, Y | Z) = H(X | Z) - H(X | Y, Z) = psi(k) - < psi(nXZ + 1) + psi(nYZ + 1) - psi(nZ + 1) >
    cmi_estimated = psi(n_neighbor) - np.mean(psi(n_YZ[idx] + 1)) - \
                    np.mean(psi(n_XZ[idx] + 1)) + np.mean(psi(n_Z[idx] + 1))
    return max(0, cmi_estimated)


