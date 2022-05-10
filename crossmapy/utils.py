# -*- coding: utf-8 -*-
from numba import njit
import numpy as np

from .embed import embed_vector


def _norm_distance(x, y, order=2):
    return np.linalg.norm(x - y, ord=order)


def _score_list(x, ys, order=2):
    ref_dis = _norm_distance(x, ys[0], order=order)
    score_list = [np.exp(-_norm_distance(x, y, order=order) / ref_dis) for y in ys]
    return score_list


def cal_distance_weight(x, ys, order=2):
    score_list = _score_list(x, ys, order=order)
    sum_score = np.sum(score_list)
    weight_list = [score / sum_score for score in score_list]
    return weight_list


def predict_with_weights(cm_points, weights):
    return np.sum([p * w for p, w in zip(cm_points, weights)], axis=0)


@njit
def weights_from_neighbors(x, x_NN):
    """
    calculate the weight of the distance between each neighbor (of x) and x.

    Parameters
    ----------
    x: 2d array
        manifold of x
    x_NN: 2d array
        the nearest neighbors of x

    Returns
    -------
    weights: 1d array
        weights of x_NN
    """
    n_NN = x_NN.shape[0]
    ref_dis = np.linalg.norm(x - x_NN[0])
    scores = np.zeros(n_NN)
    for i in range(n_NN):
        scores[i] = np.exp(-np.linalg.norm(x - x_NN[i]) / ref_dis)
    sum_score = np.sum(scores)
    weights = np.zeros(n_NN)
    for i in range(n_NN):
        weights[i] = scores[i] / sum_score
    return weights


def weights_from_distances(distances):
    n = len(distances)
    ref_dis = distances[0]
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = np.exp(-distances[i] / ref_dis)
    sum_score = np.sum(scores)
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = scores[i] / sum_score
    return weights


def exclude_range(end, element, n_excluded=0, start=0):
    """
    range excluded the interval with width n_exclude at element

    Parameters
    ----------
    end: int
        end of the range
    element: int
        element at the center of the interval
    n_excluded: int
        width of the interval
    start: int
        start of the range

    Returns
    -------
        excluded range (1d array)

    Example
    -------
    >>> exclude_range(10, 4, 2)
    array([0, 1, 7, 8, 9])
    """
    if n_excluded >= 0:
        return np.hstack((np.arange(start, element - n_excluded),
                          np.arange(element + n_excluded + 1, end)))
    else:
        return np.arange(start, end)


# @njit
def partial_correlation(x, y, z):
    """
    calculate the partial correlation coefficient of x, y and z.

    Parameters
    ----------
    x: vector or 1d array
        time series x
    y: vector or 1d array
        time series y
    z: vector or 1d array
        time series z

    Returns
    -------
    partial_cor: float
        partial correlation coefficient
    """
    pcc_mat = np.corrcoef([x, y, z])
    r_xy = pcc_mat[0, 1]
    r_xz = pcc_mat[0, 2]
    r_yz = pcc_mat[1, 2]
    partial_cor = (r_xy - r_xz * r_yz) / (((1 - r_xz ** 2) ** .5) * ((1 - r_yz ** 2) ** .5))
    return partial_cor


def random_masked_array(nrow, ncol):
    arr = np.random.random((nrow, ncol))
    arr[arr <= .5] = 0
    arr[arr > .5] = 1
    return arr


def resort_masked_array(arr):
    col_sums = np.sum(arr, axis=0)
    row_sums = np.sum(arr, axis=1)
    col_new_idx = np.argsort(col_sums)
    row_new_idx = np.argsort(row_sums)
    new_arr = arr[row_new_idx][:, col_new_idx]
    return new_arr


def compute_squared_distance_loop(X):
    m, n = X.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = np.linalg.norm(X[:, i] - X[:, j]) ** 2
            D[j, i] = D[i, j]
    return D


def compute_squared_distance_vec(X):
    m, n = X.shape
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    return H + H.T - 2 * G


def compute_distance_loop(X):
    m, n = X.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = np.linalg.norm(X[:, i] - X[:, j])
            D[j, i] = D[i, j]
    return D


def compute_distance_vec(X):
    return np.sqrt(compute_squared_distance_vec(X))


def sorted_distance_index(x, embed_dim, lag=1):
    Mx = embed_vector(x, embed_dim=embed_dim, lag=lag)
    dis_x = compute_squared_distance_vec(Mx.T)
    idx = np.argsort(dis_x)
    return idx


def idx_intersec_count_tot(src_idx, tgt_idx):
    n_row = src_idx.shape[0]
    counts = (len(set(src_idx[i]) & set(tgt_idx[i])) for i in range(n_row))
    return sum(counts)


def idx_intersec_count_ind(src_idx, mid_idx, tgt_idx):
    n_row = src_idx.shape[0]
    counts = (len(set(src_idx[i]) & set(mid_idx[i]) & set(tgt_idx[i])) for i in range(n_row))
    return sum(counts)


def idx_intersec_count_dir(src_idx, mid_idx, tgt_idx):
    return idx_intersec_count_tot(src_idx, tgt_idx) - \
           idx_intersec_count_ind(src_idx, mid_idx, tgt_idx)


def check_dis_mat(dis_mat, n_neighbor):
    non_zeros = np.count_nonzero(dis_mat, axis=1)
    if len(np.where(non_zeros < n_neighbor)[0]) > 0:
        raise RuntimeError("Error: too many 0 in distance matrix.")
    return non_zeros


def skip_diag_masking(A):
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)


def skip_diag_broadcasting(A):
    m = A.shape[0]
    idx = (np.arange(1, m + 1) + (m + 1) * np.arange(m - 1)[:, None]).reshape(m, -1)
    return A.ravel()[idx]


def skip_diag_strided(A):
    m, n = A.shape
    assert m == n, 'input must be a NxN matrix.'
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)


@njit
def skip_diag_tri(mat, k=0):
    return np.tril(mat, -1 - k)[:, :-1 - k] + np.triu(mat, 1 + k)[:, 1 + k:]


def two_array_to_edges_set(array1, array2):
    set1 = {tuple(x) for x in array1}
    set2 = {tuple(x) for x in array2}
    set1and2 = set1 & set2
    set1or2 = set1 | set2
    set1not2 = set1 - set2
    set2not1 = set2 - set1
    return set1and2, set1or2, set1not2, set2not1


def cal_fpr_tpr(scores, truths, thresholds=np.arange(0, 1.01, 0.01)):
    p_idx = np.where(truths == 1)
    n_idx = np.where(truths == 0)
    n_threshold = len(thresholds)
    tpr = np.zeros(n_threshold)
    fpr = np.zeros(n_threshold)
    for i, t in enumerate(thresholds[::-1]):
        n_tp = np.count_nonzero(scores[p_idx] >= t)
        n_fp = np.count_nonzero(scores[n_idx] >= t)
        n_tn = np.count_nonzero(scores[n_idx] < t)
        n_fn = np.count_nonzero(scores[p_idx] < t)
        tpr[i] = n_tp / (n_tp + n_fn)
        fpr[i] = n_fp / (n_fp + n_tn)
    return fpr, tpr


def find_optimal_threshold(fpr, tpr, thresholds=np.arange(0, 1.01, 0.01)):
    Youden_idx = np.argmax(tpr - fpr)
    optim_threshold = thresholds[Youden_idx]
    point = (fpr[Youden_idx], tpr[Youden_idx])
    return optim_threshold, point


def exclude_vec_mat(vec, n_excluded=0):
    return skip_diag_tri(np.tile(vec, (len(vec), 1)), n_excluded)


def series_to_embed_dismat(x, embed_dim, **kwargs):
    Mx = embed_vector(x, embed_dim, **kwargs)
    dis_mat = compute_distance_vec(Mx.T)
    return skip_diag_tri(dis_mat)


def embed_to_dismat(Mx, k=0):
    return skip_diag_tri(compute_distance_vec(Mx.T), k)


def dismat_to_idx(dis_mat):
    idx = np.argsort(dis_mat)
    return idx


def counts_zeros_of_dismat(dis_mat):
    non_zeros = np.count_nonzero(dis_mat, axis=1)
    zeros = dis_mat.shape[1] - non_zeros
    return zeros


def series_to_idx(x, embed_dim, n_neighbor, **kwargs):
    return dismat_to_idx(series_to_embed_dismat(x, embed_dim, **kwargs), n_neighbor)


def idx_to_mapping(idx, idy, zeros_y, n_neighbor):
    n_row = idx.shape[0]
    map_x2y = np.asarray([
        np.where(idx[t] == idy[t][zeros_y[t]: n_neighbor + zeros_y[t], None])[-1]
        for t in range(n_row)
    ])
    return map_x2y


def edges_to_mat(edges, n_nold=None):
    assert edges.ndim == 2 and edges.shape[1] >= 2
    nolds = np.max(edges) + 1
    if n_nold is None or n_nold < nolds:
        n_nold = nolds
    mat = np.zeros((n_nold, n_nold))
    mat[edges[:, 0], edges[:, 1]] = 1
    return mat


def mat_to_edges(mat):
    return np.asarray(np.where(mat == 1)).T


def score_to_accuracy(scores, truths, thresholds=np.arange(0, 1.01, 0.01)):
    assert scores.shape == truths.shape
    assert scores.ndim in [1, 2]
    if scores.ndim == 2:
        scores = skip_diag_tri(scores)
        truths = skip_diag_tri(truths)
    accs = []
    for t in thresholds:
        pos = truths[scores >= t]
        true_pos = np.count_nonzero(pos)
        neg = truths[scores < t]
        true_neg = len(neg) - np.count_nonzero(neg)
        accs.append((true_pos + true_neg) / truths.size)
    return accs


def to_positive(arr):
    arr_tmp = arr.copy()
    arr_tmp[np.where(arr < 0)] = 0.
    return arr_tmp


def nan_to_val(arr, val=0.):
    arr_tmp = arr.copy()
    arr_tmp[np.where(np.isnan(arr_tmp))] = val
    return arr_tmp


def revise_strength(arr):
    arr[np.where(np.isnan(arr))] = 0.
    arr[np.where(arr < 0)] = 0.


def normalize_by_maximum(arr):
    arr_tmp = arr.copy()
    return arr_tmp / np.nanmax(arr_tmp)


def cross_sum_row_and_col(arr):
    n = arr.shape[0]
    out = np.empty((n, n, n))
    for i, row in enumerate(arr):
        for j, col in enumerate(arr.T):
            out[i, j] = row + col
    return out


def exclude_range_mat(n_row, n_excluded):
    return skip_diag_tri(np.tile(np.arange(n_row), (n_row, 1)), n_excluded)


def score_seq_to_matrix(score_list):
    n_score = len(score_list)
    n_var = int(np.sqrt(n_score)) + 1
    rev_list = list(score_list)[::-1]
    mat = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(n_var):
            if i != j:
                mat[i, j] = rev_list.pop()
    return mat


def discretize_score(score, v=0.5):
    out = score.copy()
    out[out >= v] = 1.
    out[out < v] = 0.
    return out
