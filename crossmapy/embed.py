from numba import njit
import numpy as np


@njit
def embed_vector(vec, embed_dim, lag=1, slide=1):
    """
    delay embedding a vector or 1d array.

    Parameters
    ----------
    vec: list or 1d array
        input vector
    embed_dim: int
        delay embedding dimension
    lag: int
        delay time
    slide: int
        slide step

    Returns
    -------
    delay_mat: 2d array
        matrix after delay embedding
    """
    vec = np.asarray(vec)
    n_ele = vec.shape[0]
    n_dim = n_ele - (lag * (embed_dim - 1))
    row_idx = np.arange(0, n_dim, slide)

    delay_mat = np.zeros((len(row_idx), embed_dim))
    for i, idx in enumerate(row_idx):
        end_val = idx + lag * (embed_dim - 1) + 1
        part = vec[idx: end_val]
        delay_mat[i, :] = part[::lag]

    return delay_mat


def embed_data(data, embed_dim, **kwargs):
    return np.apply_along_axis(embed_vector, 1, data.T, embed_dim, **kwargs)


@njit
def embed_recover_by_index(delay_mat, lag=1, index=0):
    """
    recover the delay embedding matrix by index.

    Parameters
    ----------
    delay_mat: 2d array
        delay embedding matrix
    lag: int
        delay time
    index: int
        column index of delay_mat

    Returns
    -------
    vec: 1d array
        output vector
    """
    n_row, embed_dim = delay_mat.shape
    n_ele = n_row + (lag * (embed_dim - 1))

    vec = np.zeros(n_ele)
    s = 0
    for i in range(embed_dim):
        if i < index:
            e = s + lag
            vec[s: e] = delay_mat[:lag, i]
            s += lag
        elif i > index:
            e = s + lag
            vec[s: e] = delay_mat[-lag:, i]
            s += lag
        else:
            e = s + n_row
            vec[s: e] = delay_mat[:, i]
            s += n_row

    return vec


@njit
def embed_recover_by_average(delay_mat, lag=1):
    """
    recover the delay embedding matrix by average.

    Parameters
    ----------
    delay_mat: 2d array
        delay embedding matrix
    lag: int
        delay time

    Returns
    -------
    vec: 1d array
        output vector
    """
    n_row, embed_dim = delay_mat.shape
    n_ele = n_row + (lag * (embed_dim - 1))

    vec = np.zeros(n_ele)
    for n in range(n_ele):
        tmp = .0
        count = 0
        for j in range(embed_dim):
            i = n - j * lag
            if 0 <= i < n_row:
                tmp += delay_mat[i, j]
                count += 1
        vec[n] = tmp / count

    return vec

