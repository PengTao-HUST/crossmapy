# -*- coding: utf-8 -*-
import numpy as np

from . import embed
from . import utils


class _ConventionCausality(object):
    def __init__(self, embed_dim):
        assert isinstance(embed_dim, int) and embed_dim > 1, \
            'Embedding dimension must be integer (> 1).'
        self.embed_dim = embed_dim

    def fit(self):
        self._fit_data()

    def _preprocess(self, data):
        assert data.ndim == 2, 'Input data should be 2D array of shape (n_points, n_variables).'
        assert np.all(~np.isnan(data)) and np.all(~np.isinf(data)), \
            'Unsupported data that contains nan or inf.'

        self.data = data
        self.n_var = data.shape[1]
        cut = self.data.shape[0] - self.embed_dim
        self.embeddings = embed.embed_data(data, self.embed_dim)[:, :cut]

    def _fit_data(self):
        raise NotImplementedError


class _EmbeddingCausality(object):
    def __init__(self, embed_dim, lag=1, n_neighbor=None, n_excluded=0):
        assert isinstance(embed_dim, int) and embed_dim > 1, \
            'Embedding dimension must be integer (> 1).'

        if n_neighbor is None:
            n_neighbor = embed_dim + 1
        assert isinstance(n_neighbor, int) and n_neighbor > 2, \
            'Number of neighbors must be integer (> 2).'

        assert isinstance(lag, int) and lag >= 1, \
            'Delay time constant must be integer (>= 1).'

        assert isinstance(n_excluded, int) and n_neighbor >= 0, \
            'Number of excluded neighbors must be integer (>= 1).'

        self.embed_dim = embed_dim
        self.n_neighbor = n_neighbor
        self.lag = lag
        self.n_excluded = n_excluded

    def fit(self):
        self._fit_data()

    def _preprocess(self, data):
        assert data.ndim == 2, 'Input data should be 2D array of shape (n_points, n_variables).'
        assert np.all(~np.isnan(data)) and np.all(~np.isinf(data)), \
            'Unsupported data that contains nan or inf.'

        self.data = data
        self.n_var = data.shape[1]

        self.embeddings = embed.embed_data(data, self.embed_dim, lag=self.lag)
        self.dismats = [utils.embed_to_dismat(e, self.n_excluded) for e in self.embeddings]
        self.ids = [utils.dismat_to_idx(dismat) for dismat in self.dismats]
        self.zeros = [utils.counts_zeros_of_dismat(dismat) for dismat in self.dismats]
        self.n_embed = self.embeddings[0].shape[0]
        n_val_neighbors = self.dismats[0].shape[1]

        self.skip_var_idx = []
        for i, zeros in enumerate(self.zeros):
            if not np.all(n_val_neighbors - zeros >= self.n_neighbor):
                print(f"Warning: does not have enough neighbors for variable {i}, "
                      "set causal strength to 0. by default.")
                self.skip_var_idx.append(i)

    def _fit_data(self):
        raise NotImplementedError


class _DirectCausality(object):
    def _fit_data_dir(self):
        raise NotImplementedError

    def _check_max_conditions(self, max_conditions):
        if isinstance(max_conditions, int):
            max_c = max_conditions
            if max_c <= 0:
                max_c = 1
            if max_c > self.n_var - 2:
                max_c = self.n_var - 2
        elif isinstance(max_conditions, str):
            if max_conditions == 'auto':
                if self.n_var > 5:
                    max_c = 3
                else:
                    max_c = self.n_var - 2
            elif max_conditions == 'full':
                max_c = self.n_var - 2
            elif max_conditions == 'fast':
                max_c = 1
            else:
                print('Warning: unknown max_conditions, set to 1 by default.')
                max_c = 1
        else:
            print('Warning: unknown max_conditions, set to 1 by default.')
            max_c = 1
        return max_c
