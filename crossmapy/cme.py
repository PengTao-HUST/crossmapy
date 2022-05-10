# -*- coding: utf-8 -*-
import numpy as np

from . import mi
from . import utils
from ._base import _EmbeddingCausality, _DirectCausality


class CrossMappingEntropy(_EmbeddingCausality):
    def fit(self, data, mi_kwargs=None):
        assert data.shape[1] > 1, 'data must have more than 1 column (variable).'
        self._preprocess(data)
        self._fit_data(mi_kwargs)

    def _fit_data(self, mi_kwargs):
        n_row = self.ids[0].shape[0]
        mi_kwargs = {} if mi_kwargs is None else mi_kwargs.copy()
        self.use_index = utils.exclude_range_mat(n_row, self.n_excluded)
        self.scores = np.zeros((self.n_var, self.n_var))
        for i in range(self.n_var):
            neighbor_tgt = self._neighbor(i, i)
            for j in range(self.n_var):
                if i != j:
                    neighbor_src = self._neighbor(i, j)
                    self.scores[i, j] = mi.cal_mi_from_knn(neighbor_src, neighbor_tgt, **mi_kwargs)
        utils.revise_strength(self.scores)

    def _neighbor(self, i, j):
        return np.asarray(
                [self.embeddings[i][self.use_index[k]][
                     self.ids[j][k, self.zeros[j][k]: self.zeros[j][k] + self.n_neighbor]].ravel()
                 for k in range(self.n_embed)])


class DirectCrossMappingEntropy(CrossMappingEntropy, _DirectCausality):
    def fit(self, data, mi_kwargs=None, max_conditions='auto'):
        assert data.shape[1] > 2, 'data must have more than 2 columns (variables).'
        self._preprocess(data)
        self._fit_data(mi_kwargs)
        self._fit_data_dir(max_conditions, mi_kwargs)

    def _fit_data_dir(self, max_conditions, mi_kwargs):
        max_c = self._check_max_conditions(max_conditions)
        cross_sum = utils.cross_sum_row_and_col(self.scores)
        dir_scores = np.zeros((self.n_var, self.n_var, max_c))

        n_row = self.ids[0].shape[0]
        mi_kwargs = {} if mi_kwargs is None else mi_kwargs.copy()
        self.use_index = utils.exclude_range_mat(n_row, self.n_excluded)
        for i in range(self.n_var):
            neighbor_tgt = self._neighbor(i, i)
            for j in range(self.n_var):
                if i != j:
                    neighbor_src = self._neighbor(i, j)
                    n_condition = 0
                    tmp_idx = np.argsort(cross_sum[i, j])
                    tmp_ex_idx = tmp_idx[~np.isin(tmp_idx, [i, j])]
                    for k in range(self.n_var):
                        if k != i and k != j and k in tmp_ex_idx[:max_c]:
                            neighbor_con = self._neighbor(i, k)
                            dir_scores[i, j, n_condition] = mi.cal_cmi_from_knn(
                                neighbor_src, neighbor_tgt, neighbor_con, **mi_kwargs)
                            n_condition += 1
        self.scores = np.min(dir_scores, axis=-1)
        utils.revise_strength(self.scores)


class DirectCrossMappingEntropySimple(CrossMappingEntropy, _DirectCausality):
    def fit(self, data, mi_kwargs=None, max_conditions='auto'):
        assert data.shape[1] > 2, 'data must have more than 2 columns (variables).'
        self._preprocess(data)
        self._fit_data(mi_kwargs)
        self._fit_data_dir(max_conditions)

    def _fit_data_dir(self, max_conditions):
        max_c = self._check_max_conditions(max_conditions)
        cross_sum = utils.cross_sum_row_and_col(self.scores)
        dir_scores = np.zeros((self.n_var, self.n_var, max_c))
        for i in range(self.n_var):
            for j in range(self.n_var):
                if i != j:
                    n_condition = 0
                    tmp_idx = np.argsort(cross_sum[i, j])
                    tmp_ex_idx = tmp_idx[~np.isin(tmp_idx, [i, j])]
                    for k in range(self.n_var):
                        if k != i and k != j and k in tmp_ex_idx[:max_c]:
                            indir_score = self._indiret_score(i, j, k)
                            dir_scores[i, j, n_condition] = self.scores[i, j] - indir_score
                            n_condition += 1
        self.scores = np.min(dir_scores, axis=-1)
        utils.revise_strength(self.scores)

    def _indiret_score(self, i, j, k):
        return self.scores[i, k] * self.scores[k, j]
