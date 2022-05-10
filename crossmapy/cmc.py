# -*- coding: utf-8 -*-
import numpy as np
from itertools import tee
from sklearn import metrics

from . import embed
from . import utils
from ._base import _DirectCausality


class _EmbeddingCausality2(object):
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
        assert not (np.any(np.isnan(data)) and np.any(np.isinf(data))), \
            'Unsupported data that contains nan or inf.'

        self.n_var = data.shape[1]

        dismats = self._dismats(data)
        dismats1, dismats2 = tee(dismats, 2)
        self.ids = (utils.dismat_to_idx(dismat) for dismat in dismats1)
        self.zeros = [utils.counts_zeros_of_dismat(dismat) for dismat in dismats2]

        self.n_embed = data.shape[0] - self.lag * (self.embed_dim - 1)
        n_val_neighbors = self.n_embed - 1

        self.skip_var_idx = []
        for i, zeros in enumerate(self.zeros):
            if not np.all(n_val_neighbors - zeros >= self.n_neighbor):
                print(f"Warning: does not have enough neighbors for variable {i}, "
                      "set causal strength to 0. by default.")
                self.skip_var_idx.append(i)

    def _fit_data(self):
        raise NotImplementedError

    def _dismats(self, data):
        for i in range(self.n_var):
            e = embed.embed_vector(data[:, i], self.embed_dim, lag=self.lag)
            yield utils.embed_to_dismat(e, self.n_excluded)


class CrossMappingCardinality(_EmbeddingCausality2):
    def fit(self, data):
        assert data.shape[1] > 1, 'data must have more than 1 column (variable).'
        self._preprocess(data)
        self._fit_data()

    def _fit_data(self):
        self.scores = np.zeros((self.n_var, self.n_var))
        maps = iter(self._idx_to_mapping())
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                for j in range(self.n_var):
                    if (i != j) and (j not in self.skip_var_idx):
                        ratio = self.mapping_to_ratio(next(maps))
                        self.scores[i, j] = self.ratio_to_score(ratio)
        utils.revise_strength(self.scores)

    def _idx_to_mapping(self):
        self.ids, ids_copy = tee(self.ids, 2)
        for i, idx in enumerate(ids_copy):
            if i not in self.skip_var_idx:
                self.ids, ids_copy = tee(self.ids, 2)
                for j, idy in enumerate(ids_copy):
                    if (i != j) and (j not in self.skip_var_idx):
                        yield utils.idx_to_mapping(idx, idy, self.zeros[j], self.n_neighbor)

    @staticmethod
    def count_mapping(map_x2y, tgt_neighbor):
        return len(np.where(map_x2y < tgt_neighbor)[-1])

    def mapping_to_ratio(self, map_x2y):
        n_row = map_x2y.shape[0]
        n_ele = map_x2y.size
        ratios_x2y = np.asarray([self.count_mapping(map_x2y, i) for i in range(n_row)]) / n_ele
        return ratios_x2y

    @staticmethod
    def ratio_to_auc(ratios):
        n_ele = len(ratios)
        neighbor_ratios = np.arange(n_ele) / (n_ele - 1)
        auc = metrics.auc(neighbor_ratios, ratios)
        return auc

    @staticmethod
    def auc_to_score(aucs):
        dat = np.array(aucs)
        dat[dat < 0.5] = 0.5
        scores = 2 * (dat - .5)
        return scores

    def ratio_to_score(self, ratios):
        return self.auc_to_score(self.ratio_to_auc(ratios))


class DirectCrossMappingCardinality(CrossMappingCardinality, _DirectCausality):
    def fit(self, data, max_conditions='auto', CMC_scores=None):
        assert data.shape[1] > 2, 'data must have more than 2 columns (variables).'
        self._preprocess(data)
        if CMC_scores is None:
            self._fit_data()
        else:
            self.scores = CMC_scores
        self._fit_data_dir(max_conditions)

    def _fit_data_dir(self, max_conditions):
        max_c = self._check_max_conditions(max_conditions)
        cross_sum = utils.cross_sum_row_and_col(self.scores)
        maps = self._idx_to_mapping_list()

        dir_scores = np.zeros((self.n_var, self.n_var, max_c))
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                for j in range(self.n_var):
                    if (i != j) and (j not in self.skip_var_idx):
                        n_condition = 0
                        tmp_idx = np.argsort(cross_sum[i, j])
                        tmp_ex_idx = tmp_idx[~np.isin(tmp_idx, [i, j])]
                        for k in range(self.n_var):
                            if k != i and k != j and k in tmp_ex_idx[:max_c]:
                                try:
                                    ratios_i2j = self.mapping_to_ratio(maps[i][j])
                                    ratios_k2j = self.mapping_to_ratio(maps[k][j])
                                    map_i2k = maps[i][k]
                                    ratio = self.ratio_to_dir_ratio(
                                        ratios_i2j, ratios_k2j, map_i2k, self.n_neighbor)
                                    dir_scores[i, j, n_condition] = self.ratio_to_score(ratio)
                                except:
                                    dir_scores[i, j, n_condition] = self.scores[i, j]
                                n_condition += 1
        utils.revise_strength(dir_scores)
        self.scores = np.min(dir_scores, axis=-1)

    def _idx_to_mapping_list(self):
        maps = []
        self.ids, ids_copy = tee(self.ids, 2)
        for i, idx in enumerate(ids_copy):
            maps_ = []
            self.ids, ids_copy = tee(self.ids, 2)
            for j, idy in enumerate(ids_copy):
                if (i == j) or (i in self.skip_var_idx) or (j in self.skip_var_idx):
                    maps_.append(None)
                else:
                    maps_.append(utils.idx_to_mapping(idx, idy, self.zeros[j], self.n_neighbor))
            maps.append(maps_)
        return maps

    def ratio_to_dir_ratio(self, ratios_x2y, ratios_z2y, map_x2z, n_neighbor):
        dir_ratios_x2y = ratios_x2y - self.count_mapping(map_x2z, n_neighbor) / map_x2z.size * ratios_z2y
        return dir_ratios_x2y
