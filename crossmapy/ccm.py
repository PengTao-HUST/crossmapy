# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats

from . import utils
from ._base import _EmbeddingCausality, _DirectCausality


class ConvergeCrossMapping(_EmbeddingCausality):
    def fit(self, data):
        self._preprocess(data)
        self._fit_data()

    def _fit_data(self):
        n_row = self.ids[0].shape[0]
        self.weights = []
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                tmp_weight = np.asarray([utils.weights_from_distances(
                    self.dismats[i][j][self.ids[i][j, self.zeros[i][j]:self.n_neighbor + self.zeros[i][j]]])
                               for j in range(n_row)])
            else:
                tmp_weight = None
            self.weights.append(tmp_weight)

        self.real = [utils.exclude_vec_mat(v, self.n_excluded) for v in self.data[-n_row:].T]
        self.scores = np.zeros((self.n_var, self.n_var))
        self.predicts = []
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                predicts_ = []
                for j in range(self.n_var):
                    if (i != j) and (j not in self.skip_var_idx):
                        neighbor_real = np.asarray(
                            [self.real[i][k][self.ids[j][k, self.zeros[j][k]:self.n_neighbor + self.zeros[j][k]]]
                             for k in range(n_row)])
                        predict = np.average(neighbor_real, axis=1, weights=self.weights[j])
                        self.scores[i, j] = abs(stats.pearsonr(predict, self.data[-n_row:, i])[0])
                        predicts_.append(predict)
                    else:
                        predicts_.append(None)
            else:
                predicts_ = None
            self.predicts.append(predicts_)
        utils.revise_strength(self.scores)

    def _parse_embedding(self, embedding):
        dismat = utils.embed_to_dismat(embedding, self.n_excluded)
        idx = utils.dismat_to_idx(dismat)
        zero = utils.counts_zeros_of_dismat(dismat)
        return dismat, idx, zero

    def _eval_weight(self, dismat, idx, zero):
        weight = np.asarray([utils.weights_from_distances(
            dismat[j][idx[j, zero[j]:self.n_neighbor + zero[j]]]) for j in range(self.n_embed)])
        return weight


class PartialCrossMapping(ConvergeCrossMapping, _DirectCausality):
    def fit(self, data, max_conditions='auto'):
        self._preprocess(data)
        self._fit_data()
        self._fit_data_dir(max_conditions)

    def _fit_data_dir(self, max_conditions):
        max_c = self._check_max_conditions(max_conditions)
        cross_sum = utils.cross_sum_row_and_col(self.scores)

        n_row = self.ids[0].shape[0]
        self.use_index = utils.exclude_range_mat(n_row, self.n_excluded)
        dir_scores = np.zeros((self.n_var, self.n_var, max_c))
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                for j in range(self.n_var):
                    if (i != j) and (j not in self.skip_var_idx):
                        predict = self.predicts[i][j]
                        n_condition = 0
                        tmp_idx = np.argsort(cross_sum[i, j])
                        tmp_ex_idx = tmp_idx[~np.isin(tmp_idx, [i, j])]
                        for k in range(self.n_var):
                            if k != i and k != j and k in tmp_ex_idx[:max_c]:
                                try:
                                    neighbor_con = self._embedding_transform(k, j)
                                    dismat, idx, zero = self._parse_embedding(neighbor_con)
                                    weight_con = self._eval_weight(dismat, idx, zero)
                                    neighbor_real = np.asarray(
                                        [self.real[i][t][idx[t, zero[t]:self.n_neighbor + zero[t]]]
                                         for t in range(self.n_embed)])
                                    predict_con = np.average(neighbor_real, weights=weight_con, axis=1)
                                    dir_scores[i, j, n_condition] = abs(utils.partial_correlation(
                                        predict, self.data[-n_row:, i], predict_con))
                                except:
                                    dir_scores[i, j, n_condition] = self.scores[i, j]
                                n_condition += 1
        utils.revise_strength(dir_scores)
        self.scores = np.min(dir_scores, axis=-1)

    def _embedding_transform(self, i, j):
        embedding = np.asarray(
            [np.average(self.embeddings[i][self.use_index[k]][
                            self.ids[j][k, self.zeros[j][k]: self.zeros[j][k] + self.n_neighbor]],
                        weights=self.weights[j][k], axis=0) for k in range(self.n_embed)])
        return embedding

