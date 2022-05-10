# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression

from . import utils
from ._base import _ConventionCausality


class GrangerCausality(_ConventionCausality):
    def fit(self, data, ddof=1, **kwargs):
        self._preprocess(data)
        self._fit_data(ddof=1, **kwargs)

    def _fit_data(self, ddof=1, **kwargs):
        self.scores = np.zeros((self.n_var, self.n_var))
        for i in range(self.n_var):
            for j in range(self.n_var):
                if i != j:
                    self.scores[i, j] = self._estimate_gc_index(
                        self.embeddings[j],
                        np.hstack((self.embeddings[i], self.embeddings[j])),
                        self.data[self.embed_dim:, j],
                        ddof,
                        **kwargs
                    )
        utils.revise_strength(self.scores)

    @staticmethod
    def _estimate_gc_index(M1, M2, Y_real, ddof=1, **kwargs):
        """
        estimate the GC index based on single and composite manifolds.

        Parameters
        ----------
        M1: 2d array
            single manifold
        M2: 2d array
            composite manifold
        Y_real: 1d array
            real time series
        ddof: int
            “Delta Degrees of Freedom”
        kwargs:
            other keyword arguments are passed through to "LinearRegression"

        Returns
        -------
        gc_estimated: float
            estimated Granger causality
        """
        # calculate the error between predicted y (based on M1) and real y
        LR1 = LinearRegression(**kwargs).fit(M1, Y_real)
        Y_pred_LR1 = LR1.predict(M1)
        error1 = Y_pred_LR1 - Y_real

        # calculate the error between predicted y (based on M2) and real y
        LR2 = LinearRegression(**kwargs).fit(M2, Y_real)
        Y_pred_LR2 = LR2.predict(M2)
        error2 = Y_pred_LR2 - Y_real

        # estimate the gc index based on two errors
        gc_estimated = -np.log(
            np.var(error2, ddof=ddof) / np.var(error1, ddof=ddof))
        return gc_estimated
