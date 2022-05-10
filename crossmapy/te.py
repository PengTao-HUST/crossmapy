# -*- coding: utf-8 -*-
import numpy as np

from . import utils
from . import mi
from ._base import _ConventionCausality

class TransferEntropy(_ConventionCausality):
    def fit(self, data, mi_kwargs=None):
        self._preprocess(data)
        self._fit_data(mi_kwargs)

    def _fit_data(self, mi_kwargs=None):
        mi_kwargs = {} if mi_kwargs is None else mi_kwargs.copy()
        self.scores = np.zeros((self.n_var, self.n_var))
        for i in range(self.n_var):
            for j in range(self.n_var):
                if i != j:
                    self.scores[i, j] = mi.cal_cmi_from_knn(
                        self.data[self.embed_dim:, j:j + 1],
                        self.embeddings[i],
                        self.embeddings[j],
                        **mi_kwargs
                    )
        utils.revise_strength(self.scores)
