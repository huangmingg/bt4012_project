import math
from collections import Counter
from typing import Tuple


import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import _safe_indexing
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0
from sklearn.utils.sparsefuncs_fast import csc_mean_variance_axis0

from sampling.sampling import SamplingAlgorithm
from imblearn.over_sampling import ADASYN, SMOTENC


class Adasyn(SamplingAlgorithm):

    @staticmethod
    def run(x_train: np.array, y_train: np.array) -> Tuple[np.array, np.array]:
        balanced_x, balanced_y = ADASYN().fit_resample(x_train, y_train)
        return balanced_x, balanced_y


class AdasynNCAlgorithm(SamplingAlgorithm):

    @staticmethod
    def run(x_train: np.array, y_train: np.array, **kwargs) -> Tuple[np.array, np.array]:
        categorical_features = kwargs['categorical_features']
        balanced_x, balanced_y = ADASYNNC(
            categorical_features).fit_resample(x_train, y_train)
        return balanced_x, balanced_y


class ADASYNNC(SMOTENC):
    """
    Class for modified ADASYN algorithm which handles both categorical and numerical features. Uses SMOTENC's handling of categorical features
    and ADASYN's method for selecting the number of samples to synthesise from each minority class.
    """

    def __init__(
        self,
        categorical_features,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            categorical_features=categorical_features
        )

    def _fit_resample(self, X, y):
        self.n_features_ = X.shape[1]
        self._validate_estimator()

        # compute the median of the standard deviation of the minority class
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, self.continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=["csr", "csc"])
        X_minority = _safe_indexing(
            X_continuous, np.flatnonzero(y == class_minority)
        )

        if sparse.issparse(X):
            if X.format == "csr":
                _, var = csr_mean_variance_axis0(X_minority)
            else:
                _, var = csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        self.median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, self.categorical_features_]
        if X_continuous.dtype.name != "object":
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = np.float64
        self.ohe_ = OneHotEncoder(
            sparse=True, handle_unknown="ignore", dtype=dtype_ohe
        )

        # the input of the OneHotEncoder needs to be dense
        X_ohe = self.ohe_.fit_transform(
            X_categorical.toarray()
            if sparse.issparse(X_categorical)
            else X_categorical
        )

        # we can replace the 1 entries of the categorical features with the
        # median of the standard deviation. It will ensure that whenever
        # distance is computed between 2 samples, the difference will be equal
        # to the median of the standard deviation as in the original paper.

        # In the edge case where the median of the std is equal to 0, the 1s
        # entries will be also nullified. In this case, we store the original
        # categorical encoding which will be later used for inversing the OHE
        if math.isclose(self.median_std_, 0):
            self._X_categorical_minority_encoded = _safe_indexing(
                X_ohe.toarray(), np.flatnonzero(y == class_minority)
            )

        X_ohe.data = (
            np.ones_like(X_ohe.data, dtype=X_ohe.dtype) * self.median_std_ / 2
        )
        X_encoded = sparse.hstack((X_continuous, X_ohe), format="csr")

#         X_resampled, y_resampled = super()._fit_resample(X_encoded, y)
        X_resampled, y_resampled = self._adasyn_fit_resample(X_encoded, y)

        # reverse the encoding of the categorical features
        X_res_cat = X_resampled[:, self.continuous_features_.size:]
        X_res_cat.data = np.ones_like(X_res_cat.data)
        X_res_cat_dec = self.ohe_.inverse_transform(X_res_cat)

        if sparse.issparse(X):
            X_resampled = sparse.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size],
                    X_res_cat_dec,
                ),
                format="csr",
            )
        else:
            X_resampled = np.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size].toarray(),
                    X_res_cat_dec,
                )
            )

        indices_reordered = np.argsort(
            np.hstack((self.continuous_features_, self.categorical_features_))
        )
        if sparse.issparse(X_resampled):
            # the matrix is supposed to be in the CSR format after the stacking
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]

        return X_resampled, y_resampled

    def _adasyn_fit_resample(self, X, y):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled
