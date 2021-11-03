import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple
from sampling.sampling import SamplingAlgorithm
from scipy.stats import chi2
from sklearn.covariance import MinCovDet


class McdSmote(SamplingAlgorithm):
    """Oversample using our customized experimental algorithm McdSmote.

    Similar to robROSE, McdSMOTE uses MCD algorithm (mahalanobis distance) detect outliers of the minority class,
    which is then removed from the resampling dataset. It then performs a SMOTE algorithm to generate the synthethic dataset
    that only consists of the inlier minority class examples.

    McdSmote currently only supports binary classification.
    """

    @staticmethod
    def run(x_train: np.array, y_train: np.array, p: float=0.999, sp: float=0.95, random_state: int=4012, **kwargs) -> Tuple[np.array, np.array]:
        minority_index = np.where(y_train == 1)
        majority_index = np.where(y_train == 0)
        minority = x_train[minority_index]
        majority = x_train[majority_index]

        # runs MCD algorithm and generate the mahalanobis for the dataset
        cov = MinCovDet(assume_centered=False, support_fraction=sp, random_state=random_state).fit(minority)
        md = cov.mahalanobis(minority)
        threshold = chi2.ppf(p, minority.shape[1])

        outliers_index = np.where(md >= threshold)
        inliers_index = np.where(md < threshold)
        minority = minority[inliers_index]
        new_x_train = np.concatenate((majority, minority), axis=0)
        new_y_train = np.concatenate((np.zeros(majority.shape[0]), np.ones(minority.shape[0])), axis=None)

        # runs SMOTE algorithm to generate synthetic data for the inlier minority class examples
        balanced_x, balanced_y = SMOTE().fit_resample(new_x_train, new_y_train)
        return balanced_x, balanced_y
