import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple, List
from sampling.sampling import SamplingAlgorithm
from scipy.stats import chi2
from sklearn.covariance import MinCovDet


class McdSmoteAlgorithm(SamplingAlgorithm):
    """Oversample using our customized experimental algorithm McdSmote.

    Similar to robROSE, McdSMOTE uses MCD algorithm (mahalanobis distance) detect outliers of the minority class,
    which is then removed from the resampling dataset. It then performs a SMOTE algorithm to generate the synthethic dataset
    that only consists of the inlier minority class examples.

    McdSmote currently only supports binary classification.
    """

    @staticmethod
    def run(x_train: np.array, y_train: np.array, columns: List[str], oversampling_level: float = 0.5, 
    random_state: int = 4012, **kwargs) -> Tuple[np.array, np.array]:
        sp = kwargs['sp']
        p = kwargs['p']
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
        smote = SMOTE(sampling_strategy=oversampling_level, random_state=random_state)
        bxt, byt = smote.fit_resample(new_x_train, new_y_train)
        return bxt, byt
