import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple
from sampling.sampling import SamplingAlgorithm
from scipy.stats import chi2
from sklearn.covariance import MinCovDet


class McdSmote(SamplingAlgorithm):

    @staticmethod
    def run(x_train: np.array, y_train: np.array, p: float=0.999, sp: float=0.95, random_state: int=4012, **kwargs) -> Tuple[np.array, np.array]:
        fraud_index = np.where(y_train == 1)
        normal_index = np.where(y_train == 0)

        fraud = x_train[fraud_index]
        normal = x_train[normal_index]
        print(f'Number of original fraud rows = {fraud.shape[0]}') 
        print(f'Number of original normal rows = {normal.shape[0]}') 

        cov = MinCovDet(assume_centered=False, support_fraction=sp, random_state=random_state).fit(fraud)
        md = cov.mahalanobis(fraud)
        threshold = chi2.ppf(p, fraud.shape[1])

        outliers_index = np.where(md >= threshold)
        inliers_index = np.where(md < threshold)
        print(f'Number of outlier fraud rows = {len(outliers_index[0])}')
        print(f'Number of inlier fraud rows = {len(inliers_index[0])}')

        fraud = fraud[inliers_index]
        print(f'Number of fraud rows remaining = {fraud.shape[0]}')
        new_x_train = np.concatenate((normal, fraud), axis=0)
        new_y_train = np.concatenate((np.zeros(normal.shape[0]), np.ones(fraud.shape[0])), axis=None)

        print(f'Shape after removal of fraud outliers - {new_x_train.shape}')

        balanced_x, balanced_y = SMOTE().fit_resample(new_x_train, new_y_train)
        print(f'Number of balanced fraud rows = {len(np.where(balanced_y == 1)[0])}')
        print(f'Number of balanced normal rows = {len(np.where(balanced_y == 0)[0])}')
        return balanced_x, balanced_y
