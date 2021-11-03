import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from typing import Tuple
from sampling.sampling import SamplingAlgorithm
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors


class McdAdasyn(SamplingAlgorithm):
    """Oversample using our customized experimental algorithm McdAdasyn.

    McdAdasyn uses MCD algorithm uses MCD algorithm to compute the mahalanobis distance of the minority class examples,
    and places a distributed weight for the minority class examples (higher mahalanobis distance -> harder to learn example 
    -> higher weights). It then performs a similar algorithm to Adasyn, by using randomly generating synthetic data based on
    the K nearest neighbor, with the difference being this algorithm uses mahalanobis distance whereas Adasyn uses euclidean distance.
    
    McdAdasyn currently only supports binary classification.   
    """

    @staticmethod
    def run(x_train: np.array, y_train: np.array, p: float=0.999, sp: float=0.95, random_state: int=4012, **kwargs) -> Tuple[np.array, np.array]:
        np.random.SeedSequence(random_state)
        minority_index = np.where(y_train == 1)
        majority_index = np.where(y_train == 0)

        minority = x_train[minority_index]
        majority = x_train[majority_index]

        minority_resampled = [minority]

        d = minority.shape[0] / majority.shape[0]
        if not 0 < d < 1:
            print("Degree of imbalanced is incorrect, please check dataset")

        g = majority.shape[0] - minority.shape[0]
        cov = MinCovDet(assume_centered=False, support_fraction=sp, random_state=random_state).fit(minority)
        md = cov.mahalanobis(minority)
        md /= np.sum(md)
        n_samples_generate = np.rint(md * g).astype(int)
        nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='mahalanobis', metric_params={'V': np.cov(minority)})
        nn.fit(minority)
        distances, indices = nn.kneighbors(minority)
        for idx, count in np.ndenumerate(n_samples_generate):
            if count == 0:
                continue
            x = minority[idx]
            neighbors = indices[idx]
            neighbor_idx = np.random.choice(neighbors, size=count)
            noise = np.random.uniform(size=count).reshape(count, 1)
            x_picked = np.apply_along_axis(lambda x: minority[x], axis=0, arr=neighbor_idx)
            sampled_x = x + ((x_picked - x) * noise)
            minority_resampled.append(sampled_x)

        minority_resampled = np.vstack(minority_resampled)
        balanced_x_train = np.concatenate((majority, minority_resampled), axis=0)
        balanced_y_train = np.concatenate((np.zeros(majority.shape[0]), np.ones(minority.shape[0])), axis=None)        
        return balanced_x_train, balanced_y_train