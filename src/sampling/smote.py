import pandas as pd
import numpy as np
from preprocess.preprocess import DatasetWrapper
from sampling.sampling import SamplingAlgorithm
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.pipeline import Pipeline
from typing import Tuple, List

class SmoteAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array, columns: List[str], oversampling_level: float = 0.5, 
    random_state: int = 4012, **kwargs) -> Tuple[np.array, np.array]:
        smote = SMOTE(sampling_strategy=oversampling_level, random_state=random_state)
        bxt, byt = smote.fit_resample(x_train, y_train)
        return bxt, byt


class SmotencAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array, columns: List[str], oversampling_level: float = 0.5, 
    random_state: int = 4012, **kwargs) -> Tuple[np.array, np.array]:
        """Runs SMOTENC algorithm to balance dataset. Intended for mixed datasets (categorical + numerical features)

        Args:
            x_train (np.array): Array containing sample features, where shape is (n_samples, n_features)
            y_train (np.array): Target vector relative to x_train
            **categorical_features (list): List of indices of categorical features
        Returns:
            Tuple[np.array, np.array]: Tuple containing sample features and target vector of balanced dataset
        """

        categorical_features = kwargs['categorical_features']
        smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy=oversampling_level, random_state=random_state)
        bxt, byt = smotenc.fit_resample(x_train, y_train)
        return bxt, byt
