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
    def run(x_train: np.array, y_train: np.array, columns: List[str], **kwargs) -> Tuple[np.array, np.array]:
        balanced_x, balanced_y = SMOTE().fit_resample(x_train, y_train)
        return balanced_x, balanced_y


class SmotencAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array, columns: List[str], **kwargs) -> Tuple[np.array, np.array]:
        """Runs SMOTENC algorithm to balance dataset. Intended for mixed datasets (categorical + numerical features)

        Args:
            x_train (np.array): Array containing sample features, where shape is (n_samples, n_features)
            y_train (np.array): Target vector relative to x_train
            **categorical_features (list): List of indices of categorical features
        Returns:
            Tuple[np.array, np.array]: Tuple containing sample features and target vector of balanced dataset
        """

        categorical_features = kwargs['categorical_features']
        balanced_x, balanced_y = SMOTENC(categorical_features=categorical_features).fit_resample(x_train, y_train)
        return balanced_x, balanced_y

