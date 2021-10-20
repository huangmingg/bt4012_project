import pandas as pd
import numpy as np
from preprocess.preprocess import DatasetWrapper
from sampling.sampling import SamplingAlgorithm
from imblearn.over_sampling import SMOTE
from typing import Tuple

class SmoteAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array) -> Tuple[np.array, np.array]:
        balanced_x, balanced_y = SMOTE().fit_resample(x_train, y_train)
        return balanced_x, balanced_y
