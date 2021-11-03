import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sampling.sampling import SamplingAlgorithm
from typing import Tuple


class BaselineAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array) -> Tuple[np.array, np.array]:
        oversample = RandomOverSampler(sampling_strategy='minority')
        bxt, byt = oversample.fit_resample(x_train, y_train)
        return bxt, byt
