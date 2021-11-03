import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sampling.sampling import SamplingAlgorithm
from typing import Tuple, List


class BaselineAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array, columns: List[str], **kwargs) -> Tuple[np.array, np.array]:
        oversample = RandomOverSampler(sampling_strategy='minority')
        bxt, byt = oversample.fit_resample(x_train, y_train)
        return bxt, byt
