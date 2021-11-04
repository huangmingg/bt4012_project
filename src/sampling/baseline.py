import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sampling.sampling import SamplingAlgorithm
from typing import Tuple, List


class BaselineAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array, columns: List[str], oversampling_level: float = 0.5, 
    random_state: int = 4012, **kwargs) -> Tuple[np.array, np.array]:
        oversample = RandomOverSampler(sampling_strategy=oversampling_level, random_state=random_state)
        bxt, byt = oversample.fit_resample(x_train, y_train)
        return bxt, byt
