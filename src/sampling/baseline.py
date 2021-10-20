from sampling.sampling import SamplingAlgorithm
import numpy as np
from typing import Tuple


class BaselineAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array) -> Tuple[np.array, np.array]:
        return x_train, y_train
