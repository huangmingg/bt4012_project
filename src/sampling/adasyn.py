from sampling.sampling import SamplingAlgorithm
from imblearn.over_sampling import ADASYN
import numpy as np
from typing import Tuple


class Adasyn(SamplingAlgorithm):

    @staticmethod
    def run(x_train: np.array, y_train: np.array, **kwargs) -> Tuple[np.array, np.array]:
        balanced_x, balanced_y = ADASYN().fit_resample(x_train, y_train)
        return balanced_x, balanced_y
