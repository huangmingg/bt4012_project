import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class SamplingAlgorithm(ABC):
    
    @staticmethod
    @abstractmethod
    def run(x_train: np.array, y_train: np.array, **kwargs) -> Tuple[np.array, np.array]:
        pass
