import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List


class SamplingAlgorithm(ABC):
    
    @staticmethod
    @abstractmethod
    def run(x_train: np.array, y_train: np.array, columns: List[str], 
    oversampling_level: float = 0.5, random_state: int = 4012, **kwargs) -> Tuple[np.array, np.array]:
        pass
