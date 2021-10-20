from abc import ABC, abstractmethod
from imblearn.over_sampling import SMOTE
import random
import pandas as pd

from sklearn.neighbors import NearestNeighbors

class SamplingAlgorithm(ABC):
    @staticmethod
    @abstractmethod
    def run(imbalanced_train: DatasetWrapper) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (imbalanced_train.x_train, imbalanced_train.y_train)
    

        
        
    
