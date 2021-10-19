from abc import ABC, abstractmethod
from imblearn.over_sampling import SMOTE
import random
import pandas as pd

from sklearn.neighbors import NearestNeighbors

class SamplingAlgorithm(ABC):
    def run(self, imbalanced_train: pd.DataFrame):
        self.x_train = imbalanced_train.x_train
        self.y_train = imbalanced_train.y_train
    
class SmoteAlgorithm():
    def run(self, imbalanced_train: pd.DataFrame):
        print("Running SMOTE algorithm")
        self.x_train, self.y_train = SMOTE().fit_resample(imbalanced_train.x_train, imbalanced_train.y_train)
        
     
        
        
    
