import pandas as pd
import numpy as np
from preprocess.preprocess import DatasetWrapper
from sampling.sampling import SamplingAlgorithm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.pipeline import Pipeline
from typing import Tuple

class SmoteAlgorithm(SamplingAlgorithm):
    
    @staticmethod
    def run(x_train: np.array, y_train: np.array, **kwargs) -> Tuple[np.array, np.array]:
#         minority_ratio = len(y_train[y_train==0])*5
#         majority_ratio = len(y_train[y_train==1])*6
#         under = RandomUnderSampler(sampling_strategy={1: majority_ratio})
#         steps = [('o', over), ('u', under)]
#         pipeline = Pipeline(steps=steps)
#         balanced_x, balanced_y = pipeline.fit_resample(x_train, y_train)

        balanced_x, balanced_y = SMOTE().fit_resample(x_train, y_train)
        return balanced_x, balanced_y
