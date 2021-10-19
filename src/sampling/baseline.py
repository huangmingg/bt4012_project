import pandas as pd
from sampling.sampling import SamplingAlgorithm


class BaselineAlgorithm():
    def run(self, imbalanced_train: pd.DataFrame):
        print("Running Baseline Algorithm")
        self.x_train = imbalanced_train.x_train
        self.y_train = imbalanced_train.y_train
    