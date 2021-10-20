import pandas as pd
from preprocess.preprocess import DatasetWrapper
from sampling.sampling import SamplingAlgorithm


class SmoteAlgorithm(SamplingAlgorithm):
    @staticmethod
    def run(imbalanced_train: DatasetWrapper) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Running SMOTE algorithm")
        x_train, y_train = SMOTE().fit_resample(imbalanced_train.x_train, imbalanced_train.y_train)
        return (x_train, y_train)