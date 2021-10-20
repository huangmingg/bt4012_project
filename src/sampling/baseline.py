import pandas as pd
from preprocess.preprocess import DatasetWrapper
from sampling.sampling import SamplingAlgorithm


class BaselineAlgorithm(SamplingAlgorithm):
    @staticmethod
    def run(imbalanced_train: DatasetWrapper) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (imbalanced_train.x_train, imbalanced_train.y_train)