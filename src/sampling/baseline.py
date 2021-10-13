import pandas as pd
from sampling.sampling import SamplingAlgorithm


class BaselineAlgorithm(SamplingAlgorithm):


    @staticmethod
    def run(imbalanced_train: pd.DataFrame) -> pd.DataFrame:
        return imbalanced_train