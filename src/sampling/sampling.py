from abc import ABC, abstractmethod
import pandas as pd


class SamplingAlgorithm(ABC):

    @staticmethod
    @abstractmethod
    def run(imbalanced_train: pd.DataFrame) -> pd.DataFrame:
        return imbalanced_train
