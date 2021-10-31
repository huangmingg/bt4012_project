from abc import ABC, abstractmethod
from sampling.sampling import SamplingAlgorithm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from typing import Tuple
import os
import pandas as pd
import numpy as np


class DatasetWrapper(ABC):

    def __init__(self, filename: str) -> None:
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_df = pd.read_csv(os.path.join(parent_dir, 'data', filename))
        self.preprocess()

    @abstractmethod
    def preprocess(self) -> None:
        pass

    def balance(self, algorithm: SamplingAlgorithm) -> None:
        self.bxt, self.yxt = algorithm.run(self.x_train, self.y_train, columns=self.columns)


class CreditCardDataset(DatasetWrapper):

    def __init__(self, filepath: os.path) -> None:
        super().__init__(filepath)

    def preprocess(self) -> None:
        """Initializes attribute x_train, x_test, y_train, y_test"""

        self.raw_df = self.raw_df[~self.raw_df.duplicated(keep='last')]

        y = self.raw_df['Class']

        # drop unnecessary columns
        x = self.raw_df.drop(['Class', 'Time'], axis=1)
        self.columns = x.columns

        # split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x.to_numpy(), y.to_numpy(), train_size=0.80, random_state=4012)
        scaler = preprocessing.StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
