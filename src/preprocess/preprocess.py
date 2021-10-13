from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Tuple
import os
import pandas as pd


class DatasetWrapper(ABC):

    def __init__(self, filename: str) -> None:
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_df = pd.read_csv(os.path.join(parent_dir, 'data', filename))
        self.process()


    @abstractmethod
    def process(self) -> None:
        pass



class CreditCardDataset(DatasetWrapper):

    def __init__(self, filepath: os.path) -> None:
        super().__init__(filepath)


    def process(self) -> None:
        '''Initializes attribute x_train, x_test, y_train, y_test'''
        y = self.raw_df['Class']

        ## drop unnecessary columns
        x = self.raw_df.drop(['Class', 'Time'], axis=1)
        x = x[['V1']]
        self.columns = x.columns

        ## handle missing data

        ## scale data

        ## split data

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, train_size=0.80, random_state=4012)
        # print(self.raw_df)


