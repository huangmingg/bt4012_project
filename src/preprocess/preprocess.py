import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sampling.sampling import SamplingAlgorithm
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class DatasetWrapper(ABC):

    def __init__(self, filename: str) -> None:
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_df = pd.read_csv(os.path.join(parent_dir, 'data', filename))
        self.preprocess()

    @abstractmethod
    def preprocess(self) -> None:
        pass

    def balance(self, algorithm: SamplingAlgorithm, **kwargs) -> None:
        oversampling_level: List[float] = kwargs.get('oversampling_level', [0.5])
        random_state: int = kwargs.get('random_state', 4012)
        # safe dictionary key deletion
        kwargs.pop("oversampling_level", None)
        kwargs.pop("random_state", None)
        self.b = {}
        for level in oversampling_level:
            self.b[level] = []
            for i in range(len(self.x_train)):
                bxt, byt = algorithm.run(self.x_train[i], self.y_train[i], self.columns, level, random_state, **kwargs)
                self.b[level].append((bxt, byt))
            

class CreditCardDataset(DatasetWrapper):

    def __init__(self, filepath: os.path) -> None:
        super().__init__(filepath)

    def preprocess(self) -> None:
        self.raw_df = self.raw_df[~self.raw_df.duplicated(keep='last')]
        y = self.raw_df['Class'].to_numpy()
        x = self.raw_df.drop(['Class', 'Time'], axis=1)
        self.columns = x.columns
        x = x.to_numpy()
        scaler = StandardScaler()
        self.x_train, self.x_test, self.y_train, self.y_test = [], [], [], []
        r = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=4012)
        for train_index, test_index in r.split(x, y):
            self.x_train.append(scaler.fit_transform(x[train_index]))
            self.x_test.append(scaler.transform(x[test_index]))
            self.y_train.append(y[train_index])
            self.y_test.append(y[test_index])        
        

class SwarmDataset(DatasetWrapper):
    def __init__(self, filepath: os.path) -> None:
        super().__init__(filepath)

    def preprocess(self) -> None:
        self.raw_df = self.raw_df[~self.raw_df.duplicated(keep='last')].head(2300)
        y = self.raw_df['Swarm_Behaviour']
        x = self.raw_df.drop(['Swarm_Behaviour'], axis=1)
        self.columns = x.columns
        x = x.to_numpy()
        scaler = StandardScaler()
        self.x_train, self.x_test, self.y_train, self.y_test = [], [], [], []
        r = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=4012)
        for train_index, test_index in r.split(x, y):
            self.x_train.append(scaler.fit_transform(x[train_index]))
            self.x_test.append(scaler.transform(x[test_index]))
            self.y_train.append(y[train_index])
            self.y_test.append(y[test_index])               
        

class AdultDataset(DatasetWrapper):

        def __init__(self, filepath: os.path) -> None:
            super().__init__(filepath)

        def preprocess(self) -> None:
            self.raw_df[self.raw_df=='?'] = np.nan
            self.raw_df = self.raw_df.dropna(subset=['workclass', 'occupation', 'native.country'])
            self.raw_df = self.raw_df.drop_duplicates()

            # identify numerical and categorical columns
            x = self.raw_df.drop('income', axis=1)
            y = self.raw_df.income
            self.columns = x.columns
            self.num_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
            self.cat_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship' ,'race', 'sex', 'native.country']  
            self.num_columns_ind = [self.columns.tolist().index(num) for num in self.num_columns]
            self.cat_columns_ind = [self.columns.tolist().index(cat) for cat in self.cat_columns]      

            x = x.to_numpy()
            scaler = StandardScaler()
            self.x_train, self.x_test, self.y_train, self.y_test = [], [], [], []
            r = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=4012)
            for train_index, test_index in r.split(x, y):
                x_train = x[train_index]
                x_test = x[test_index]
                x_train[:,self.num_columns_ind] = scaler.fit_transform(x_train[:,self.num_columns_ind])
                x_test[:,self.num_columns_ind] = scaler.transform(x_test[:,self.num_columns_ind])
                self.x_train.append(x_train)
                self.x_test.append(x_test)
                self.y_train.append(y[train_index])
                self.y_test.append(y[test_index])        
        
            # # split train-test
            # x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y.to_numpy(), train_size=0.80, random_state=4012, stratify=y)

            # scale numerical columns
            # ss = StandardScaler()
            # x_train[:,self.num_columns_ind] = ss.fit_transform(x_train[:,self.num_columns_ind])
            # x_test[:,self.num_columns_ind] = ss.transform(x_test[:,self.num_columns_ind])

            # self.x_train = x_train
            # self.x_test = x_test
            # self.y_train = y_train
            # self.y_test = y_test


