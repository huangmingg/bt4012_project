from abc import ABC, abstractmethod
from preprocess.preprocess import DatasetWrapper
from typing import Tuple
import pickle
import numpy as np


class ClassifierWrapper(ABC):

    def __init__(self, data: DatasetWrapper) -> None:
        self.data = data

    @abstractmethod
    def train(self, *args) -> None:
        pass


    @abstractmethod
    def save(self) -> bool:
        pass


    @abstractmethod
    def evaluate(self, display: bool = True) -> None:
        pass
