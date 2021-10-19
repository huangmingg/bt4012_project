from abc import ABC, abstractmethod
from preprocess.preprocess import DatasetWrapper
from typing import Tuple
import pickle
import numpy as np


class ClassifierWrapper(ABC):

    def __init__(self, data: DatasetWrapper) -> None:
        self.data = data

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def save(self, src) -> bool:
        pass

    def evaluate(self, display: bool = True) -> None:
        pass
