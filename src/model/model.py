from abc import ABC, abstractmethod
from preprocess.preprocess import DatasetWrapper
from typing import Tuple
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class ClassifierWrapper(ABC):

    def __init__(self, data: DatasetWrapper) -> None:
        self.data = data
        self.result = []

    @abstractmethod
    def evaluate(self) -> None:
        for i in range(len(self.data.bxt)):
            self.model.fit(self.data.bxt[i], self.data.byt[i])
            y_score = self.model.predict_proba(self.data.x_test[i])
            y_pred = y_score[:,1]
            rocauc = roc_auc_score(self.data.y_test[i], y_pred)
            auprc = average_precision_score(self.data.y_test[i], y_pred)
            self.result.append({'rocauc': rocauc, 'auprc': auprc})


    def compute_results(self) -> Tuple[np.float, np.float, np.float, np.float]:
        rocauc = np.array(list(map(lambda x: x.get('rocauc'), self.result)))
        auprc = np.array(list(map(lambda x: x.get('auprc'), self.result)))
        return round(np.mean(rocauc), 3), round(np.std(rocauc), 3), round(np.mean(auprc), 3), round(np.std(auprc), 3)
