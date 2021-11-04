from abc import ABC, abstractmethod
from preprocess.preprocess import DatasetWrapper
from typing import Tuple, List
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class ClassifierWrapper(ABC):

    def __init__(self, data: DatasetWrapper) -> None:
        self.data = data
        self.result = {}

    @abstractmethod
    def evaluate(self) -> None:
        for l in self.data.b.keys():
            self.result[l] = []
            for idx, (x, y) in enumerate(self.data.b[l]):
                self.model.fit(x, y)
                y_score = self.model.predict_proba(self.data.x_test[idx])
                y_pred = y_score[:,1]
                rocauc = roc_auc_score(self.data.y_test[idx], y_pred)
                auprc = average_precision_score(self.data.y_test[idx], y_pred)
                self.result[l].append((rocauc, auprc))


    def compute_results(self) -> List[Tuple[str, Tuple[np.float, np.float, np.float, np.float]]]:
        output = []
        for j in self.result.keys():
            rocauc = np.array(list(map(lambda x: x[0], self.result[j])))
            auprc = np.array(list(map(lambda x: x[1], self.result[j])))
            output.append((str(j), (round(np.mean(rocauc), 3), round(np.std(rocauc), 3), 
            round(np.mean(auprc), 3), round(np.std(auprc), 3))))
        return output
