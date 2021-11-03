from model.model import ClassifierWrapper, DatasetWrapper
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score


class XGBWrapper(ClassifierWrapper):

    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = XGBClassifier(use_label_encoder=False, random_state=4012)

    def evaluate(self):
        super().evaluate()