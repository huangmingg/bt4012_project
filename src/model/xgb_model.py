from model.model import ClassifierWrapper, DatasetWrapper
from xgboost import XGBClassifier
# import seaborn as sns 
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class XGBWrapper(ClassifierWrapper):

    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = XGBClassifier(use_label_encoder=False, random_state=4012)

    def train(self, *args) -> None:
        self.model.fit(self.data.bxt, self.data.byt, eval_metric=['auc'])


    def save(self, src):
        pass

    def evaluate(self):
        y_pred = self.model.predict(self.data.x_test)
        conmatrix = confusion_matrix(self.data.y_test, y_pred)
        print(conmatrix)
        print(classification_report(self.data.y_test, y_pred))
