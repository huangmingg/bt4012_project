from model.model import ClassifierWrapper, DatasetWrapper
from xgboost import XGBClassifier
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np


class XGBWrapper(ClassifierWrapper):

    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = XGBClassifier(use_label_encoder=True, random_state=4012)

    def train(self) -> None:
        self.model.fit(self.data.bxt, self.data.yxt, eval_metric=['auc'])
        ans = self.model.predict(self.data.x_test)
        print(ans)
        print('Successfully fitted model')

    def save(self, src):
        pass
