from model.model import ClassifierWrapper, DatasetWrapper
from xgboost import XGBClassifier
import seaborn as sns 
import matplotlib.pyplot as plt


class XGBWrapper(ClassifierWrapper):


    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = XGBClassifier(use_label_encoder=False, random_state=4012)


    def train(self):
        self.model.fit(self.data.x_train, self.data.y_train, eval_metric=['auc'])
        print('Successfully fitted XGB model')

    def save(self, src):
        pass