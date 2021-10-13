from model.model import ClassifierWrapper, DatasetWrapper
from xgboost import XGBClassifier
import seaborn as sns 
import matplotlib.pyplot as plt


class XGBWrapper(ClassifierWrapper):


    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = XGBClassifier(objective='binary:logistic', random_state=4012)


    def train(self):
        self.model.fit(self.data.x_train, self.data.y_train, 
            eval_set=[(self.data.x_train, self.data.y_train), (self.data.x_test, self.data.y_test)], eval_metric=['merror', 'mlogloss'], verbose=True)
            

    def save(self, src):
        pass