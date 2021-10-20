from model.model import ClassifierWrapper, DatasetWrapper
from sklearn import linear_model, model_selection, metrics
import seaborn as sns 
import matplotlib.pyplot as plt


class LGWrapper(ClassifierWrapper):


    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = linear_model.LogisticRegression(random_state=4012, max_iter=10000)


    def train(self):
        self.model.fit(self.data.x_train, self.data.y_train)
        print('Successfully fitted LG model')

    def save(self, src):
        pass