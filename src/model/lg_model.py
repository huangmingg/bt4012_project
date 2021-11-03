from model.model import ClassifierWrapper, DatasetWrapper
from sklearn.linear_model import LogisticRegression
# import seaborn as sns 
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class LGWrapper(ClassifierWrapper):


    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = LogisticRegression(random_state=4012, max_iter=10000)

    def train(self, *args):
        self.model.fit(self.data.bxt, self.data.byt)


    def save(self):
        pass


    def evaluate(self):
        y_pred = self.model.predict(self.data.x_test)
        conmatrix = confusion_matrix(self.data.y_test, y_pred)
        print(conmatrix)
        print(classification_report(self.data.y_test, y_pred))
