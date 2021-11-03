from model.model import ClassifierWrapper, DatasetWrapper
from sklearn.linear_model import LogisticRegression
# import seaborn as sns 
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score


class LGWrapper(ClassifierWrapper):


    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = LogisticRegression(random_state=4012, max_iter=10000)

    def train(self, *args):
        self.model.fit(self.data.bxt, self.data.byt)


    def save(self):
        pass


    def evaluate(self):
        y_score = self.model.predict(self.data.x_test)
        # y_score = self.model.predict_proba(self.data.x_test)
        # y_score = y_score[:,1]
        print('Logistic Regression: ', roc_auc_score(self.data.y_test, y_score), average_precision_score(self.data.y_test, y_score) )
