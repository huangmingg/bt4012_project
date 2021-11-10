from model.model import ClassifierWrapper, DatasetWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

class LGWrapper(ClassifierWrapper):

    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = LogisticRegression(random_state=4012, max_iter=10000)

    def evaluate(self):
        super().evaluate()