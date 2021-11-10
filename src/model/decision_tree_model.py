from sklearn.tree import DecisionTreeClassifier
from model.model import ClassifierWrapper
from preprocess.preprocess import DatasetWrapper


class DecisionTreeWrapper(ClassifierWrapper):

    def __init__(self, data: DatasetWrapper, *args) -> None:
        super().__init__(data)
        self.model = DecisionTreeClassifier(random_state=4012, max_depth=8)

    def evaluate(self):
        super().evaluate()