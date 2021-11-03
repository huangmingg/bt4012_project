from sampling.sampling import SamplingAlgorithm

from model.model import ClassifierWrapper
from preprocess.preprocess import CreditCardDataset, DatasetWrapper
from config.config import ALGORITHMS, DATASETS, MODELS
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os


def evaluate(d: DatasetWrapper, a: SamplingAlgorithm, m: ClassifierWrapper) -> None:
    d.balance(a)
    model = m(d)
    model.train()


def showResults(y_test, y_pred):
    conmatrix = confusion_matrix(y_test, y_pred)
    print(conmatrix)
    print(classification_report(y_test, y_pred))


def main():
    datasets = [wrapper(filename) for wrapper, filename in DATASETS]
    # lg_model = LogisticRegression(random_state=4012, max_iter=100000)
    # xgb_model = XGBClassifier(use_label_encoder=True, random_state=4012)

    for m in MODELS:
        for d in datasets:
            for a in ALGORITHMS:
                d.balance(a)
                model = m(d)
                model.train()
                model.evaluate()
            

           


if __name__ == '__main__':
    main()