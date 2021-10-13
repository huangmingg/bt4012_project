from preprocess.preprocess import CreditCardDataset
from sampling.baseline import BaselineAlgorithm
from model.xgb_model import XGBWrapper

DATASETS = [
    (CreditCardDataset, 'creditcard.csv')
    ]


ALGORITHMS = [
    BaselineAlgorithm
    ]


MODELS = [
    XGBWrapper
    ]