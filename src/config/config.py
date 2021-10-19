from preprocess.preprocess import CreditCardDataset
from sampling.baseline import BaselineAlgorithm 
from sampling.sampling import SmoteAlgorithm
from model.xgb_model import XGBWrapper
from model.lg_model import LGWrapper

DATASETS = [
    (CreditCardDataset, 'creditcard.csv')
    ]


ALGORITHMS = [
    BaselineAlgorithm,
    SmoteAlgorithm
    ]


MODELS = [
    XGBWrapper,
    LGWrapper
    ]