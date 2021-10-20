from preprocess.preprocess import CreditCardDataset
from sampling.baseline import BaselineAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.smote import SmoteAlgorithm
from sampling.sampling import SamplingAlgorithm
from model.xgb_model import XGBWrapper
from model.lg_model import LGWrapper


DATASETS = [
    (CreditCardDataset, 'creditcard_truncate.csv')
    ]

ALGORITHMS: SamplingAlgorithm = [
    SmoteAlgorithm
    ]

MODELS = [
    XGBWrapper,
    LGWrapper
    ]