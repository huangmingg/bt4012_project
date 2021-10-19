from preprocess.preprocess import CreditCardDataset
from sampling.baseline import BaselineAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.sampling import SamplingAlgorithm
from model.xgb_model import XGBWrapper


DATASETS = [
    (CreditCardDataset, 'creditcard_truncate.csv')
    ]


ALGORITHMS: SamplingAlgorithm = [
    RobRoseAlgorithm
    ]


MODELS = [
    XGBWrapper
    ]