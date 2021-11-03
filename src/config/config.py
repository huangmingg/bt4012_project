from preprocess.preprocess import CreditCardDataset, AdultDataset
from sampling.baseline import BaselineAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.smote import SmoteAlgorithm
from sampling.mcd_adasyn import McdAdasyn
from sampling.mcd_smote import McdSmote
from sampling.sampling import SamplingAlgorithm
from model.xgb_model import XGBWrapper
from model.lg_model import LGWrapper


DATASETS = [
    (CreditCardDataset, 'creditcard_truncate.csv'),
    (AdultDataset, 'adult.csv')
    ]

ALGORITHMS: SamplingAlgorithm = [
    SmoteAlgorithm
    ]

MODELS = [
    XGBWrapper,
    LGWrapper
    ]