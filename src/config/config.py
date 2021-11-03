from model.model import ClassifierWrapper
from model.xgb_model import XGBWrapper
from model.lg_model import LGWrapper
from preprocess.preprocess import CreditCardDataset, AdultDataset, DatasetWrapper
from sampling.baseline import BaselineAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.smote import SmoteAlgorithm
from sampling.mcd_adasyn import McdAdasynAlgorithm
from sampling.mcd_smote import McdSmoteAlgorithm
from sampling.sampling import SamplingAlgorithm


DATASETS: DatasetWrapper = [
    (CreditCardDataset, 'creditcard.csv'),
    (AdultDataset, 'adult.csv')
    ]

ALGORITHMS: SamplingAlgorithm = [
    McdAdasynAlgorithm,
    McdSmoteAlgorithm
    ]

MODELS: ClassifierWrapper = [
    XGBWrapper,
    LGWrapper
    ]
