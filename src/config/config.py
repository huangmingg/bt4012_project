from preprocess.preprocess import CreditCardDataset, SwarmDataset, AdultDataset
from sampling.baseline import BaselineAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.adasyn import Adasyn
from sampling.smote import SmoteAlgorithm
from sampling.mcd_adasyn import McdAdasynAlgorithm
from sampling.mcd_smote import McdSmoteAlgorithm
from sampling.sampling import SamplingAlgorithm
from model.xgb_model import XGBWrapper
from model.lg_model import LGWrapper


DATASETS = [
    (CreditCardDataset, 'creditcard_truncate.csv'),
    (SwarmDataset, 'Swarm_Behaviour.csv')
    (AdultDataset, 'adult.csv')
    ]

ALGORITHMS: SamplingAlgorithm = [
    RobRoseAlgorithm,
    Adasyn,
    SmoteAlgorithm
    McdAdasynAlgorithm,
    McdSmoteAlgorithm
    ]

MODELS = [
    XGBWrapper,
    LGWrapper
    ]