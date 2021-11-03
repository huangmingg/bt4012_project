import os
from config.config import ALGORITHMS
from model.lg_model import LGWrapper
from model.model import ClassifierWrapper
from preprocess.preprocess import CreditCardDataset
from sampling.adasyn import AdasynAlgorithm
from sampling.baseline import BaselineAlgorithm
from sampling.mcd_adasyn import McdAdasynAlgorithm
from sampling.mcd_smote import McdSmoteAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.sampling import SamplingAlgorithm
from sampling.smote import SmoteAlgorithm


def main():
    DATASETS = [
        (CreditCardDataset, 'creditcard.csv'),
        # (SwarmDataset, 'Swarm_Behaviour.csv'),
        # (AdultDataset, 'adult.csv')
    ]

    MODELS: ClassifierWrapper = [
        # XGBWrapper,
        LGWrapper
    ]

    ALGORITHMS: SamplingAlgorithm = [
        (BaselineAlgorithm, {}),
        (SmoteAlgorithm, {}),
        (AdasynAlgorithm, {}),
        (RobRoseAlgorithm, {"r": 0.5, "alpha": 0.95, "const": 1, "seed": 4012}),
        (McdAdasynAlgorithm, {"random_state": 4012, "sp": 0.95}),
        (McdSmoteAlgorithm, {"random_state": 4012, "sp": 0.95, "p": 0.999}),
    ]

    datasets = [wrapper(filename) for wrapper, filename in DATASETS]

    # to_evaluate = [
    #     ('Imbalanced', bx_imbal, by_imbal),
    #     ('SMOTENC', bx_smotenc, by_smotenc),
    #     ('ADASYNNC', bx_adasync, by_adasync),
    #     ('ROBROSE', bx_robrose, by_robrose),
    # ]

    for m in MODELS:
        for d in datasets:
            for a in ALGORITHMS:
                d.balance(a[0], **a[1])
                model = m(d)
                print(f"Evaluating {type(model).__name__} model for algorithm {a.__name__} using dataset {type(d).__name__}")
                model.train()
                model.evaluate()
            

if __name__ == '__main__':
    main()