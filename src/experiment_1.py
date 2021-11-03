import os
from config.config import ALGORITHMS
from model.lg_model import LGWrapper
from model.xgb_model import XGBWrapper
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
    ]

    MODELS: ClassifierWrapper = [
        XGBWrapper,
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
    for m in MODELS:
        for d in datasets:
            for a in ALGORITHMS:
                d.balance(a[0], **a[1])
                model = m(d)
                print(f"Evaluating {type(model).__name__} model for algorithm {a[0].__name__} using dataset {type(d).__name__}")
                model.evaluate()
                rocauc_mean, rocauc_std, auprc_mean, auprc_std = model.compute_results()
                print(f"ROCAUC Mean: {rocauc_mean}, ROCAUC Std: {rocauc_std}, AUPRC Mean: {auprc_mean}, AUPRC Std: {auprc_std}")
            

if __name__ == '__main__':
    main()