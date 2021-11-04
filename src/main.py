import os
import pandas as pd
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
    MODELS: ClassifierWrapper = [
        LGWrapper,
        XGBWrapper,
    ]

    DATASETS = [
        (CreditCardDataset, "creditcard.csv", {"random_state": 4012, "n_repeats": 2, "n_splits": 5}),
    ]

    ALGORITHMS: SamplingAlgorithm = [
        (BaselineAlgorithm, {"random_state": 4012, "oversampling_level": [0.05, 0.1, 0.2, 0.5]}),
        (SmoteAlgorithm, {"random_state": 4012, "oversampling_level": [0.05, 0.1, 0.2, 0.5]}),
        (AdasynAlgorithm, {"random_state": 4012, "oversampling_level": [0.05, 0.1, 0.2, 0.5]}),
        (RobRoseAlgorithm, {"random_state": 4012, "oversampling_level": [0.05, 0.1, 0.2, 0.5], "alpha": 0.95, "const": 1}),
    ]

    for m in MODELS:
        for d, fp, p in DATASETS:
            d = d(fp)
            d.preprocess(**p)
            for a in ALGORITHMS:
                d.balance(a[0], **a[1])
                model = m(d)
                print(f"Evaluating {type(model).__name__} model for algorithm {a[0].__name__} using dataset {type(d).__name__}")
                model.evaluate()
                res = model.compute_results()
                for l, val in res:
                    print(f"At oversampling ratio {l}, ROCAUC Mean: {val[0]}, ROCAUC Std: {val[1]}, AUPRC Mean: {val[2]}, AUPRC Std: {val[3]}")
            

if __name__ == '__main__':
    main()