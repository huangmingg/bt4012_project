import os
import pandas as pd
from model.decision_tree_model import DecisionTreeWrapper
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
    OVERSAMPLING_LEVEL = [0.01, 0.05, 0.1, 0.5, 1]
    RANDOM_STATE = 4012

    MODELS: ClassifierWrapper = [
        LGWrapper,
        DecisionTreeWrapper,
    ]

    DATASETS = [
        (CreditCardDataset, "creditcard.csv", {"random_state": RANDOM_STATE, "n_repeats": 2, "n_splits": 5}),
    ]

    ALGORITHMS: SamplingAlgorithm = [
        (BaselineAlgorithm, {"random_state": RANDOM_STATE, "oversampling_level": OVERSAMPLING_LEVEL}),
        (SmoteAlgorithm, {"random_state": RANDOM_STATE, "oversampling_level": OVERSAMPLING_LEVEL}),
        (AdasynAlgorithm, {"random_state": RANDOM_STATE, "oversampling_level": OVERSAMPLING_LEVEL}),
        (RobRoseAlgorithm, {"random_state": RANDOM_STATE, "oversampling_level": OVERSAMPLING_LEVEL, "alpha": 0.95, "const": 1}),
        (McdAdasynAlgorithm, {"random_state": RANDOM_STATE, "oversampling_level": OVERSAMPLING_LEVEL, "sp": 0.95}),
        (McdSmoteAlgorithm, {"random_state": RANDOM_STATE, "oversampling_level": OVERSAMPLING_LEVEL, "sp": 0.95, "p": 0.999}),
    ]

    results = []

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
                    results.append(
                        {
                            "algo": a[0].__name__,
                            "model": type(model).__name__,
                            "dataset": type(d).__name__,
                            "sampling_ratio": {l},
                            "rocauc_mean": val[0],
                            "rocauc_std": val[1],
                            "auprc_mean": val[2],
                            "auprc_std": val[3]     
                        }
                    )
                    print(f"At oversampling ratio {l}, ROCAUC Mean: {val[0]}, ROCAUC Std: {val[1]}, AUPRC Mean: {val[2]}, AUPRC Std: {val[3]}")


if __name__ == '__main__':
    main()