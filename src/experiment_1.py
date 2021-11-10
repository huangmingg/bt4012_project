import os
import pandas as pd
import timeit
from model.lg_model import LGWrapper
from model.xgb_model import XGBWrapper
from model.model import ClassifierWrapper
from preprocess.preprocess import CreditCardDataset
from sampling.adasyn import AdasynAlgorithm
from sampling.baseline import BaselineAlgorithm
from sampling.mcd_adasyn import McdAdasynAlgorithm
from sampling.mcd_smote import McdSmoteAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.smote_boost import SMOTEBoostWrapperAlgorithm
from sampling.sampling import SamplingAlgorithm
from sampling.smote import SmoteAlgorithm


def main():
    DATASETS = [
        (CreditCardDataset, "creditcard.csv", {"random_state": 4012, "n_repeats": 2, "n_splits": 5}),
    ]

    ALGORITHMS: SamplingAlgorithm = [
        # (BaselineAlgorithm, {"random_state": 4012, "oversampling_level": [1]}),
        (SMOTEBoostWrapperAlgorithm, {"random_state": 4012, "oversampling_level": [1]}),
        # (SmoteAlgorithm, {"random_state": 4012, "oversampling_level": [1]}),
        # (AdasynAlgorithm, {"random_state": 4012, "oversampling_level": [1]}),
        # (RobRoseAlgorithm, {"random_state": 4012, "oversampling_level": [0.05, 0.1, 0.2, 0.5], "alpha": 0.95, "const": 1}),
        # (McdAdasynAlgorithm, {"random_state": 4012, "oversampling_level": [0.05, 0.1, 0.2, 0.5], "sp": 0.95}),
        # (McdSmoteAlgorithm, {"random_state": 4012, "oversampling_level": [0.05, 0.1, 0.2, 0.5], "sp": 0.95, "p": 0.999}),
    ]

    NUM_RUNS = 1
    result = []

    for d, fp, p in DATASETS:
        d = d(fp)
        d.preprocess(**p)
        # for a in ALGORITHMS:
        #     duration = timeit.Timer(lambda: d.balance(a[0], **a[1])).timeit(number=NUM_RUNS)
        #     avg_duration = duration / NUM_RUNS
        #     result.append({'algo': a[0].__name__, 'duration': avg_duration})
        #     print(f'On average it took {avg_duration} seconds')
                
            

if __name__ == '__main__':
    main()