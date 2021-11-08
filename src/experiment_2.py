import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


from preprocess.preprocess import AdultDataset
from sampling.sampling import SamplingAlgorithm
from sampling.baseline import BaselineAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.smote import SmotencAlgorithm
from sampling.adasyn import AdasynNCAlgorithm
from model.model import ClassifierWrapper
from model.lg_model import LGWrapper
from model.decision_tree_model import DecisionTreeWrapper


def main():
    MODELS: ClassifierWrapper = [
        LGWrapper,
        DecisionTreeWrapper,
    ]

    DATASETS = [
        (AdultDataset, "adult.csv", {"random_state": 4012,
                                     "n_repeats": 1,
                                     "n_splits": 5}
         ),
        (AdultDataset, "adult.csv", {"random_state": 4012,
                                     "imbal_level": 0.05,
                                     "n_repeats": 1,
                                     "n_splits": 5}
         ),
        (AdultDataset, "adult.csv", {"random_state": 4012,
                                     "imbal_level": 0.01,
                                     "n_repeats": 1,
                                     "n_splits": 5}
         ),
    ]

    ALGORITHMS: SamplingAlgorithm = [
        # Baseline, Smotenc, Adasynnc interprets as ratio of minority:majority in resampled data
        (BaselineAlgorithm, {}),
        (SmotencAlgorithm, {"random_state": 4012,
                            "oversampling_level": [1.0]}),
        (AdasynNCAlgorithm, {"random_state": 4012,
                             "oversampling_level": [1.0]}),
        # robROSE interprets as minority proportion in resample data
        (RobRoseAlgorithm, {"random_state": 4012, "oversampling_level": [
         0.5], "alpha": 0.95, "const": 1}),
    ]

    for m in MODELS:
        for d, fp, p in DATASETS:
            d = d(fp)
            d.preprocess(**p)
            for a in ALGORITHMS:
                d.balance(a[0], categorical_features=d.cat_columns_ind, **a[1])

                x_test_orig = copy.deepcopy(d.x_test)

                # One Hot Encode the balanced datasets prior to evaluation (only for Experiment 2)
                for level in d.b.keys():  # For each oversampling level
                    for idx, (x, y) in enumerate(d.b[level]):  # For each fold
                        # Fit-transform OHE on balanced train fold, then transform test fold
                        ohe = OneHotEncoder(
                            handle_unknown='ignore', sparse=False)

                        bxt_ohe = ohe.fit_transform(x[:, d.cat_columns_ind])
                        new_bxt = np.hstack([x[:, d.num_columns_ind], bxt_ohe])
                        d.b[level][idx] = new_bxt, y

                        x_test = d.x_test[idx]
                        x_test_ohe = ohe.transform(
                            x_test[:, d.cat_columns_ind])
                        new_x_test = np.hstack(
                            [x_test[:, d.num_columns_ind], x_test_ohe])
                        d.x_test[idx] = new_x_test

                model = m(d)
                print(f"Evaluating {type(model).__name__} model for algorithm {a[0].__name__} using dataset {type(d).__name__}")
                model.evaluate()
                res = model.compute_results()
                for l, val in res:
                    print(f"At oversampling ratio {l}, ROCAUC Mean: {val[0]}, ROCAUC Std: {val[1]}, AUPRC Mean: {val[2]}, AUPRC Std: {val[3]}")

                # Reset x_test
                d.x_test = x_test_orig


if __name__ == '__main__':
    main()

