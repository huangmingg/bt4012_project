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
    """
    Evaluates SMOTENC, ADASYNNC and robROSE on Adult Dataset (at various imbalance levels). Different oversampling levels are used to assess the algorithm's effectiveness.
    """
    MODELS: ClassifierWrapper = [
        LGWrapper,
        DecisionTreeWrapper,
    ]

    DATASETS_1 = [(AdultDataset, "adult.csv", {"random_state": 4012, "n_repeats": 2, "n_splits": 5})] # Original
    DATASETS_2 = [(AdultDataset, "adult.csv", {"random_state": 4012, "imbal_level": 0.05, "n_repeats": 2, "n_splits": 5})] # Imbalance level 5%
    DATASETS_3 = [(AdultDataset, "adult.csv", {"random_state": 4012, "imbal_level": 0.01, "n_repeats": 2, "n_splits": 5})] # Imbalance level 1%


    # For original imbalance level
    ALGORITHMS_1: SamplingAlgorithm = [
        (BaselineAlgorithm, {}),
        (SmotencAlgorithm, {"random_state": 4012,
                            "oversampling_level": [1.0]}),
        (AdasynNCAlgorithm, {"random_state": 4012,
                             "oversampling_level": [1.0] }),
        (RobRoseAlgorithm, {"random_state": 4012, "oversampling_level": [1.0], "alpha": 0.95, "const": 1}),
    ]

    # For imbalance level of 5%
    ALGORITHMS_2: SamplingAlgorithm = [
        (BaselineAlgorithm, {}),
        (SmotencAlgorithm, {"random_state": 4012,
                            "oversampling_level": [1/9, 3/7, 1.0]}),
        (AdasynNCAlgorithm, {"random_state": 4012,
                             "oversampling_level": [1/9, 3/7, 1.0] }),
        (RobRoseAlgorithm, {"random_state": 4012, "oversampling_level": [1/9, 3/7, 1.0], "alpha": 0.95, "const": 1}),
    ]

    # For imbalance level of 1%
    ALGORITHMS_3: SamplingAlgorithm = [
        (BaselineAlgorithm, {}),
        (SmotencAlgorithm, {"random_state": 4012,
                            "oversampling_level": [1/19, 1/9, 3/7, 1.0]}),
        (AdasynNCAlgorithm, {"random_state": 4012,
                             "oversampling_level": [1/19, 1/9, 3/7, 1.0] }),
        (RobRoseAlgorithm, {"random_state": 4012, "oversampling_level": [1/19, 1/9, 3/7, 1.0], "alpha": 0.95, "const": 1}),
    ]   

    EXPERIMENTS = zip(['Original', '5%', '1%'], [DATASETS_1, DATASETS_2, DATASETS_3], [ALGORITHMS_1, ALGORITHMS_2, ALGORITHMS_3])

    for label, datasets, algorithms in EXPERIMENTS:
        print(f'Experimenting with {label} imbalance level:')
        experiment_2(MODELS, datasets, algorithms)



def experiment_2(MODELS, DATASETS, ALGORITHMS):
    for m in MODELS:
        for d, fp, p in DATASETS:
            d = d(fp)
            d.preprocess(**p)

            fold_ohes = []       
            for idx, x_fold in enumerate(d.x_train):
                ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                # Fit on train fold (do not transform since oversampling is done with categorical directly)
                # Safe to fit on train fold before oversampling since oversampling will not introduce new categories
                ohe.fit(x_fold[:, d.cat_columns_ind]) 
                x_test = d.x_test[idx]
                x_test_ohe = ohe.transform(x_test[:, d.cat_columns_ind]) # Transform test fold
                new_x_test = np.hstack([x_test[:, d.num_columns_ind], x_test_ohe])
                d.x_test[idx] = new_x_test
                fold_ohes.append(ohe)  # Store OHE used for each fold to encode balanced dataset after oversampling


            for a in ALGORITHMS:
                d.balance(a[0], categorical_features=d.cat_columns_ind, **a[1])
                
                # One Hot Encode the balanced datasets prior to evaluation (only for Experiment 2)
                for level in d.b.keys(): # For each oversampling level
                    for idx, (x, y) in enumerate(d.b[level]):  # For each fold
                        
                        # OHE balanced dataset using the OHE fit on that fold's train data
                        bxt_ohe = fold_ohes[idx].transform(x[:, d.cat_columns_ind])
                        new_bxt = np.hstack([x[:, d.num_columns_ind], bxt_ohe])
                        d.b[level][idx] = new_bxt, y

                model = m(d)
                print(f"Evaluating {type(model).__name__} model for algorithm {a[0].__name__} using dataset {type(d).__name__}")
                model.evaluate()
                res = model.compute_results()
                for l, val in res:
                    print(f"At oversampling ratio {l}, ROCAUC Mean: {val[0]}, ROCAUC Std: {val[1]}, AUPRC Mean: {val[2]}, AUPRC Std: {val[3]}")

if __name__ == '__main__':
    main()

