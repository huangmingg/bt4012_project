import os

from sklearn.metrics import roc_auc_score, average_precision_score

from sampling.sampling import SamplingAlgorithm
from model.model import ClassifierWrapper
from preprocess.preprocess import DatasetWrapper
from config.config import ALGORITHMS, DATASETS, MODELS


def evaluate(d: DatasetWrapper, a: SamplingAlgorithm, m: ClassifierWrapper) -> None:
    d.balance(a)
    model = m(d)
    model.train()

def experiment_2():
    """
    Experiment on Adult dataset to compare algorithm performances on continuous+nominal dataset
    Resampling is done with both categorical and numerical variables, before one-hot encoding and model fitting.
    """    
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from preprocess.preprocess import AdultDataset
    from sampling.baseline import BaselineAlgorithm
    from sampling.robrose import RobRoseAlgorithm
    from sampling.smote import SmotencAlgorithm
    from sampling.adasyn import AdasynNCAlgorithm
    from model.lg_model import LGWrapper
    from model.xgb_model import XGBWrapper


    dataset = AdultDataset('adult.csv')
    bx_imbal, by_imbal = BaselineAlgorithm.run(dataset.x_train, dataset.y_train)
    bx_smotenc, by_smotenc= SmotencAlgorithm.run(dataset.x_train, dataset.y_train, categorical_features=dataset.cat_columns_ind)
    bx_adasync, by_adasync= AdasynNCAlgorithm.run(dataset.x_train, dataset.y_train, categorical_features=dataset.cat_columns_ind)
    bx_robrose, by_robrose = RobRoseAlgorithm.run(dataset.x_train, dataset.y_train, label='income', columns=dataset.columns, r=0.5, alpha=0.95, const=1, seed=4012)

    to_evaluate = [
        ('Imbalanced', bx_imbal, by_imbal),
        ('SMOTENC', bx_smotenc, by_smotenc),
        ('ADASYNNC', bx_adasync, by_adasync),
        ('ROBROSE', bx_robrose, by_robrose),
    ]

    for algo, bx, by in to_evaluate:
        print(algo)
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        x_train_ohe = ohe.fit_transform(bx[:, dataset.cat_columns_ind]) 
        x_test_ohe = ohe.transform(dataset.x_test[:, dataset.cat_columns_ind])

        dataset.bxt = np.hstack([bx[:, dataset.num_columns_ind], x_train_ohe])
        dataset.yxt = by

        x_test_comb = np.hstack([dataset.x_test[:,dataset.num_columns_ind], x_test_ohe])

        lg_model = LGWrapper(dataset)
        lg_model.model.fit(dataset.bxt, dataset.yxt)
        y_score = lg_model.model.predict_proba(x_test_comb)[:,1]
        print('Logistic Regression: ', roc_auc_score(dataset.y_test, y_score), average_precision_score(dataset.y_test, y_score, pos_label='>50K') )

        xgb_model = XGBWrapper(dataset)
        xgb_model.model.fit(dataset.bxt, dataset.yxt)
        y_score = xgb_model.model.predict_proba(x_test_comb)[:,1]
        print('XGBOOST', roc_auc_score(dataset.y_test, y_score), average_precision_score(dataset.y_test, y_score, pos_label='>50K'))

def experiment_3():
    from sampling.sampling import SamplingAlgorithm
    from preprocess.preprocess import CreditCardDataset, SwarmDataset
    from sampling.baseline import BaselineAlgorithm
    from sampling.robrose import RobRoseAlgorithm
    from sampling.smote import SmoteAlgorithm
    from sampling.adasyn import Adasyn
    from model.lg_model import LGWrapper
    from model.xgb_model import XGBWrapper
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    dataset = SwarmDataset("Swarm_Behaviour.csv")
    bx_imbal, by_imbal = BaselineAlgorithm.run(dataset.x_train, dataset.y_train)
    bx_smote, by_smote= SmoteAlgorithm.run(dataset.x_train, dataset.y_train )
    bx_adasyn, by_adasyn= Adasyn.run(dataset.x_train, dataset.y_train)
    
    to_evaluate = [
        ('Imbalanced', bx_imbal, by_imbal),
        ('SMOTE', bx_smote, by_smote),
        ('ADASYN', bx_adasyn, by_adasyn),
    ]

    for algo, bx, by in to_evaluate:
        print(algo)
        lg_model = LGWrapper(dataset)
        lg_model.model.fit(bx, by)
        y_score = lg_model.model.predict(dataset.x_test)
        print('Logistic Regression')
        print('Classification Report')
        print(classification_report(dataset.y_test, y_score))
        print("ROC AUC Score: ", roc_auc_score(dataset.y_test, y_score))

        xgb_model = XGBWrapper(dataset)
        xgb_model.model.fit(bx, by)
        y_score = xgb_model.model.predict(dataset.x_test)
        print('XGBOOST')
        print('Classification Report')
        print(classification_report(dataset.y_test, y_score))
        print("ROC AUC Score: ", roc_auc_score(dataset.y_test, y_score))
        
def main():
    datasets = [wrapper(filename) for wrapper, filename in DATASETS]
    for d in datasets:
        for a in ALGORITHMS:
            for m in MODELS:
                evaluate(d, a, m)

if __name__ == '__main__':
    # main()
    experiment_2()


