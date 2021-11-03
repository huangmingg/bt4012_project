import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocess.preprocess import AdultDataset
from sampling.baseline import BaselineAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.smote import SmotencAlgorithm
from sampling.adasyn import AdasynNCAlgorithm
from model.lg_model import LGWrapper
from model.xgb_model import XGBWrapper

def experiment_2():
    """
    Experiment on Adult dataset to compare algorithm performances on continuous+nominal dataset
    Resampling is done with both categorical and numerical variables, before one-hot encoding and model fitting.
    """    
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
        pos_index = np.where(lg_model.model.classes_ == '>50K')[0][0]
        y_score = lg_model.model.predict_proba(x_test_comb)[:,pos_index]
        print('Logistic Regression: ', roc_auc_score(dataset.y_test, y_score), average_precision_score(dataset.y_test, y_score, pos_label='>50K') )

        xgb_model = XGBWrapper(dataset)
        xgb_model.model.fit(dataset.bxt, dataset.yxt)
        pos_index = np.where(xgb_model.model.classes_ == '>50K')[0][0]
        y_score = xgb_model.model.predict_proba(x_test_comb)[:,pos_index]
        print('XGBOOST', roc_auc_score(dataset.y_test, y_score), average_precision_score(dataset.y_test, y_score, pos_label='>50K'))

if __name__ == '__main__':
    experiment_2()
