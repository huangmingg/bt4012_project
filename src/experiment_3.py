import os

from sklearn.metrics import roc_auc_score, average_precision_score
from preprocess.preprocess import CreditCardDataset, SwarmDataset

from model.model import ClassifierWrapper
from preprocess.preprocess import DatasetWrapper
from config.config import ALGORITHMS, DATASETS, MODELS

from sampling.sampling import SamplingAlgorithm
from sampling.baseline import BaselineAlgorithm
from sampling.robrose import RobRoseAlgorithm
from sampling.smote import SmoteAlgorithm
from sampling.adasyn import Adasyn
from model.lg_model import LGWrapper
from model.xgb_model import XGBWrapper
from sklearn.metrics import roc_auc_score, average_precision_score

def dataInfo():
    dataset = SwarmDataset("Swarm_Behaviour.csv")
    print("High Dimensional Dataset")
    print("Number of Positives: ", len(dataset.raw_df[dataset.raw_df['Swarm_Behaviour']==1]))
    print("Number of Negatives: ", len(dataset.raw_df[dataset.raw_df['Swarm_Behaviour']==0]))
    dataset = CreditCardDataset("creditcard.csv")
    print("Low Dimensional Dataset")
    print("Number of Positives: ", len(dataset.raw_df[dataset.raw_df['Class']==1]))
    print("Number of Negatives: ", len(dataset.raw_df[dataset.raw_df['Class']==0]))

def experiment_3():
    print("Using High Dimensional Data")
    dataset = SwarmDataset("Swarm_Behaviour.csv")
    experiment_3_run(dataset)
    print("Using Low Dimensional Data")
    dataset = CreditCardDataset("creditcard.csv")
    experiment_3_run(dataset)
        
def experiment_3_run(dataset):

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
        print("ROC AUC Score: ", roc_auc_score(dataset.y_test, y_score))
        print("AUPRC Score: ", average_precision_score(dataset.y_test, y_score))

        xgb_model = XGBWrapper(dataset)
        xgb_model.model.fit(bx, by)
        y_score = xgb_model.model.predict(dataset.x_test)
        print('XGBOOST')
        print('Classification Report')
        print("ROC AUC Score: ", roc_auc_score(dataset.y_test, y_score))
        print("AUPRC Score: ", average_precision_score(dataset.y_test, y_score))
        
        
if __name__ == '__main__':
    experiment_3()        
        
