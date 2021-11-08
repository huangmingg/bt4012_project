import os

from sklearn.metrics import roc_auc_score, average_precision_score
from preprocess.preprocess import CreditCardDataset, SwarmDataset

from model.model import ClassifierWrapper
from preprocess.preprocess import DatasetWrapper
from sampling.sampling import SamplingAlgorithm
from sampling.baseline import BaselineAlgorithm
from sampling.smote import SmoteAlgorithm
from sampling.adasyn import AdasynAlgorithm
from model.lg_model import LGWrapper
from model.xgb_model import XGBWrapper
from sklearn.metrics import roc_auc_score, average_precision_score

def dataInfo():
    dataset = SwarmDataset("Swarm_Behaviour.csv")
    print("High Dimensional Dataset")
    print("Number of Positives: ", len(dataset.raw_df[dataset.raw_df['Swarm_Behaviour']==1]))
    print("Number of Negatives: ", len(dataset.raw_df[dataset.raw_df['Swarm_Behaviour']==0]))
    print("Proportion of Postives over the whole dataset: ", len(dataset.raw_df[dataset.raw_df['Swarm_Behaviour']==1])/len(dataset.raw_df))
    print("Proportion of Postives over the majority: ", len(dataset.raw_df[dataset.raw_df['Swarm_Behaviour']==1])/len(dataset.raw_df[dataset.raw_df['Swarm_Behaviour']==0]))
    dataset = CreditCardDataset("creditcard.csv")
    print("Low Dimensional Dataset")
    print("Number of Positives: ", len(dataset.raw_df[dataset.raw_df['Class']==1]))
    print("Number of Negatives: ", len(dataset.raw_df[dataset.raw_df['Class']==0]))
    print("Proportion of Postives over the whole dataset: ", len(dataset.raw_df[dataset.raw_df['Class']==1])/len(dataset.raw_df))
    print("Proportion of Postives over the majority: ", len(dataset.raw_df[dataset.raw_df['Class']==1])/len(dataset.raw_df[dataset.raw_df['Class']==0]))
        
def experiment_3():
    
    MODELS: ClassifierWrapper = [
        LGWrapper,
        XGBWrapper,
    ]

    DATASETS = [
        (SwarmDataset, "Swarm_Behaviour.csv", {"random_state": 4012, "n_repeats": 2, "n_splits": 5}),
    ]

    ALGORITHMS: SamplingAlgorithm = [
        (BaselineAlgorithm, {"random_state": 4012, "oversampling_level": [0.7,0.75,0.8,0.85]}),
        (SmoteAlgorithm, {"random_state": 4012, "oversampling_level": [0.7,0.75,0.8,0.85]}),
        (AdasynAlgorithm, {"random_state": 4012, "oversampling_level": [0.7,0.75,0.8,0.85]}),
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
    experiment_3()        
        
