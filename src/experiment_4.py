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
from model.decision_tree_model import DecisionTreeWrapper
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
        DecisionTreeWrapper,
    ]

    DATASETS = [
        (SwarmDataset, "Swarm_Behaviour.csv", {"random_state": 4012,
                                     "n_repeats": 1,
                                     "imbal_level": None,
                                     "n_splits": 5}
         ),
        (SwarmDataset, "Swarm_Behaviour.csv", {"random_state": 4012,
                                     "imbal_level": 0.1,
                                     "n_repeats": 1,
                                     "n_splits": 5}
         ),
        (SwarmDataset, "Swarm_Behaviour.csv", {"random_state": 4012,
                                     "imbal_level": 0.05,
                                     "n_repeats": 1,
                                     "n_splits": 5}
         ),
        (SwarmDataset, "Swarm_Behaviour.csv", {"random_state": 4012,
                                     "imbal_level": 0.01,
                                     "n_repeats": 1,
                                     "n_splits": 5}
         ),
    ]

    ALGORITHMS: SamplingAlgorithm = [
        (BaselineAlgorithm, {}),
        (SmoteAlgorithm, {"random_state": 4012, "oversampling_level": [0.7, 0.8, 0.9]}),
        (AdasynAlgorithm, {"random_state": 4012, "oversampling_level": [0.7, 0.8, 0.9]}),
    ]

    for m in MODELS:
        for d, fp, p in DATASETS:
            d = d(fp)
            d.preprocess(**p)
            if p["imbal_level"]:
                print("Imbalanced Level:" + str(p["imbal_level"]))
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
        
