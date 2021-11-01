from sampling.sampling import SamplingAlgorithm
from model.model import ClassifierWrapper
from preprocess.preprocess import DatasetWrapper
from config.config import ALGORITHMS, DATASETS, MODELS
from sklearn import linear_model, model_selection, metrics
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os


def evaluate(d: DatasetWrapper, a: SamplingAlgorithm, m: ClassifierWrapper) -> None:
    d.balance(a)
    model = m(d)
    model.train()

def showResults(y_test, y_pred):
    conmatrix = confusion_matrix(y_test, y_pred)
    print(conmatrix)
    print(classification_report(y_test, y_pred))

def main():
    datasets = [wrapper(filename) for wrapper, filename in DATASETS]
    dataset = datasets[0]
    adasyn = ALGORITHMS[1]
    smote = ALGORITHMS[2]
    lg_model = linear_model.LogisticRegression(random_state=4012, max_iter=100000)
    xgb_model = XGBClassifier(use_label_encoder=True, random_state=4012)
    
    '''For BaseLine'''
    lg_baseline = lg_model.fit(dataset.x_train, dataset.y_train)
    xgb_baseline = xgb_model.fit(dataset.x_train, dataset.y_train, eval_metric=['auc'])
    lg_baseline_pred = lg_baseline.predict(dataset.x_test)
    xgb_baseline_pred = xgb_baseline.predict(dataset.x_test)
    
    '''For ADASYN'''
    dataset.balance(adasyn)
    lg_adasyn = lg_model.fit(dataset.bxt, dataset.yxt)
    xgb_adasyn = xgb_model.fit(dataset.bxt, dataset.yxt, eval_metric=['auc'])
    lg_adasyn_pred = lg_adasyn.predict(dataset.x_test)
    xgb_adasyn_pred = xgb_adasyn.predict(dataset.x_test)
    
    '''For SMOTE'''
    dataset.balance(smote)
    lg_smote = lg_model.fit(dataset.bxt, dataset.yxt)
    xgb_smote = xgb_model.fit(dataset.bxt, dataset.yxt, eval_metric=['auc'])
    lg_smote_pred = lg_smote.predict(dataset.x_test)
    xgb_smote_pred = xgb_smote.predict(dataset.x_test)
    
    '''Evaluations'''
    print("BaseLine with Logistic Regression Summary\n")
    showResults(dataset.y_test, lg_baseline_pred)
    print("\nBaseLine with XGBoost Summary\n")
    showResults(dataset.y_test, xgb_baseline_pred)
    print("ADASYN with Logistic Regression Summary\n")
    showResults(dataset.y_test, lg_adasyn_pred)
    print("\nADASYN with XGBoost Summary\n")
    showResults(dataset.y_test, xgb_adasyn_pred)
    print("\nSMOTE with Logistic Regression Summary\n")
    showResults(dataset.y_test, lg_smote_pred)
    print("\nSMOTE with XGBoost Summary\n")
    showResults(dataset.y_test, xgb_smote_pred)

if __name__ == '__main__':
    main()
