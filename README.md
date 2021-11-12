# BT4012 Fraud Analytics Project

## Introduction

This literature review focuses on various oversampling techniques, namely Synthetic Minority Oversampling Technique (SMOTE), Adaptive Synthetic (ADASYN) and Robust Random Oversampling Examples (robROSE). The papers presented different oversampling approaches to resolve class imbalance, which is prevalent in fraud analytics.

## Prerequisite

- Python 3
- Anaconda (we will be using this to set up VE)
- R / RStudio (In PATH so that RScript is callable from terminal)  

## Installation guide

1. Setting up the virtual environment

- Create conda environment ```conda create -n bt4012 python=3.8```
- Activate environment ```conda activate bt4012```
- Install dependencies ```pip install -r requirements.txt```

2. Data preparation

    Download datasets at:

- Credit Card Fraud Dataset: <https://www.kaggle.com/mlg-ulb/creditcardfraud/download>
- Adult Dataset: <https://www.kaggle.com/uciml/adult-census-income/download>
- Swarm Behaviour Dataset: <https://www.kaggle.com/deepcontractor/swarm-behaviour-classification/download>

    Place downloaded files in ```data``` folder

3. Run Experiment

- Before running, go to the experiment python file and update the dataset path (if the data filename do not tally with the default)
- Run the individual experiement ```python src/experiment_1.py``` ```python src/experiment_2.py```  ```python src/experiment_3.py```
