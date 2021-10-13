from sampling.sampling import SamplingAlgorithm
from model.model import ClassifierWrapper
from preprocess.preprocess import DatasetWrapper
from config.config import ALGORITHMS, DATASETS, MODELS
import os


def main():
    algorithms = [i() for i in ALGORITHMS]
    datasets = [wrapper(filename) for wrapper, filename in DATASETS]
    for d in datasets:
        d.process()
        for a in algorithms:
            for m in MODELS:
                model = m(d)
                model.train()


if __name__ == '__main__':
    main()