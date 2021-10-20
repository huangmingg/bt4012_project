from sampling.sampling import SamplingAlgorithm
from model.model import ClassifierWrapper
from preprocess.preprocess import DatasetWrapper
from config.config import ALGORITHMS, DATASETS, MODELS
import os


def main():
    datasets = [wrapper(filename) for wrapper, filename in DATASETS]
    algorithms = [i() for i in ALGORITHMS]
    for d in datasets:
        print("Processing data")
        d.process()
        for a in algorithms:
            x, y = a.run(d)
            for m in MODELS:
                model = m(a)
                model.train()


if __name__ == '__main__':
    main()