from sampling.sampling import SamplingAlgorithm
from model.model import ClassifierWrapper
from preprocess.preprocess import DatasetWrapper
from config.config import ALGORITHMS, DATASETS, MODELS


def evaluate(d: DatasetWrapper, a: SamplingAlgorithm, m: ClassifierWrapper) -> None:
    d.balance(a)
    model = m(d)
    model.train()

def main():
    datasets = [wrapper(filename) for wrapper, filename in DATASETS]
    for d in datasets:
        for a in ALGORITHMS:
            for m in MODELS:
                evaluate(d, a, m)

if __name__ == '__main__':
    main()



