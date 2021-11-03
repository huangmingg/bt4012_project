from config.config import ALGORITHMS, DATASETS, MODELS


def main():
    datasets = [wrapper(filename) for wrapper, filename in DATASETS]
    for d in datasets:
        for a in ALGORITHMS:
            for m in MODELS:
                d.balance(a)
                model = m(d)
                print(f"Evaluating {type(model).__name__} model for algorithm {a.__name__} using dataset {type(d).__name__}")
                model.train()
                model.evaluate()


if __name__ == '__main__':
    main()



