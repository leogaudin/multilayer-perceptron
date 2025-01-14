import pandas as pd
import sys


def train_test_split(data: pd.DataFrame, test_size=0.2):
    if test_size <= 0 or test_size >= 1:
        raise Exception("Invalid test size")

    test = data.sample(frac=test_size)
    train = data.drop(test.index)

    return train, test


def main():
    if len(sys.argv) != 3:
        raise Exception(
            "Usage: python train_test_split.py <path_to_data> <test_size>"
        )

    path = sys.argv[1]
    test_size = float(sys.argv[2])

    if test_size <= 0 or test_size >= 1:
        raise Exception("Invalid test size")

    data = pd.read_csv(path)
    train, test = train_test_split(data, test_size=0.2)
    new_path = path.split(".")[0]
    train.to_csv(new_path + "_train.csv", index=False)
    test.to_csv(new_path + "_test.csv", index=False)


if __name__ == "__main__":
    main()
