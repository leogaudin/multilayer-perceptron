import pandas as pd
import numpy as np


def load_data(train_path: str, test_path: str):
    df_train = pd.read_csv(train_path, index_col=False, header=0)
    df_test = pd.read_csv(test_path, index_col=False, header=0)

    features = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave points",
        "Symmetry",
        "Fractal dimension",
    ]

    X_train = df_train.drop(columns=["Diagnosis"])
    X_train = X_train[features].values
    y_train = df_train["Diagnosis"].values

    X_test = df_test.drop(columns=["Diagnosis"])
    X_test = X_test[features].values
    y_test = df_test["Diagnosis"].values

    return X_train, y_train, X_test, y_test


def to_categorical(y):
    set = np.unique(y)
    y_one_hot = np.zeros((len(y), len(set)))
    for i, val in enumerate(y):
        y_one_hot[i, np.where(set == val)] = 1
    return y_one_hot
