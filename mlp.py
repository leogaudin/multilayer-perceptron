# import numpy as np
import layers
import activations
import initializers
import losses
from model import Model
from preprocessing import load_data, to_categorical
from scaler import StandardScaler


def main():
    X_train, y_train, X_test, y_test = load_data(
        train_path="data_train.csv",
        test_path="data_test.csv"
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    model = Model(
        layers=[
            layers.Dense((30, 20), initializers.random),
            activations.ReLU(leak=0),
            layers.Dense((20, 10), initializers.random),
            activations.ReLU(leak=0),
            layers.Dense((10, 5), initializers.random),
            activations.ReLU(leak=0),
            layers.Dense((5, 2), initializers.random),
            activations.Softmax()
        ],
        scaler=scaler,
        loss=losses.CCE(),
        patience=42,
    )

    model.fit(
        x_train=X_train,
        y_train=to_categorical(y_train),
        # y_train=np.where(y_train == "M", 1, 0).reshape(-1, 1),
        x_test=X_test,
        y_test=to_categorical(y_test),
        # y_test=np.where(y_test == "M", 1, 0).reshape(-1, 1),
        epochs=1000,
        batch_size=32,
        learning_rate=0.001,
    )

    model.save()


if __name__ == "__main__":
    main()
