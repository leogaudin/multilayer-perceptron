from model import Model
from preprocessing import load_data, to_categorical
import numpy as np
import losses


def main():
    model, _ = Model.load(filename="model")
    _, _, X_test, y_test = load_data(test_path="data_test.csv")

    y_pred = model.predict(X_test)
    y_true = to_categorical(y_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    print("Test accuracy:", f"{(y_pred == y_true).mean() * 100:.2f}%")

    loss = losses.CCE()
    print("Test loss:", loss.compute(y_pred, y_true) / len(y_true))


if __name__ == "__main__":
    main()
