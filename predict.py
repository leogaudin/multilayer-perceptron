from model import Model
from preprocessing import load_data
import numpy as np
import losses


def main():
    model, _ = Model.load(filename="model")
    _, _, X_test, y_test = load_data(test_path="data_test.csv")

    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_true = np.where(y_test == "M", 1, 0).reshape(-1, 1)
    print("Test accuracy:", f"{(y_pred == y_true).mean() * 100:.2f}%")

    loss = losses.BCE()
    print("Test loss:", loss.compute(y_pred, y_true) / len(y_true))


if __name__ == "__main__":
    main()
