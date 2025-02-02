from model import load_model
from preprocessing import load_data
import losses
import torch


def main():
    model, _ = load_model(filename="model")
    _, _, X_test, y_test = load_data(test_path="data_test.csv")

    y_pred = model.predict(X_test)
    y_true = y_test
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)

    print(
        "Test accuracy:",
        f"{(y_pred == y_true).float().mean().item() * 100:.2f}%"
    )

    loss = losses.CCE()
    print(
        "Test loss:",
        (loss.compute(y_pred, y_true).cpu() / len(y_true)).float().item()
    )


if __name__ == "__main__":
    main()
