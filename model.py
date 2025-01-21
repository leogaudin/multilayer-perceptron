from losses import Loss
from layers import Layer
from scaler import StandardScaler
import pickle
from metrics import true_falses


class Model:
    def __init__(
        self,
        layers: list[Layer],
        loss: Loss,
        optimizer,
        scaler: StandardScaler = None,
        patience=float("inf"),
    ):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.scaler = scaler
        self.patience = patience

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(
                output_gradient,
                self.optimizer
            )
        return output_gradient

    def fit(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs,
        batch_size=None,
    ):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        if self.scaler is not None:
            x_train = self.scaler.transform(x_train)
            x_test = self.scaler.transform(x_test)

        patience_counter = 0
        best_loss = float("inf")

        for epoch in range(epochs):
            if batch_size is None:
                batch_size = len(x_train)

            for i in range(0, len(x_train), batch_size):
                x_train_batch = x_train[i:i + batch_size]
                y_train_batch = y_train[i:i + batch_size]

                y_pred = self.forward(x_train_batch)
                gradients = self.loss.prime(y_pred, y_train_batch)
                self.backward(gradients)

            y_pred_train = self.forward(x_train)
            train_losses.append(self.loss.compute(y_pred_train, y_train)
                                / len(y_train))
            train_accuracies.append((y_pred_train.argmax(axis=1)
                                     == y_train.argmax(axis=1)).mean())

            y_pred_test = self.forward(x_test)
            test_losses.append(self.loss.compute(y_pred_test, y_test)
                               / len(y_test))
            test_accuracies.append((y_pred_test.argmax(axis=1)
                                    == y_test.argmax(axis=1)).mean())

            if test_losses[-1] < best_loss:
                best_loss = test_losses[-1]
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                "EPOCH", epoch, "\t",
                "loss:", f"{train_losses[-1]:.7f}", "\t",
                "val_loss:", f"{test_losses[-1]:.7f}", "\t",
                "accuracy:", f"{train_accuracies[-1]:.7f}", "\t",
                "val_accuracy:", f"{test_accuracies[-1]:.7f}"
            )

            (
                true_positives,
                true_negatives,
                false_positives,
                false_negatives
            ) = true_falses(y_test, y_pred_test)

            print(
                "Confusion:", "\t",
                "true_positives:", true_positives, "\t",
                "true_negatives:", true_negatives, "\t",
                "false_positives:", false_positives, "\t",
                "false_negatives:", false_negatives,
                "\n"
            )

            if patience_counter >= self.patience:
                print("EARLY STOPPING TRIGGERED")
                break

        print("Training finished.")

        return train_losses, test_losses, train_accuracies, test_accuracies

    def predict(self, x):
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return self.forward(x)

    def save(self, filename="model"):
        data = {
            "model": self,
            "scaler": self.scaler,
        }
        file = open(filename + ".pkl", "wb")
        pickle.dump(data, file)
        file.close()


def load_model(filename="model") -> tuple[Model, StandardScaler]:
    file = open(filename + ".pkl", "rb")
    data = pickle.load(file)
    file.close()
    return data['model'], data['scaler']
