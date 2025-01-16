from classes.loss import Loss
from classes.layer import Layer
import matplotlib.pyplot as plt


class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient

    def fit(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs,
        learning_rate,
        loss_function: Loss,
        batch_size=None,
    ):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            if batch_size is None:
                batch_size = len(x_train)

            for i in range(0, len(x_train), batch_size):
                x_train_batch = x_train[i:i + batch_size]
                y_train_batch = y_train[i:i + batch_size]

                y_pred = self.forward(x_train_batch)
                gradients = loss_function.prime(y_pred, y_train_batch)
                self.backward(gradients, learning_rate)

            y_pred_train = self.forward(x_train)
            train_losses.append(loss_function.compute(y_pred_train, y_train)
                                / len(y_train))
            train_accuracies.append((y_pred_train.argmax(axis=1)
                                     == y_train.argmax(axis=1)).mean())

            y_pred_test = self.forward(x_test)
            test_losses.append(loss_function.compute(y_pred_test, y_test)
                               / len(y_test))
            test_accuracies.append((y_pred_test.argmax(axis=1)
                                    == y_test.argmax(axis=1)).mean())

            print(
                "EPOCH", epoch, "\t",
                "loss: ", f"{train_losses[-1]:.7f}", "\t",
                "val_loss: ", f"{test_losses[-1]:.7f}", "\t",
                "accuracy: ", f"{train_accuracies[-1]:.7f}", "\t",
                "val_accuracy: ", f"{test_accuracies[-1]:.7f}"
            )

        fig, ax = plt.subplots(2)

        ax[0].plot(train_losses, label="Train Loss")
        ax[0].plot(test_losses, label="Test Loss")
        ax[0].legend()

        ax[1].plot(train_accuracies, label="Train Accuracy")
        ax[1].plot(test_accuracies, label="Test Accuracy")
        ax[1].legend()

        plt.show()
