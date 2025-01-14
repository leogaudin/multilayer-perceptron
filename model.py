import numpy as np
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
        batch_size=128,
    ):
        loss_accum = []
        test_loss_accum = []

        for epoch in range(epochs):
            batch = np.random.choice(len(x_train), batch_size)
            x_train_batch = x_train[batch]
            y_train_batch = y_train[batch]
            loss = 0
            test_loss = 0

            y_pred = self.forward(x_train_batch)
            loss += loss_function.compute(y_pred, y_train_batch)
            grad = loss_function.prime(y_pred, y_train_batch)
            self.backward(grad, learning_rate)

            for x, y in zip(x_test, y_test):
                y_pred = self.forward(x)
                test_loss += loss_function.compute(y_pred, y)

            loss_accum.append(loss / len(x_train))
            test_loss_accum.append(test_loss / len(x_test))

            print(
                "Epoch: ", epoch,
                "Loss: ", loss_accum[-1],
                "Test Loss: ", test_loss_accum[-1],
            )

        plt.plot(loss_accum, label="Train Loss")
        # plt.plot(test_loss_accum, label="Test Loss")
        plt.legend()
        plt.show()
