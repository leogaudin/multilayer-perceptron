import numpy as np
from classes.loss import Loss
from classes.layer import Layer


class Model:
    def __init__(self, layers: list[Layer], loss: Loss):
        self.layers = layers
        self.loss = loss

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient

    def fit(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                y_pred = self.forward(x)
                loss = self.loss.compute(y_pred, y)
                grad = self.loss.prime(y_pred, y)
                self.backward(grad, learning_rate)
            print("Epoch: ", epoch, " Loss: ", loss)

    def evaluate(self, x_test, y_test):
        correct = 0
        for x, y in zip(x_test, y_test):
            y_pred = self.forward(x)
            if np.argmax(y_pred) == np.argmax(y):
                correct += 1
        return correct / len(x_test)
