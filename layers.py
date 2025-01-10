import numpy as np
from classes.layer import Layer


class Dense(Layer):
    def __init__(self, shape, initializer):
        self.weights = initializer(shape)
        self.biases = np.zeros((1, shape[1]))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        if self.input is None:
            raise Exception("No input to backpropagate")

        output_gradient = np.sum(output_gradient, axis=0)
        output_gradient.reshape(1, output_gradient.shape[0])

        input_gradient = np.dot(self.weights, output_gradient.T)

        if len(self.input.shape) == 1:
            self.input = np.reshape(self.input, (1, self.input.shape[0]))

        self.weights -= learning_rate * (output_gradient * self.input.T)
        self.biases -= learning_rate * np.sum(output_gradient, axis=0)

        return input_gradient
