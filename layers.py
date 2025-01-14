import numpy as np
from classes.layer import Layer


class Dense(Layer):
    def __init__(self, shape, initializer):
        self.weights = initializer(shape)
        self.biases = initializer((1, shape[1]))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient
