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
        pass
