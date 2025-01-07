import numpy as np


class Dense:
    def __init__(self, shape, activation, initializer):
        self.weights = initializer(shape)
        self.biases = np.zeros((1, shape[1]))
        self.activation = activation

    def compute(self, input):
        return np.dot(input, self.weights) + self.biases

    def forward(self, input):
        return self.activation(self.compute(input))
