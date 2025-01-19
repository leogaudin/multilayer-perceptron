import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def backward(self, output_gradient):
        pass


class Dense(Layer):
    def __init__(self, shape, initializer):
        self.weights = initializer(shape)
        self.biases = initializer((1, shape[1]))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient, optimizer):
        return optimizer.update(self, output_gradient)
