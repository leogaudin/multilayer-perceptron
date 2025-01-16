import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, layer, output_gradient):
        if not hasattr(layer, "weights_velocity")\
                or not hasattr(layer, "biases_velocity"):
            layer.weights_velocity = np.zeros_like(layer.weights)
            layer.biases_velocity = np.zeros_like(layer.biases)

        weights_gradient = np.dot(layer.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, layer.weights.T)

        layer.weights_velocity = self.momentum * layer.weights_velocity \
            + self.learning_rate * weights_gradient
        layer.biases_velocity = self.momentum * layer.biases_velocity \
            + self.learning_rate * biases_gradient

        layer.weights -= layer.weights_velocity
        layer.biases -= layer.biases_velocity

        return input_gradient
