import numpy as np
from classes.activation import Activation


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(
            activation=lambda x: 1 / (1 + np.exp(-x)),
            activation_prime=lambda x: x * (1 - x)
        )


class ReLU(Activation):
    def __init__(self):
        super().__init__(
            activation=lambda x: np.maximum(0, x),
            activation_prime=lambda x: np.where(x > 0, 1, 0)
        )


class Softmax(Activation):
    def __init__(self):
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input):
        exp_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def activation_prime(self, input):
        return input
