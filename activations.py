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
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input):
        self.output = np.maximum(0, input)
        return self.output

    def activation_prime(self, output_gradient):
        return np.where(self.output > 0, 1, 0)


class Softmax(Activation):
    def __init__(self):
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input):
        exp_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def activation_prime(self, output_gradient):
        return output_gradient
