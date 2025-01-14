import numpy as np
from classes.activation import Activation
from classes.layer import Layer


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input):
        return 1 / (1 + np.exp(-input))

    def activation_prime(self, input):
        return input * (1 - input)


class ReLU(Activation):
    def __init__(self):
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input):
        return np.maximum(0, input)

    def activation_prime(self, input):
        return np.where(input > 0, 1, 0)


class Softmax(Layer):
    def forward(self, input):
        exp_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient
