import numpy as np
from layers import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, optimizer):
        return output_gradient * self.activation_prime(self.input)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input):
        return 1 / (1 + np.exp(-input))

    def activation_prime(self, input):
        return self.activation(input) * (1 - self.activation(input))


class ReLU(Activation):
    def __init__(self, leak=0):
        self.leak = leak
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input):
        return np.maximum(input, self.leak * input)

    def activation_prime(self, input):
        return np.maximum(input > 0, self.leak)


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
        return 1
