import torch
from layers import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input) -> torch.Tensor:
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, optimizer) -> torch.Tensor:
        return output_gradient * self.activation_prime(self.input)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input) -> torch.Tensor:
        return 1 / (1 + torch.exp(-input))

    def activation_prime(self, input) -> torch.Tensor:
        return self.activation(input) * (1 - self.activation(input))


class ReLU(Activation):
    def __init__(self, leak=0):
        self.leak = leak
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input) -> torch.Tensor:
        return torch.maximum(input, torch.tensor(self.leak) * input)

    def activation_prime(self, input) -> torch.Tensor:
        return torch.maximum(input > 0, torch.tensor(self.leak))


class Softmax(Activation):
    def __init__(self):
        super().__init__(
            activation=self.activation,
            activation_prime=self.activation_prime
        )

    def activation(self, input) -> torch.Tensor:
        exp_x = torch.exp(input - torch.max(input, axis=1, keepdim=True)[0])
        return exp_x / torch.sum(exp_x, axis=1, keepdim=True)

    def activation_prime(self, input) -> torch.Tensor:
        return 1
