import torch


class Layer:
    def __init__(self):
        pass

    def forward(self, input) -> torch.Tensor:
        pass

    def backward(self, output_gradient) -> torch.Tensor:
        pass


class Dense(Layer):
    def __init__(self, shape, initializer):
        self.weights = initializer(shape)
        self.biases = initializer((1, shape[1]))

    def forward(self, input) -> torch.Tensor:
        self.input = input
        return torch.matmul(input, self.weights) + self.biases

    def backward(self, output_gradient, optimizer) -> torch.Tensor:
        return optimizer.update(self, output_gradient)
