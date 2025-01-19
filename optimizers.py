import numpy as np


def calculate_gradients(output_gradient, layer):
    weights_gradient = np.dot(layer.input.T, output_gradient)
    biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
    input_gradient = np.dot(output_gradient, layer.weights.T)

    return weights_gradient, biases_gradient, input_gradient


class Optimizer:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, layer, output_gradient):
        if not hasattr(layer, "weights_velocity")\
                or not hasattr(layer, "biases_velocity"):
            layer.weights_velocity = np.zeros_like(layer.weights)
            layer.biases_velocity = np.zeros_like(layer.biases)

        (
            weights_gradient,
            biases_gradient,
            input_gradient
        ) = calculate_gradients(output_gradient, layer)

        layer.weights_velocity = self.momentum * layer.weights_velocity \
            + self.learning_rate * weights_gradient
        layer.biases_velocity = self.momentum * layer.biases_velocity \
            + self.learning_rate * biases_gradient

        layer.weights -= layer.weights_velocity
        layer.biases -= layer.biases_velocity

        return input_gradient


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon

    def update(self, layer, output_gradient):
        if not hasattr(layer, "weights_velocity")\
                or not hasattr(layer, "biases_velocity"):
            layer.weights_velocity = np.zeros_like(layer.weights)
            layer.biases_velocity = np.zeros_like(layer.biases)

        (
            weights_gradient,
            biases_gradient,
            input_gradient
        ) = calculate_gradients(output_gradient, layer)

        layer.weights_velocity = self.decay * layer.weights_velocity \
            + (1 - self.decay) * weights_gradient ** 2
        layer.biases_velocity = self.decay * layer.biases_velocity \
            + (1 - self.decay) * biases_gradient ** 2

        layer.weights -= self.learning_rate * weights_gradient \
            / (np.sqrt(layer.weights_velocity) + self.epsilon)
        layer.biases -= self.learning_rate * biases_gradient \
            / (np.sqrt(layer.biases_velocity) + self.epsilon)

        return input_gradient


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        momentum_decay=0.9,
        rms_decay=0.999,
        epsilon=1e-7
    ):
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.rms_decay = rms_decay
        self.epsilon = epsilon

    def update(self, layer, output_gradient):
        if not hasattr(layer, "weights_momentum")\
                or not hasattr(layer, "biases_momentum")\
                or not hasattr(layer, "weights_rms")\
                or not hasattr(layer, "biases_rms"):
            layer.weights_momentum = np.zeros_like(layer.weights)
            layer.biases_momentum = np.zeros_like(layer.biases)
            layer.weights_rms = np.zeros_like(layer.weights)
            layer.biases_rms = np.zeros_like(layer.biases)

        (
            weights_gradient,
            biases_gradient,
            input_gradient
        ) = calculate_gradients(output_gradient, layer)

        layer.weights_momentum = self.momentum_decay * layer.weights_momentum \
            + (1 - self.momentum_decay) * weights_gradient
        layer.biases_momentum = self.momentum_decay * layer.biases_momentum \
            + (1 - self.momentum_decay) * biases_gradient
        layer.weights_rms = self.rms_decay * layer.weights_rms \
            + (1 - self.rms_decay) * weights_gradient ** 2
        layer.biases_rms = self.rms_decay * layer.biases_rms \
            + (1 - self.rms_decay) * biases_gradient ** 2

        layer.weights -= self.learning_rate /\
            (self.epsilon + np.sqrt(layer.weights_rms))\
            * layer.weights_momentum
        layer.biases -= self.learning_rate /\
            (self.epsilon + np.sqrt(layer.biases_rms))\
            * layer.biases_momentum

        return input_gradient
