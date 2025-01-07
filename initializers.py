import numpy as np


def zero(shape):
    return np.zeros(shape)


def he(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


def xavier(shape):
    return np.random.randn(*shape) * np.sqrt(1 / shape[1])


def random(shape):
    return np.random.randn(*shape)
