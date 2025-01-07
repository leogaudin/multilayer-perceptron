import numpy as np


def cce(y_pred, y_true):
    n = y_pred.shape[0]
    clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

    if len(y_true.shape) == 1:
        confidences = clipped[range(n), y_true]

    elif len(y_true.shape) == 2:
        confidences = np.sum(y_true * clipped, axis=1)

    else:
        raise ValueError('Invalid shape of y_true')

    return np.mean(-np.log(confidences))


def ssr(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))
