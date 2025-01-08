import numpy as np
from classes.loss import Loss


class MSE(Loss):
    def __init__(self):
        super().__init__(self.mse, self.mse_prime)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class MAE(Loss):
    def __init__(self):
        super().__init__(self.mae, self.mae_prime)

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mae_prime(self, y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.size


# def cce(y_pred, y_true):
#     n = y_pred.shape[0]
#     clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

#     if len(y_true.shape) == 1:
#         confidences = clipped[range(n), y_true]

#     elif len(y_true.shape) == 2:
#         confidences = np.sum(y_true * clipped, axis=1)

#     else:
#         raise ValueError('Invalid shape of y_true')

#     return np.mean(-np.log(confidences))
