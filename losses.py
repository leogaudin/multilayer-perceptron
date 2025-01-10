import numpy as np
from classes.loss import Loss


class CCE(Loss):
    def __init__(self):
        super().__init__(self.cce, self.cce_prime)

    def cce(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.sum(np.dot(y_true.T, np.log(y_pred_clipped).T))

    def cce_prime(self, y_pred, y_true):
        # ONLY WORKS IF SOFTMAX WAS APPLIED TO THE OUTPUT LAYER
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return y_pred_clipped - y_true
        # return -y_true / y_pred_clipped
