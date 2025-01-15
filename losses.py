import numpy as np
from classes.loss import Loss


class CCE(Loss):
    def __init__(self):
        super().__init__(self.cce, self.cce_prime)

    def cce(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred))

    # ONLY WORKS IF SOFTMAX WAS APPLIED TO Y_PRED
    def cce_prime(self, y_pred, y_true):
        return y_pred - y_true
