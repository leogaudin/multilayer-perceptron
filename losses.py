import torch


class Loss:
    def __init__(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def compute(self, y_pred, y_true) -> torch.Tensor:
        return self.loss(y_pred, y_true)

    def prime(self, y_pred, y_true) -> torch.Tensor:
        return self.loss_prime(y_pred, y_true)


class CCE(Loss):
    def __init__(self):
        super().__init__(self.cce, self.cce_prime)

    def cce(self, y_pred, y_true) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        return -torch.sum(y_true * torch.log(y_pred))

    # ONLY WORKS IF SOFTMAX WAS APPLIED TO Y_PRED
    def cce_prime(self, y_pred, y_true) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        return y_pred - y_true


class BCE(Loss):
    def __init__(self):
        super().__init__(self.bce, self.bce_prime)

    def bce(self, y_pred, y_true) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        return -torch.sum(y_true * torch.log(y_pred)
                          + (1 - y_true) * torch.log(1 - y_pred)) / len(y_true)

    def bce_prime(self, y_pred, y_true) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))


class MSE(Loss):
    def __init__(self):
        super().__init__(self.mse, self.mse_prime)

    def mse(self, y_pred, y_true) -> torch.Tensor:
        return torch.mean((y_true - y_pred) ** 2)

    def mse_prime(self, y_pred, y_true) -> torch.Tensor:
        return y_pred - y_true
