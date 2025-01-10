class Loss:
    def __init__(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def compute(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

    def prime(self, y_pred, y_true):
        return self.loss_prime(y_pred, y_true)
