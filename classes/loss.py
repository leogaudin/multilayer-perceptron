class Loss:
    def __init__(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def compute(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def prime(self, y_true, y_pred):
        return self.loss_prime(y_true, y_pred)
