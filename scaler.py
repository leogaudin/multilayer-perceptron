import torch


class StandardScaler():
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def save(self, filename):
        torch.save({'mean': self.mean, 'std': self.std}, filename)

    def load(self, filename):
        data = torch.load(filename)
        if 'mean' not in data or 'std' not in data:
            raise ValueError('No mean or std in the file.')
        self.mean = data['mean']
        self.std = data['std']

    def fit(self, X):
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean
