import numpy as np
import torch
from metadata import device


def zero(shape) -> torch.Tensor:
    return torch.zeros(shape, device=device)


def he(shape) -> torch.Tensor:
    return torch.randn(*shape, device=device) * np.sqrt(2 / shape[0])


def xavier(shape) -> torch.Tensor:
    return torch.randn(*shape, device=device) * np.sqrt(1 / shape[1])


def random(shape) -> torch.Tensor:
    return torch.randn(*shape, device=device)
