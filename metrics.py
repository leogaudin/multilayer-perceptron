import torch


def true_falses(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)

    true_positives = torch.sum((y_pred == 1) & (y_true == 1)).item()
    true_negatives = torch.sum((y_pred == 0) & (y_true == 0)).item()
    false_positives = torch.sum((y_pred == 1) & (y_true == 0)).item()
    false_negatives = torch.sum((y_pred == 0) & (y_true == 1)).item()

    return true_positives, true_negatives, false_positives, false_negatives
