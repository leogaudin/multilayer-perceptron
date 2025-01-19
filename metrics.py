import numpy as np


def true_falses(y_true, y_pred):
    # Assumes that one-hot encoding makes B [1, 0] and M [0, 1].
    # That should be the case if np.unique is used.
    # Here B is negative and M is positive.
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))

    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))

    return true_positives, true_negatives, false_positives, false_negatives
