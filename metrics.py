import numpy as np

def accuracy(y_true, y_pred):
    """Doğruluk oranını hesaplar."""
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """Kesinlik oranını hesaplar."""
    # True positives: y_true ve y_pred ikisi de 1 olan durumlar
    true_positive = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 1))

    # False positives: y_pred 1, ancak y_true 0 olan durumlar
    false_positive = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 0))

    if true_positive + false_positive == 0:
        return 0.0
    return true_positive / (true_positive + false_positive)

def recall(y_true, y_pred):
    """Duyarlılık oranını hesaplar."""
    true_positive = np.sum((np.array(y_pred) == 1) & np.array((y_true) == 1))
    false_negative = np.sum(np.array((y_pred) == 0) & np.array((y_true) == 1))
    if true_positive + false_negative == 0:
        return 0.0
    return true_positive / (true_positive + false_negative)

def f1_score(y_true, y_pred):
    """F1-Score hesaplar."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)
