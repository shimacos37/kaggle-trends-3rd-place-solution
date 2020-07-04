from typing import List
import numpy as np


def weighted_normalized_absolute_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: List = [0.3, 0.175, 0.175, 0.175, 0.175],
):
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    is_nan = np.isnan(y_true)
    y_pred[is_nan] = 0
    y_true[is_nan] = 0
    diff = np.abs(y_pred - y_true).sum(0)
    norm = y_true.sum(0)
    loss = (diff / norm) * weights
    return loss.sum()


def normalized_absolute_errors(
    y_true: np.ndarray, y_pred: np.ndarray,
):
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    is_nan = np.isnan(y_true)
    y_pred[is_nan] = 0
    y_true[is_nan] = 0
    diff = np.abs(y_pred - y_true).sum(0)
    norm = y_true.sum(0)
    loss = diff / norm
    return loss.sum()
