"""
evaluation.py
--------------
Error metrics computed explicitly with NumPy (no sklearn dependency in the
core logic, so the maths is visible) — MSE, RMSE, MAE, R^2 and residuals.
"""

import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_true - y_pred


def mse(y_true, y_pred) -> float:
    r = residuals(y_true, y_pred)
    return float(np.mean(r ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    r = residuals(y_true, y_pred)
    return float(np.mean(np.abs(r)))


def r_squared(y_true, y_pred) -> float:
    """R^2 = 1 - SS_res / SS_tot.

    SS_res: sum of squared residuals (unexplained variance)
    SS_tot: total variance of y around its mean (baseline: 'predict the mean')
    """
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def evaluate_all(y_true, y_pred) -> dict:
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r_squared(y_true, y_pred),
    }
