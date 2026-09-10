"""
prediction.py
--------------
Wraps curve_fitting.predict with the real-world guard rails the brief asks
for: telling the user whether their query is an INTERPOLATION (inside the
range the model was trained on -> trustworthy) or an EXTRAPOLATION (outside
that range -> the polynomial is unconstrained out there and can shoot off
to unrealistic values, so the answer must be flagged, not just returned).
"""

from dataclasses import dataclass

import numpy as np

from .curve_fitting import predict as _predict


@dataclass
class PredictionResult:
    x_value: float
    y_pred: float
    mode: str          # "interpolation" or "extrapolation"
    train_min: float
    train_max: float
    message: str


def predict_with_context(coeffs: np.ndarray, x_value: float, train_min: float, train_max: float) -> PredictionResult:
    if x_value < train_min or x_value > train_max:
        mode = "extrapolation"
        message = (
            f"{x_value} is OUTSIDE the training range "
            f"[{train_min:.1f}, {train_max:.1f}]. Polynomials are unconstrained "
            "beyond the fitted range, so this prediction is a mathematical "
            "extension of the curve, not a validated estimate — treat it as "
            "indicative only, especially for higher-degree fits."
        )
    else:
        mode = "interpolation"
        message = (
            f"{x_value} lies inside the training range "
            f"[{train_min:.1f}, {train_max:.1f}], so this is an interpolation "
            "and is the model's more reliable regime."
        )

    y_pred = float(_predict(coeffs, x_value))
    return PredictionResult(
        x_value=x_value, y_pred=y_pred, mode=mode,
        train_min=train_min, train_max=train_max, message=message,
    )
