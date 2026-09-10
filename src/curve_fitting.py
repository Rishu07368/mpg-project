"""
curve_fitting.py
-----------------
This is the heart of the project and the direct extension of Experiment 2.1.

Two implementations are provided on purpose:

1. `fit_polynomial_manual` — builds the Vandermonde design matrix and solves
   the normal equations  (X^T X) c = X^T y  explicitly, exactly the maths
   taught in the least-squares theory section of the experiment. This is
   what a viva examiner will expect you to be able to explain on a
   whiteboard.

2. `fit_polynomial_numpy` — a thin wrapper around `numpy.polyfit`, which
   uses a numerically-stabler QR/SVD solve internally. This is what the
   original experiment used, and what the production app actually calls,
   because for ill-conditioned high-degree fits the manual normal-equation
   solve can lose precision. Both return the same coefficients (within
   floating-point tolerance) for well-conditioned low-degree fits — we
   verify that in tests/test_curve_fitting.py.
"""

import numpy as np


def _design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Vandermonde matrix: columns [x^degree, x^(degree-1), ..., x^1, x^0]."""
    return np.vstack([x ** p for p in range(degree, -1, -1)]).T


def fit_polynomial_manual(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Solve the normal equations (X^T X) c = X^T y directly.

    For a degree-2 fit y = a*x^2 + b*x + c this is precisely the parabola
    fitting shown in the experiment, generalised to any degree.
    """
    if len(x) <= degree:
        raise ValueError(
            f"Cannot fit a degree-{degree} polynomial with only {len(x)} data "
            f"point(s): the normal equations become singular/underdetermined. "
            f"Need at least degree+1 = {degree + 1} points."
        )
    X = _design_matrix(x, degree)
    XtX = X.T @ X
    Xty = X.T @ y
    coeffs = np.linalg.solve(XtX, Xty)
    return coeffs  # highest power first, matching numpy.polyfit's convention


def fit_polynomial_numpy(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Wrapper around numpy.polyfit — same convention as Experiment 2.1."""
    return np.polyfit(x, y, degree)


def predict(coeffs: np.ndarray, x_new) -> np.ndarray:
    """Evaluate the fitted polynomial at new x value(s)."""
    return np.polyval(coeffs, x_new)


def polynomial_equation_string(coeffs: np.ndarray, var: str = "x") -> str:
    """Render coefficients as a readable equation string, e.g.
    '-0.0013 x^2 + 0.5432 x^1 + 12.30'.
    """
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if power == 0:
            terms.append(f"{c:+.5f}")
        elif power == 1:
            terms.append(f"{c:+.5f}{var}")
        else:
            terms.append(f"{c:+.5f}{var}^{power}")
    return " ".join(terms)
