import numpy as np
import pytest

from src.curve_fitting import fit_polynomial_manual, fit_polynomial_numpy, predict, polynomial_equation_string
from src.evaluation import mse, rmse, mae, r_squared
from src.preprocessing import clean_data, train_test_split_manual, get_xy
from src.prediction import predict_with_context
import pandas as pd


# ---------- curve_fitting ----------

def test_manual_matches_numpy_on_known_parabola():
    """y = 2x^2 - 3x + 1 exactly (noise-free) -> both solvers must recover it."""
    x = np.linspace(-5, 5, 20)
    y = 2 * x ** 2 - 3 * x + 1
    manual = fit_polynomial_manual(x, y, 2)
    numpy_fit = fit_polynomial_numpy(x, y, 2)
    np.testing.assert_allclose(manual, [2, -3, 1], atol=1e-8)
    np.testing.assert_allclose(numpy_fit, [2, -3, 1], atol=1e-8)
    np.testing.assert_allclose(manual, numpy_fit, atol=1e-6)


def test_predict_matches_manual_evaluation():
    coeffs = np.array([1.0, 0.0, -4.0])  # x^2 - 4
    assert predict(coeffs, 2.0) == pytest.approx(0.0, abs=1e-9)
    assert predict(coeffs, 0.0) == pytest.approx(-4.0, abs=1e-9)


def test_equation_string_contains_expected_terms():
    coeffs = np.array([1.5, -2.0, 3.0])
    s = polynomial_equation_string(coeffs, "hp")
    assert "hp^2" in s and "hp^1" not in s  # degree-1 term is rendered without ^1
    assert "hp" in s


def test_degree_zero_is_a_constant_mean_fit():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    coeffs = fit_polynomial_manual(x, y, 0)
    assert coeffs[0] == pytest.approx(y.mean())


# ---------- evaluation ----------

def test_metrics_are_zero_for_perfect_fit():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mse(y_true, y_pred) == 0
    assert rmse(y_true, y_pred) == 0
    assert mae(y_true, y_pred) == 0
    assert r_squared(y_true, y_pred) == pytest.approx(1.0)


def test_r_squared_zero_when_predicting_the_mean():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert r_squared(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)


def test_r_squared_handles_constant_y_without_crash():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 5.0, 5.0])
    assert r_squared(y_true, y_pred) == 0.0  # SS_tot == 0 edge case, defined as 0


# ---------- preprocessing ----------

def test_clean_data_drops_missing_rows():
    df = pd.DataFrame({
        "horsepower": [100.0, np.nan, 150.0, 200.0],
        "mpg": [20.0, 25.0, np.nan, 15.0],
    })
    cleaned, report = clean_data(df)
    assert report.rows_before == 4
    assert report.missing_dropped == 2  # rows 2 and 3 each have one NaN
    assert report.rows_after == 2


def test_clean_data_flags_outliers_without_deleting_them():
    df = pd.DataFrame({
        "horsepower": [90, 92, 95, 91, 93, 500],  # 500 is a clear IQR outlier
        "mpg": [25, 24, 23, 26, 25, 10],
    })
    cleaned, report = clean_data(df)
    assert report.rows_after == 6  # nothing deleted, only flagged
    assert report.outliers_flagged >= 1


def test_train_test_split_sizes_and_no_overlap():
    x = np.arange(100, dtype=float)
    y = x * 2
    (x_tr, y_tr), (x_te, y_te) = train_test_split_manual(x, y, test_fraction=0.25, seed=1)
    assert len(x_te) == 25
    assert len(x_tr) == 75
    assert set(x_tr.tolist()).isdisjoint(set(x_te.tolist()))


# ---------- prediction / interpolation vs extrapolation ----------

def test_prediction_flags_interpolation():
    coeffs = np.array([0.0, 1.0, 0.0])  # y = x
    result = predict_with_context(coeffs, 50, train_min=0, train_max=100)
    assert result.mode == "interpolation"
    assert result.y_pred == pytest.approx(50)


def test_prediction_flags_extrapolation():
    coeffs = np.array([0.0, 1.0, 0.0])
    result = predict_with_context(coeffs, 500, train_min=0, train_max=100)
    assert result.mode == "extrapolation"
    assert "OUTSIDE" in result.message


# ---------- edge cases / invalid input ----------

def test_fit_raises_on_mismatched_lengths():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        fit_polynomial_manual(x, y, 1)


def test_fit_raises_on_degree_too_high_for_points():
    # 3 points cannot support a degree-5 fit (needs >= 6 points) -> explicit,
    # informative ValueError rather than a silent/garbage numeric solve.
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 4.0, 9.0])
    with pytest.raises(ValueError):
        fit_polynomial_manual(x, y, 5)
