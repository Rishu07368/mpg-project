"""
run_analysis.py
----------------
End-to-end pipeline: load -> clean -> split -> fit (degrees 1-8) -> evaluate
-> save plots + a results.json with REAL computed numbers (no invented
metrics). Run this once to regenerate everything in outputs/.
"""

import json
from pathlib import Path

import numpy as np

from src.data_loader import load_mpg_data
from src.preprocessing import clean_data, train_test_split_manual, get_xy
from src.curve_fitting import fit_polynomial_manual, fit_polynomial_numpy, predict, polynomial_equation_string
from src.evaluation import evaluate_all, residuals
from src.prediction import predict_with_context
from src.visualization import (
    plot_raw_scatter, plot_fits, plot_residuals, plot_actual_vs_predicted,
    plot_model_comparison_bar, plot_train_vs_test_error, plot_prediction_point,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

X_LABEL, Y_LABEL = "Horsepower (hp)", "Fuel Efficiency (miles per US gallon)"


def main():
    results = {}

    # 1. Load
    raw = load_mpg_data()
    results["dataset"] = {
        "source": "seaborn-data GitHub mirror of the UCI/StatLib Auto MPG dataset (1970-1982 US market cars)",
        "n_rows_raw": len(raw),
        "columns": list(raw.columns),
    }

    # 2. Clean
    clean, report = clean_data(raw, x_col="horsepower", y_col="mpg")
    results["cleaning"] = {
        "rows_before": report.rows_before,
        "rows_after": report.rows_after,
        "missing_dropped": report.missing_dropped,
        "outliers_flagged": report.outliers_flagged,
    }

    x_all, y_all = get_xy(clean, drop_outliers=False)

    # 3. Sanity check: manual normal-equation solve vs numpy.polyfit agree
    manual_deg2 = fit_polynomial_manual(x_all, y_all, 2)
    numpy_deg2 = fit_polynomial_numpy(x_all, y_all, 2)
    results["manual_vs_numpy_check"] = {
        "manual_coeffs": manual_deg2.tolist(),
        "numpy_coeffs": numpy_deg2.tolist(),
        "max_abs_diff": float(np.max(np.abs(manual_deg2 - numpy_deg2))),
    }

    # 4. Train/test split (held out test set to detect overfitting honestly)
    (x_train, y_train), (x_test, y_test) = train_test_split_manual(x_all, y_all, test_fraction=0.2, seed=42)
    results["split"] = {"n_train": len(x_train), "n_test": len(x_test)}

    # 5. Fit multiple degrees, evaluate on BOTH train and test
    degrees = [1, 2, 3, 4, 6, 8]
    per_degree = {}
    fits_for_plot = {}
    for d in degrees:
        coeffs = fit_polynomial_numpy(x_train, y_train, d)
        y_train_pred = predict(coeffs, x_train)
        y_test_pred = predict(coeffs, x_test)
        train_metrics = evaluate_all(y_train, y_train_pred)
        test_metrics = evaluate_all(y_test, y_test_pred)
        per_degree[d] = {
            "coeffs": coeffs.tolist(),
            "equation": polynomial_equation_string(coeffs, "hp"),
            "train": train_metrics,
            "test": test_metrics,
        }
        if d in (2, 3, 8):
            fits_for_plot[f"Degree {d}"] = coeffs

    results["models"] = per_degree

    # 6. Pick best model by TEST RMSE (honest selection criterion)
    best_degree = min(degrees, key=lambda d: per_degree[d]["test"]["RMSE"])
    results["best_degree_by_test_rmse"] = best_degree

    # ---------------- PLOTS ----------------
    f1 = plot_raw_scatter(x_all, y_all, X_LABEL, Y_LABEL, "Auto MPG: Horsepower vs Fuel Efficiency (raw data)")
    f1.savefig(OUT / "01_raw_scatter.png", dpi=140)

    f2 = plot_fits(x_train, y_train, fits_for_plot, X_LABEL, Y_LABEL,
                   "Polynomial Fits (trained on 80% split) — Degree 2 vs 3 vs 8")
    f2.savefig(OUT / "02_fits_comparison.png", dpi=140)

    # Residuals for the chosen best model, on TEST data
    best_coeffs = np.array(per_degree[best_degree]["coeffs"])
    y_test_pred_best = predict(best_coeffs, x_test)
    resid = residuals(y_test, y_test_pred_best)
    f3 = plot_residuals(x_test, resid, X_LABEL, f"Residuals of Degree {best_degree} Model (Test Set)")
    f3.savefig(OUT / "03_residuals.png", dpi=140)

    f4 = plot_actual_vs_predicted(y_test, y_test_pred_best, f"Actual vs Predicted MPG — Degree {best_degree} (Test Set)")
    f4.savefig(OUT / "04_actual_vs_predicted.png", dpi=140)

    test_rmse_list = [per_degree[d]["test"]["RMSE"] for d in degrees]
    f5 = plot_model_comparison_bar(degrees, test_rmse_list, "Test RMSE (mpg)", "Model Comparison: Test RMSE by Polynomial Degree")
    f5.savefig(OUT / "05_model_comparison.png", dpi=140)

    train_rmse_list = [per_degree[d]["train"]["RMSE"] for d in degrees]
    f6 = plot_train_vs_test_error(degrees, train_rmse_list, test_rmse_list,
                                   "Overfitting Diagnostic: Train vs Test RMSE vs Polynomial Degree")
    f6.savefig(OUT / "06_train_vs_test_overfitting.png", dpi=140)

    # 7. Example prediction (interpolation) + example extrapolation
    train_min, train_max = float(x_train.min()), float(x_train.max())
    pred_interp = predict_with_context(best_coeffs, 100.0, train_min, train_max)
    pred_extrap = predict_with_context(best_coeffs, 300.0, train_min, train_max)
    results["example_predictions"] = {
        "interpolation_at_100hp": pred_interp.__dict__,
        "extrapolation_at_300hp": pred_extrap.__dict__,
    }

    f7 = plot_prediction_point(x_all, y_all, best_coeffs, 100.0, pred_interp.y_pred,
                                X_LABEL, Y_LABEL, f"Prediction Example — Degree {best_degree} Model",
                                train_min, train_max)
    f7.savefig(OUT / "07_prediction_example.png", dpi=140)

    plot_close_all()

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


def plot_close_all():
    import matplotlib.pyplot as plt
    plt.close("all")


if __name__ == "__main__":
    main()
