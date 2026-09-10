"""
visualization.py
-----------------
All Matplotlib plotting lives here so app.py / scripts stay declarative:
"give me a figure for X" rather than repeating plt boilerplate everywhere.

Every function returns a Matplotlib Figure (never calls plt.show()), so it
works identically from a script (savefig) or from Streamlit (st.pyplot).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_raw_scatter(x, y, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="#c0392b", s=25, alpha=0.75, edgecolor="white", linewidth=0.4, label="Observed data")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fits(x, y, fits: dict, x_label, y_label, title, x_smooth=None):
    """fits: {"Degree 2": coeffs_array, "Degree 3": coeffs_array, ...}"""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, y, color="#c0392b", s=20, alpha=0.6, label="Observed data", zorder=1)

    if x_smooth is None:
        x_smooth = np.linspace(x.min(), x.max(), 300)

    colors = ["#2980b9", "#27ae60", "#8e44ad", "#f39c12", "#16a085"]
    for i, (label, coeffs) in enumerate(fits.items()):
        y_smooth = np.polyval(coeffs, x_smooth)
        ax.plot(x_smooth, y_smooth, label=label, linewidth=2.2, color=colors[i % len(colors)], zorder=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(x, residual_values, x_label, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(x, residual_values, color="#2980b9", s=20, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "--", color="gray", linewidth=1.5, label="Perfect prediction")
    ax.scatter(y_true, y_pred, color="#27ae60", s=20, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Actual MPG")
    ax.set_ylabel("Predicted MPG")
    ax.set_title(title)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_model_comparison_bar(degrees, metric_values, metric_name, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f"Deg {d}" for d in degrees], metric_values, color="#2980b9", alpha=0.85)
    for b, v in zip(bars, metric_values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_train_vs_test_error(degrees, train_rmse, test_rmse, title):
    """The single most important diagnostic plot in the project: it makes
    overfitting visually undeniable once test RMSE stops improving (or gets
    worse) while train RMSE keeps dropping as degree increases."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(degrees, train_rmse, "o-", label="Train RMSE", color="#2980b9", linewidth=2)
    ax.plot(degrees, test_rmse, "o-", label="Test RMSE", color="#c0392b", linewidth=2)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE (mpg)")
    ax.set_title(title)
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_prediction_point(x, y, coeffs, x_query, y_query, x_label, y_label, title, train_min, train_max):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(x, y, color="#c0392b", s=18, alpha=0.5, label="Observed data")
    x_smooth = np.linspace(min(x.min(), x_query) - 2, max(x.max(), x_query) + 2, 300)
    y_smooth = np.polyval(coeffs, x_smooth)
    ax.plot(x_smooth, y_smooth, color="#2980b9", linewidth=2, label="Fitted curve")
    ax.axvspan(train_min, train_max, color="#27ae60", alpha=0.07, label="Training range")
    marker_color = "#27ae60" if train_min <= x_query <= train_max else "#e67e22"
    ax.scatter([x_query], [y_query], color=marker_color, s=140, marker="*",
               edgecolor="black", linewidth=0.8, zorder=5, label=f"Prediction @ x={x_query}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
