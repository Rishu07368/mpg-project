"""
app.py — Streamlit front-end.

Run locally with:  streamlit run app.py

This file only wires UI together; all real logic (fitting, metrics,
plotting) lives in src/ so it stays testable and reusable outside Streamlit.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_mpg_data
from src.preprocessing import clean_data, train_test_split_manual, get_xy
from src.curve_fitting import fit_polynomial_numpy, predict, polynomial_equation_string
from src.evaluation import evaluate_all, residuals
from src.prediction import predict_with_context
from src.visualization import (
    plot_raw_scatter, plot_fits, plot_residuals, plot_actual_vs_predicted,
    plot_model_comparison_bar, plot_train_vs_test_error, plot_prediction_point,
)

st.set_page_config(page_title="Fuel Efficiency Curve-Fitting Lab", layout="wide")

X_LABEL, Y_LABEL = "Horsepower (hp)", "Fuel Efficiency (mpg)"


@st.cache_data
def get_data():
    raw = load_mpg_data()
    clean, report = clean_data(raw, x_col="horsepower", y_col="mpg")
    return raw, clean, report


raw_df, clean_df, clean_report = get_data()
x_all, y_all = get_xy(clean_df, drop_outliers=False)
(x_train, y_train), (x_test, y_test) = train_test_split_manual(x_all, y_all, test_fraction=0.2, seed=42)

st.title("🚗 Fuel Efficiency Predictor — Polynomial Curve Fitting")
st.caption(
    "Predicting a car's fuel efficiency (mpg) from its engine horsepower, "
    "using the least-squares polynomial fitting method from Experiment 2.1 — "
    "applied to a real 392-car dataset instead of six toy points."
)

tabs = st.tabs([
    "📊 Dashboard", "🔍 Data Explorer", "📈 Curve Fitting",
    "⚖️ Model Comparison", "🎯 Prediction", "📉 Residual Analysis",
])

# ---------------- Dashboard ----------------
with tabs[0]:
    st.subheader("Project Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cars (raw)", len(raw_df))
    c2.metric("Missing rows dropped", clean_report.missing_dropped)
    c3.metric("Outliers flagged (IQR)", clean_report.outliers_flagged)
    c4.metric("Usable data points", clean_report.rows_after)

    st.markdown(
        """
        **Who would use this?** A car buyer comparing engines, or an automotive
        engineer doing an early-stage estimate of fuel efficiency from a planned
        engine's horsepower, before a full physical prototype/dyno test exists.

        **Independent variable (x):** Horsepower (hp)
        **Dependent variable (y):** Fuel efficiency (miles per US gallon)

        **Dataset:** Auto MPG — 1970s–1980s US-market cars, originally from the
        UCI Machine Learning Repository / StatLib (Carnegie Mellon), retrieved
        here from the public `seaborn-data` GitHub mirror.
        """
    )
    st.pyplot(plot_raw_scatter(x_all, y_all, X_LABEL, Y_LABEL, "Raw Data: Horsepower vs Fuel Efficiency"))

# ---------------- Data Explorer ----------------
with tabs[1]:
    st.subheader("Raw Data Table")
    min_hp, max_hp = float(raw_df["horsepower"].min(skipna=True)), float(raw_df["horsepower"].max(skipna=True))
    hp_range = st.slider("Filter by horsepower", min_hp, max_hp, (min_hp, max_hp))
    filtered = raw_df[(raw_df["horsepower"] >= hp_range[0]) & (raw_df["horsepower"] <= hp_range[1])]
    st.dataframe(filtered, use_container_width=True, height=300)

    st.subheader("Summary Statistics")
    st.dataframe(raw_df[["mpg", "horsepower", "weight", "cylinders"]].describe(), use_container_width=True)

    st.subheader("Data Cleaning Report")
    st.write(
        f"- Rows before cleaning: **{clean_report.rows_before}**\n"
        f"- Rows with missing horsepower/mpg dropped: **{clean_report.missing_dropped}**\n"
        f"- Rows after cleaning: **{clean_report.rows_after}**\n"
        f"- Outliers flagged via IQR rule on horsepower (kept, not deleted): **{clean_report.outliers_flagged}**"
    )
    st.caption("Outliers are flagged, not silently removed — they're usually high-performance cars, which are real, valid data points.")

# ---------------- Curve Fitting ----------------
with tabs[2]:
    st.subheader("Fit Polynomials of Different Degrees")
    degrees_to_show = st.multiselect("Degrees to plot", [1, 2, 3, 4, 6, 8], default=[2, 3, 8])
    fits = {f"Degree {d}": fit_polynomial_numpy(x_train, y_train, d) for d in degrees_to_show}
    if fits:
        st.pyplot(plot_fits(x_train, y_train, fits, X_LABEL, Y_LABEL,
                             "Polynomial Fits (trained on 80% split)"))
        for label, coeffs in fits.items():
            st.code(f"{label}:  y = {polynomial_equation_string(coeffs, 'hp')}", language="text")
    else:
        st.info("Pick at least one degree above.")

# ---------------- Model Comparison ----------------
with tabs[3]:
    st.subheader("Model Comparison Table (honest: evaluated on held-out TEST data)")
    degrees = [1, 2, 3, 4, 6, 8]
    rows = []
    per_degree_coeffs = {}
    for d in degrees:
        coeffs = fit_polynomial_numpy(x_train, y_train, d)
        per_degree_coeffs[d] = coeffs
        test_metrics = evaluate_all(y_test, predict(coeffs, x_test))
        train_metrics = evaluate_all(y_train, predict(coeffs, x_train))
        rows.append({
            "Degree": d,
            "Train MSE": round(train_metrics["MSE"], 3),
            "Test MSE": round(test_metrics["MSE"], 3),
            "Test RMSE": round(test_metrics["RMSE"], 3),
            "Test MAE": round(test_metrics["MAE"], 3),
            "Test R²": round(test_metrics["R2"], 4),
        })
    comp_df = pd.DataFrame(rows)
    best_row = comp_df.loc[comp_df["Test RMSE"].idxmin()]
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.success(
        f"Lowest test RMSE: **Degree {int(best_row['Degree'])}** "
        f"(RMSE = {best_row['Test RMSE']} mpg). Note degree 2 and 3 are nearly "
        "identical — the simpler quadratic is preferable in practice (Occam's razor): "
        "it achieves almost the same accuracy with a model that's easier to interpret "
        "and less prone to wiggling on new data."
    )

    train_rmse_list = [round(evaluate_all(y_train, predict(per_degree_coeffs[d], x_train))["RMSE"], 4) for d in degrees]
    test_rmse_list = [round(evaluate_all(y_test, predict(per_degree_coeffs[d], x_test))["RMSE"], 4) for d in degrees]
    st.pyplot(plot_train_vs_test_error(degrees, train_rmse_list, test_rmse_list,
                                        "Overfitting Diagnostic: Train vs Test RMSE"))
    st.markdown(
        "As degree increases, **train error keeps falling** (the curve fits the "
        "training points ever more closely) while **test error stops improving and "
        "then rises** — that gap is overfitting: the high-degree polynomial is "
        "chasing noise in the training data rather than the underlying trend."
    )

# ---------------- Prediction ----------------
with tabs[4]:
    st.subheader("Predict Fuel Efficiency from Horsepower")
    chosen_degree = st.selectbox("Model to use", [1, 2, 3, 4, 6, 8], index=1)
    coeffs = fit_polynomial_numpy(x_train, y_train, chosen_degree)
    hp_query = st.number_input("Enter horsepower (hp)", min_value=10.0, max_value=600.0, value=100.0, step=5.0)

    result = predict_with_context(coeffs, hp_query, float(x_train.min()), float(x_train.max()))

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.metric("Predicted fuel efficiency", f"{result.y_pred:.1f} mpg")
        if result.mode == "interpolation":
            st.success(result.message)
        else:
            st.warning(result.message)
        st.write(
            f"**Interpretation:** At {hp_query:.0f} hp, the degree-{chosen_degree} "
            f"model predicts approximately **{result.y_pred:.1f} miles per gallon**."
        )
    with col2:
        st.pyplot(plot_prediction_point(
            x_all, y_all, coeffs, hp_query, result.y_pred, X_LABEL, Y_LABEL,
            f"Prediction — Degree {chosen_degree} model",
            float(x_train.min()), float(x_train.max()),
        ))

# ---------------- Residual Analysis ----------------
with tabs[5]:
    st.subheader("Residual Analysis")
    degree_r = st.selectbox("Model for residual analysis", [1, 2, 3, 4, 6, 8], index=1, key="resid_degree")
    coeffs_r = fit_polynomial_numpy(x_train, y_train, degree_r)
    resid_test = residuals(y_test, predict(coeffs_r, x_test))

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_residuals(x_test, resid_test, X_LABEL, f"Residuals — Degree {degree_r} (Test Set)"))
    with c2:
        st.pyplot(plot_actual_vs_predicted(y_test, predict(coeffs_r, x_test), f"Actual vs Predicted — Degree {degree_r}"))

    mean_resid = float(np.mean(resid_test))
    st.write(
        f"Mean residual on test set: **{mean_resid:+.3f} mpg**. "
        "If residuals cluster in a curved pattern rather than scattering randomly "
        "around zero, that's a sign the model's degree is too low (underfitting) "
        "and is systematically missing part of the relationship. Random scatter "
        "around zero — as seen here for degree 2–3 — indicates the polynomial has "
        "captured the main trend and what's left is closer to irreducible noise "
        "(measurement variation, unmodeled factors like transmission type, era, aerodynamics)."
    )

st.divider()
st.caption(
    "Extension of CU Experiment 2.1 (Curve Fitting — Least Squares Method) applied "
    "to a real 392-record automotive dataset. Built with NumPy (explicit least-squares "
    "and numpy.polyfit), Matplotlib, and Streamlit."
)
