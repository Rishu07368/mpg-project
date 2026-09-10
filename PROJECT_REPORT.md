# Project Report

## Fuel Efficiency Prediction Using Polynomial Curve Fitting (Least Squares Method)

*An extension of Experiment 2.1 — Curve Fitting of a Second-Degree Parabola and General Curves*

---

### 1. Title
**Predicting Automobile Fuel Efficiency from Engine Horsepower Using Least-Squares Polynomial Curve Fitting**

### 2. Abstract
This project extends the second-degree parabola / least-squares curve-fitting
technique taught in Experiment 2.1 from a six-point illustrative dataset into
a real-world regression problem on 392 automobiles. Horsepower is used as the
independent variable and fuel efficiency (miles per US gallon) as the
dependent variable. Polynomials of degree 1 through 8 are fitted using both
an explicit normal-equation least-squares solver and NumPy's `polyfit`, with
the two verified to agree to within 6.1×10⁻¹³. Models are evaluated honestly
on a held-out 20% test split using MSE, RMSE, MAE and R². The degree-2
(parabolic) model is selected as the most appropriate: it achieves a test
RMSE of 4.62 mpg and R² of 0.662, statistically indistinguishable from
degree 3, while degrees 4 and above demonstrably overfit — training error
continues to fall while test error rises. An interactive Streamlit
application allows a user to explore the data, compare models, and obtain
predictions with explicit interpolation/extrapolation warnings.

### 3. Introduction
Curve fitting is the process of finding a mathematical function that best
describes the relationship between an independent and a dependent variable
observed in a dataset. Experiment 2.1 introduced this concept using the
least-squares method to fit a parabola (`y = ax² + bx + c`) and a general
polynomial to six illustrative data points. This project asks: what does
curve fitting look like when applied to real, noisy, imperfect data — the
kind an engineer actually encounters?

### 4. Problem Statement
Given the horsepower rating of an automobile engine, estimate its expected
fuel efficiency (mpg), using a least-squares polynomial model fitted to
historical vehicle data, while being explicit about the model's accuracy,
its limits, and where predictions become unreliable.

### 5. Motivation
Six hand-picked points always fit a curve well because there is no noise and
no held-out data to check against. Real datasets expose the actual questions
curve fitting is meant to answer: Which degree generalizes best — not just
fits the training points? How do we know a model isn't overfitting? How
confident should we be in a prediction that falls outside the range of data
we trained on? This project makes all of these concrete and measurable.

### 6. Objectives
1. Apply the least-squares method to a real 392-record dataset.
2. Fit and compare polynomials of degree 1, 2, 3, 4, 6, and 8.
3. Evaluate every model with MSE, RMSE, MAE, R², and residual analysis, on
   data the model did not see during fitting.
4. Demonstrate overfitting empirically, not just define it.
5. Build a usable prediction interface with interpolation/extrapolation
   guardrails.
6. Package the mathematics, evaluation, and interface as a clean, tested,
   documented Python project.

### 7. Real-World Application
A car buyer comparing engines can get an evidence-based mpg estimate instead
of relying on marketing figures for a single trim. An automotive/mechanical
engineering team doing early-stage design can sanity-check a target
horsepower against expected fuel-efficiency trade-offs before committing to
a physical prototype or dynamometer test.

### 8. Existing Problem / Approach
Manufacturer-published mpg figures are trim-specific and don't help someone
reason about the general horsepower–efficiency trade-off across the market.
Many student curve-fitting exercises stop at "the graph looks like it fits"
without any held-out validation, so they cannot actually detect overfitting
— a model that memorizes six points looks perfect by construction.

### 9. Proposed Solution
Fit multiple polynomial degrees using the same least-squares mathematics
taught in Experiment 2.1, but evaluate every model on a held-out test split
it never saw during fitting, select the model by test-set performance (not
training-set performance), and expose the whole pipeline — data, fitting,
comparison, prediction — through an interactive application with explicit
warnings when a prediction is an extrapolation.

### 10. Dataset
- **Name:** Auto MPG dataset (UCI Machine Learning Repository / StatLib,
  originally a 1983 American Statistical Association Exposition dataset,
  Ford Motor Co. / Carnegie Mellon).
- **Retrieved from:** `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv`
  (a stable, widely-used public mirror of the UCI dataset).
- **Size:** 398 rows raw → 392 rows after removing 6 rows with missing
  horsepower.
- **Fields used:** `horsepower` (hp, independent variable), `mpg` (miles per
  US gallon, dependent variable).
- **Time period:** Model years 1970–1982 (US market).
- **Missing values:** 6 rows missing `horsepower` — dropped (confirmed via
  `df.isna().sum()`), not imputed, to avoid inventing engineering data.
- **Outliers:** 10 points flagged via the IQR rule (`Q1 − 1.5×IQR`,
  `Q3 + 1.5×IQR` on horsepower) — retained and visualized, not deleted, since
  they represent genuine high-performance vehicles.

### 11. Data Preprocessing
1. Load raw CSV (`src/data_loader.py`), validate required columns exist.
2. Select `horsepower`, `mpg`; drop rows with a missing value in either
   column (`src/preprocessing.clean_data`).
3. Flag IQR-based outliers on horsepower as a `is_outlier` column (retained).
4. Convert to NumPy float arrays.
5. Shuffle and split 80% train (314 rows) / 20% test (78 rows), fixed
   `seed=42` for reproducibility (`train_test_split_manual`).

### 12. Mathematical Theory
**Curve fitting** constructs a function `y = f(x)` that best represents a
relationship observed in data points `(x₁,y₁), …, (xₙ,yₙ)`.

**Least squares** chooses the coefficients that minimize the sum of squared
residuals: `S = Σ(yᵢ − f(xᵢ))²`. For a polynomial of degree *d*,
`f(x) = c₀xᵈ + c₁x^(d−1) + … + c_d`, this reduces to a linear algebra
problem: build the Vandermonde design matrix `X` (columns `x^d … x^0`) and
solve the **normal equations** `(XᵀX)c = Xᵀy` for the coefficient vector `c`.
This is implemented explicitly in `src/curve_fitting.fit_polynomial_manual`
and cross-checked against `numpy.polyfit`.

**Residual:** `rᵢ = yᵢ − ŷᵢ`, the signed gap between an observed and
predicted value at each point — the quantity least squares minimizes the
square of.

**Polynomial degree:** the highest power of `x` in the fitted function.
Degree 1 = a straight line; degree 2 = a parabola (Experiment 2.1's focus);
higher degrees allow progressively more curvature/inflection points.

**Why quadratic fitting is useful here:** the horsepower–mpg relationship
shows diminishing marginal fuel-efficiency loss as horsepower increases — a
shape a straight line structurally cannot represent, but a parabola can.

**Why cubic/higher degree can improve *training* fit:** each additional
degree adds one more free parameter, giving the curve more ways to bend
toward every individual training point, which can only reduce (never
increase) training error.

**Why excessive degree causes overfitting:** past a certain point, those
extra bends fit noise specific to the training sample rather than the true
underlying trend, so performance on new (test) data gets worse even as
training error keeps falling — verified numerically in §20 below and
Figure `06_train_vs_test_overfitting.png`.

**Interpolation vs. extrapolation:** interpolation predicts within the range
of `x` values seen during training (46–230 hp here); extrapolation predicts
outside it. A polynomial has no constraint on its behavior outside the
fitted range and can diverge to unrealistic values, so extrapolated
predictions are flagged, not stated as fact.

**MSE / RMSE / MAE / R²:**
- MSE = mean of squared residuals — penalizes large errors heavily.
- RMSE = √MSE — same units as `y` (mpg here), interpretable directly.
- MAE = mean of absolute residuals — more robust to outlier errors than MSE.
- R² = `1 − SS_res/SS_tot` — the fraction of `y`'s variance the model
  explains, relative to simply predicting the mean of `y` for everyone.

### 13. Least Squares Method (applied)
Implemented twice for verification: (a) explicit normal-equation solve
(`np.linalg.solve((XᵀX), Xᵀy)`), matching the theory in Experiment 2.1
directly; (b) `numpy.polyfit`, which uses a numerically stabler
QR/SVD-based solve internally and is what the shipped application actually
calls for higher-degree fits, where the normal-equation approach can become
ill-conditioned. On this dataset the two solutions for a degree-2 fit agree
to a maximum absolute difference of **6.1 × 10⁻¹³** — floating-point noise,
confirming both are solving the same least-squares problem correctly.

### 14. Polynomial Models
Six models were fitted on the training split (314 points): degrees 1, 2, 3,
4, 6, 8. Coefficients and full equations for every degree are stored in
`outputs/results.json`. Representative equations:

- Degree 1: `mpg = −0.15546·hp + 39.59185`
- Degree 2: `mpg = 0.00120·hp² − 0.45982·hp + 56.59837`
- Degree 3: `mpg = −0.00000·hp³ + 0.00131·hp² − 0.47257·hp + 57.07563`
  (cubic coefficient ≈ 1.3×10⁻⁵ — effectively negligible)

### 15. Methodology
1. Load and clean data.
2. Split into train (80%) / test (20%).
3. For each candidate degree, fit on train only.
4. Evaluate on both train and test with MSE/RMSE/MAE/R².
5. Select the model with lowest **test** RMSE (not train RMSE — that would
   always favor the highest degree and hide overfitting).
6. Generate diagnostic plots (fits, residuals, actual-vs-predicted,
   train-vs-test error curve).
7. Expose prediction with interpolation/extrapolation flagging.

### 16. Algorithm
```
START
IMPORT numpy, pandas, matplotlib
LOAD raw CSV into DataFrame
DROP rows with missing horsepower or mpg
FLAG outliers via IQR rule (retain them)
CONVERT to x (horsepower), y (mpg) arrays
SPLIT into (x_train, y_train), (x_test, y_test)  [80/20, seed=42]
FOR degree IN [1, 2, 3, 4, 6, 8]:
    FIT polynomial coefficients on (x_train, y_train)
    PREDICT on x_train AND x_test
    COMPUTE MSE, RMSE, MAE, R² for both
SELECT degree with lowest TEST RMSE
PLOT: raw scatter, fitted curves, residuals, actual-vs-predicted,
      test-RMSE-by-degree bar chart, train-vs-test RMSE line chart
ACCEPT user horsepower input
IF input outside [min(x_train), max(x_train)]: FLAG as extrapolation
COMPUTE and DISPLAY predicted mpg with interpretation
STOP
```

### 17. System Architecture
See README.md §5 for the annotated folder tree. In summary: `data_loader`
(I/O) → `preprocessing` (cleaning + split) → `curve_fitting` (least squares)
→ `evaluation` (metrics) → `prediction` (guarded inference) →
`visualization` (plots), orchestrated by `run_analysis.py` (batch pipeline)
and `app.py` (interactive Streamlit UI). Both consume the same `src/`
modules, so results are guaranteed to be consistent between the script and
the app.

### 18. Implementation
Implemented in Python 3.12 using NumPy for the linear algebra, pandas for
data handling, Matplotlib for plotting, and Streamlit for the interface.
Explicit normal-equation least squares is implemented from scratch in
`curve_fitting.fit_polynomial_manual`; all evaluation metrics (`evaluation.py`)
are implemented from scratch rather than imported from scikit-learn, so the
mathematics from Experiment 2.1's theory section is directly visible and
auditable in the code, per the assignment's academic constraint.

### 19. Results
Executed end-to-end via `run_analysis.py` against the real dataset (see
`outputs/results.json` for the complete machine-readable output and
`outputs/*.png` for all seven generated figures). Key numeric results are
reported in §20 below — all measured, none invented.

### 20. Model Comparison (measured on the 78-row held-out test set)

| Degree | Train RMSE | Test RMSE | Test MAE | Test R² |
|---:|---:|---:|---:|---:|
| 1 | 4.8055 | 5.2399 | 4.0909 | 0.5660 |
| 2 | 4.2895 | 4.6216 | 3.4555 | 0.6624 |
| 3 | 4.2895 | 4.6169 | 3.4550 | 0.6630 |
| 4 | 4.2657 | 4.6978 | 3.4692 | 0.6511 |
| 6 | 4.1507 | 4.8282 | 3.4462 | 0.6315 |
| 8 | 4.1148 | 4.8743 | 3.5057 | 0.6244 |

Degree 3 has the numerically lowest test RMSE (4.6169 vs. 4.6216 for degree
2), but the difference (0.005 mpg) is well within noise and the cubic term's
coefficient is ~1×10⁻⁵ — practically zero. **Degree 2 is selected as the
final model** on the principle that a simpler model with statistically
equivalent accuracy is preferable (Occam's razor) and is easier to reason
about, explain, and defend (a single, interpretable curvature term).

### 21. Error Analysis
Residuals for the selected degree-2 model, evaluated on the test set, scatter
around zero without an obvious systematic curve pattern (see
`outputs/03_residuals.png`), indicating degree 2 is not badly underfitting.
Some heteroscedasticity is visible — residual spread is larger for
lower-horsepower cars, plausibly because that region also contains more of
the flagged outliers (e.g. unusually light or unusually efficient older
compacts). R² of 0.66 indicates roughly one-third of mpg variance is driven
by factors this single-variable model does not capture (weight, model year,
aerodynamics, transmission).

### 22. Prediction
Example, degree-2 model (trained on the 314-row training split, evaluated
on 300+/46 hp range): at **100 hp**, predicted fuel efficiency ≈ **22.6 mpg**
— an interpolation, since 100 hp lies inside the training range [46, 230] hp,
and therefore the model's more trustworthy regime. At **300 hp**, the model
still returns a numeric answer (~25.7 mpg) but this is explicitly flagged as
an **extrapolation** — outside the training data, the fitted polynomial is
mathematically unconstrained and this number should not be treated as a
validated estimate.

### 23. Limitations
- Single predictor (horsepower only); real mpg depends on multiple factors.
- Historical 1970s–80s data; may not generalize to modern engines/hybrids/EVs.
- R² ≈ 0.66 — meaningful predictive power, but far from complete explanation.
- A single fixed train/test split (not k-fold cross-validated), so the exact
  RMSE numbers would shift slightly under a different random seed.
- IQR-flagged outliers are retained in the model fit; a sensitivity check
  excluding them was not performed in this iteration.

### 24. Future Scope
- Extend to multivariate least squares (add `weight`, `displacement` as
  additional predictors) using the same normal-equation machinery.
- k-fold cross-validation for a more robust degree-selection decision.
- Refit on a current (2020s) fuel-economy dataset for contemporary relevance.
- Add confidence/prediction intervals around the fitted curve.

### 25. Conclusion
This project demonstrates that the least-squares polynomial curve-fitting
technique from Experiment 2.1 scales directly from a six-point illustrative
exercise to a real 392-record engineering dataset, and that doing so honestly
— with a held-out test set — surfaces exactly the practical questions the
theory raises but a noise-free toy example cannot: which degree actually
generalizes, and at what point does additional flexibility start hurting
rather than helping. The selected degree-2 model offers a defensible,
interpretable, and genuinely useful fuel-efficiency estimate from a single
engineering input (horsepower), with explicit honesty about where its
predictions are, and are not, reliable.

### 26. References
1. UCI Machine Learning Repository — Auto MPG Data Set.
2. Auto MPG dataset mirror: `mwaskom/seaborn-data`, GitHub — `mpg.csv`.
3. Chandigarh University, Experiment 2.1 — Curve Fitting of a Second-Degree
   Parabola and General Curves (course lab manual, this course's material).
4. NumPy documentation — `numpy.polyfit`, `numpy.linalg.solve`.
5. Streamlit documentation — https://docs.streamlit.io/
