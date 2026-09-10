# Fuel Efficiency Predictor — Polynomial Curve Fitting (Least Squares)

An extension of **Chandigarh University Experiment 2.1** ("Curve fitting of a
second-degree parabola and general curves") from a 6-point toy demo into a
real, deployable engineering tool: predicting a car's **fuel efficiency
(mpg)** from its **engine horsepower**, fitted on a real 392-record dataset,
with honest train/test evaluation and an overfitting diagnostic built in.

## 1. The real-world problem

**Who uses this:** a car buyer comparing engines, or an automotive/mechanical
engineer doing a first-pass estimate of fuel efficiency for a planned engine
horsepower rating, before a physical prototype or dyno test exists.

**x (independent variable):** Horsepower (hp)
**y (dependent variable):** Fuel efficiency (miles per US gallon)

Horsepower vs mpg is a genuinely non-linear engineering relationship: fuel
efficiency drops steeply as horsepower rises from small economy engines, then
the marginal penalty shrinks at the high end — exactly the diminishing-returns
shape a 2nd/3rd-degree polynomial is suited to, and exactly why a straight
line under-fits this data (see `outputs/01_raw_scatter.png`).

## 2. Dataset

- **Source:** Auto MPG dataset, originally UCI Machine Learning Repository /
  StatLib (Carnegie Mellon, Ford Motor Co. study, cars from model years
  1970–1982). Retrieved from the public GitHub mirror
  `mwaskom/seaborn-data/mpg.csv` — a stable, widely-cited copy of the exact
  UCI dataset used in hundreds of statistics/ML textbooks.
- **Rows:** 398 raw records → **392 usable** after dropping 6 rows with
  missing `horsepower`.
- **Fields used:** `horsepower` (hp), `mpg` (miles/US gallon). Other columns
  (`cylinders`, `weight`, `displacement`, `model_year`, `origin`, `name`) are
  available in the Data Explorer tab but not part of the fitted model, to
  keep the curve-fitting concept from Experiment 2.1 (single x → single y)
  intact rather than quietly turning this into multivariate regression.
- **Outliers:** 10 points flagged by the IQR rule on horsepower (mostly
  high-performance 1970s cars) — flagged, not deleted, and visible in the
  scatter plot as genuine, valid data.
- **Units:** horsepower (hp, SAE gross/net depending on year — a real
  limitation, see below); mpg (US gallons, not imperial).

## 3. What the code actually demonstrates (mapped to Experiment 2.1)

| Experiment 2.1 concept | Where it lives here |
|---|---|
| `y = ax² + bx + c` least squares parabola | `src/curve_fitting.fit_polynomial_manual` (explicit normal-equation solve) and `fit_polynomial_numpy` (numpy.polyfit, matches to 1e-12) |
| General/higher-degree polynomial | Same functions, `degree` parameter (tested up to degree 8) |
| Predicted vs observed plot | `src/visualization.plot_fits`, `plot_actual_vs_predicted` |
| — (new, not in the original experiment) | Train/test split, residual analysis, MSE/RMSE/MAE/R², interpolation-vs-extrapolation guardrails, overfitting diagnostic |

The manual normal-equation solver (`(XᵀX)c = Xᵀy`) is verified against
`numpy.polyfit` in `tests/` — on the real dataset the two agree to within
**6.1 × 10⁻¹³** (floating-point noise), confirming the "least squares" theory
from the experiment is exactly what's being computed.

## 4. Actual results (from `outputs/results.json` — real, computed, not invented)

Split: 314 training cars / 78 held-out test cars (80/20, seed=42).

| Degree | Train RMSE (mpg) | Test RMSE (mpg) | Test R² |
|---:|---:|---:|---:|
| 1 (straight line) | 4.81 | 5.24 | 0.566 |
| **2 (parabola)** | 4.29 | **4.62** | **0.662** |
| 3 (cubic) | 4.29 | 4.62 | 0.663 |
| 4 | 4.27 | 4.70 | 0.651 |
| 6 | 4.15 | 4.83 | 0.631 |
| 8 | 4.11 | 4.87 | 0.624 |

**Selected model: degree 2 (the parabola).** Degree 3 is statistically
indistinguishable from degree 2 (its cubic coefficient is ~1e-5, essentially
zero) — Occam's razor favours the simpler model. From degree 4 onward, train
error keeps falling while **test error gets worse** — textbook overfitting,
visualised in `outputs/06_train_vs_test_overfitting.png`, where the degree-8
curve visibly wiggles at both ends of the horsepower range while degree
2 and 3 stay smooth and nearly overlap (`outputs/02_fits_comparison.png`).

**Fitted equation (degree 2, on the full 392-point cleaned dataset):**
`mpg = 0.00120·hp² − 0.45982·hp + 56.59837`

**Example prediction:** at **100 hp**, the degree-2 model predicts
**≈ 22.6 mpg** (interpolation — 100 hp is inside the training range of
46–230 hp, so this is the model's reliable regime). At **300 hp** the model
still returns a number (~25.7 mpg) but this is flagged as **extrapolation** —
outside the training range, the polynomial is mathematically unconstrained
and the app explicitly warns the user rather than presenting it as fact.

## 5. Architecture

```
mpg-project/
├── data/
│   └── mpg.csv                  # real dataset (392 usable rows)
├── src/
│   ├── data_loader.py           # reads the CSV, validates expected columns
│   ├── preprocessing.py         # drop missing, IQR outlier flagging, train/test split
│   ├── curve_fitting.py         # manual normal-equation solver + numpy.polyfit wrapper
│   ├── evaluation.py            # MSE, RMSE, MAE, R², residuals — implemented from scratch
│   ├── prediction.py            # predict + interpolation/extrapolation guardrail
│   └── visualization.py         # every Matplotlib figure used by the app/scripts
├── tests/
│   └── test_curve_fitting.py    # 14 unit tests (all passing) — see §7
├── outputs/                     # generated plots + results.json (regenerate via run_analysis.py)
├── run_analysis.py              # end-to-end pipeline: load → clean → split → fit → evaluate → plot
├── app.py                       # Streamlit application (6 tabs, see §6)
├── requirements.txt
└── README.md
```

**File responsibilities, one line each:**
- `data_loader.py` — I/O only, nothing else, so it's trivially mockable in tests.
- `preprocessing.py` — turns messy raw data into clean `(x, y)` NumPy arrays; owns the train/test split.
- `curve_fitting.py` — the actual least-squares mathematics (Experiment 2.1's core).
- `evaluation.py` — scores a fit; used identically for train and test sets.
- `prediction.py` — the "ask a question, get an honest answer" layer.
- `visualization.py` — pure plotting functions, no business logic, so they're reusable from both the CLI script and Streamlit.

## 6. Running it

```bash
pip install -r requirements.txt

# Regenerate all plots + results.json from the real dataset:
python run_analysis.py

# Run the interactive app:
streamlit run app.py

# Run the test suite:
python -m pytest tests/ -v
```

The Streamlit app has 6 tabs: **Dashboard** (project + dataset summary),
**Data Explorer** (filterable raw table + stats), **Curve Fitting**
(pick degrees, see fitted equations), **Model Comparison** (the metrics
table + overfitting diagnostic plot above), **Prediction** (enter a
horsepower value, get mpg + interpolation/extrapolation flag), and
**Residual Analysis**.

## 7. Testing

`tests/test_curve_fitting.py` — 14 tests, all passing:
- Manual least-squares solver matches a known noise-free parabola exactly, and matches `numpy.polyfit`.
- Metrics are 0/1 for a perfect fit; R² is 0 when predicting the mean; R² doesn't crash on constant `y`.
- Missing-value rows are dropped correctly; outliers are flagged, not deleted.
- Train/test split produces correct sizes with zero overlap.
- Prediction correctly flags interpolation vs extrapolation.
- Mismatched-length inputs and under-determined fits (more polynomial terms than data points) raise clear errors instead of silently returning garbage.

## 8. Limitations (stated honestly, not hidden)

- Single-variable model: real fuel efficiency also depends on weight,
  aerodynamics, transmission, era/emissions regulations — this project
  isolates horsepower deliberately to stay a direct extension of the
  single-variable curve-fitting experiment, not because horsepower is the
  only factor.
- Historical data (1970s–80s US cars): predictions describe *that* population
  of vehicles, not necessarily a 2026 hybrid or EV.
- R² ≈ 0.66 means roughly a third of the variance in mpg is **not** explained
  by horsepower alone — a real reminder that R² alone doesn't certify a model
  is "good enough" for a given decision.
- Extrapolation beyond ~230 hp is explicitly flagged as unreliable by the app.

## 9. Future extensions

- Add `weight` as a second predictor (multivariate least squares — a natural
  next step mathematically, same normal-equation machinery, one more column
  in the design matrix).
- k-fold cross-validation instead of a single train/test split, for a more
  robust degree-selection decision.
- Swap in a modern dataset (e.g. current EPA fuel-economy data) to make
  predictions applicable to today's vehicles.
