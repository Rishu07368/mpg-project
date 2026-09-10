# Viva Questions & Answers

### Basics

**1. What is curve fitting?**
Finding a mathematical function `y = f(x)` that best represents the
relationship between observed data points.

**2. Why is curve fitting used in data analysis?**
It converts scattered, noisy observations into a compact model that can
summarize trends, be interpreted, and predict new values.

**3. What is the least squares method?**
A technique that chooses the function's parameters so the sum of squared
differences between observed and predicted values (residuals) is minimized.

**4. Why squares and not just the sum of residuals?**
Squaring makes all deviations positive (so they don't cancel out) and
penalizes larger errors more heavily; it also produces a solvable linear
system (the normal equations).

**5. What is a second-degree polynomial?**
A quadratic: `y = ax² + bx + c`, whose graph is a parabola.

**6. What is a general polynomial curve?**
`y = c₀xᵈ + c₁x^(d−1) + … + c_d` for any degree `d`; degree 2 is a special
case.

**7. What is the difference between linear and polynomial curve fitting?**
Linear fitting restricts the model to a straight line (`y = mx + c`);
polynomial fitting allows curvature by adding higher powers of `x`.

**8. Which Python function did you use for curve fitting?**
`numpy.polyfit(x, y, degree)`, plus a hand-written normal-equation solver
(`fit_polynomial_manual`) for verification.

**9. Which Python library did you use for numerical computation?**
NumPy — for the Vandermonde matrix, linear algebra solve, and evaluation.

**10. Which library did you use for plotting?**
Matplotlib (all charts), wrapped by a Streamlit interface for interactivity.

### About this specific project

**11. What real-world problem does your project solve?**
Predicting a car's fuel efficiency (mpg) from its engine horsepower, useful
to a buyer comparing engines or an engineer doing an early efficiency
estimate before a physical prototype exists.

**12. What are your independent and dependent variables?**
Independent: horsepower (hp). Dependent: fuel efficiency (miles per US
gallon).

**13. What dataset did you use, and where is it from?**
The Auto MPG dataset (UCI Machine Learning Repository / StatLib, 1970–1982
US cars), retrieved from the public `seaborn-data` GitHub mirror — 398 raw
rows, 392 after cleaning.

**14. How did you handle missing data?**
Six rows had a missing horsepower value; they were dropped rather than
imputed, since inventing an engineering measurement would be misleading.

**15. How did you handle outliers?**
Flagged them with the IQR rule (`< Q1−1.5·IQR` or `> Q3+1.5·IQR` on
horsepower) but did not delete them — they represent genuine high-performance
vehicles, and hiding them would misrepresent the data.

**16. What degree did you finally select, and why?**
Degree 2. Degree 3 has a marginally lower test RMSE (4.617 vs 4.622 mpg),
but the difference is negligible and its cubic coefficient is ~10⁻⁵ —
effectively zero — so the simpler quadratic is preferred (Occam's razor).

**17. Why not just use the highest-degree polynomial for best accuracy?**
Because "best accuracy" must be measured on data the model didn't train on.
Past degree 3 here, training error kept falling but test error rose — the
model was fitting noise, not the underlying trend (overfitting).

**18. What does your manual least-squares solver do differently from numpy.polyfit?**
It builds the Vandermonde design matrix explicitly and solves the normal
equations `(XᵀX)c = Xᵀy` directly with `numpy.linalg.solve`. `numpy.polyfit`
solves the same underlying least-squares problem but uses a more
numerically stable QR/SVD-based method internally, which matters more at
higher degrees. On this dataset the two agree to within 6×10⁻¹³.

### Mathematics

**19. What does numpy.polyfit() return?**
An array of polynomial coefficients, ordered highest power to lowest
(constant term last).

**20. What is the purpose of numpy.polyval()?**
It evaluates a polynomial (given its coefficients) at one or more `x`
values, i.e. it computes predictions from a fitted model.

**21. Why do we use linspace() in plotting?**
To generate a dense, evenly-spaced set of `x` values so the fitted curve
renders as a smooth line rather than a jagged line between the (sparser)
actual data points.

**22. What is meant by polynomial degree?**
The highest exponent of `x` present in the function; it controls how many
bends/inflection points the curve can have.

**23. What happens if the degree of polynomial is increased?**
The curve gains flexibility and can fit the training points more closely
(training error keeps dropping), but beyond some point it starts fitting
noise rather than signal, and error on new data worsens.

**24. What is overfitting in curve fitting?**
When a model fits the training data (including its noise) too closely,
capturing patterns that don't generalize — visible here as training RMSE
still falling at degree 8 while test RMSE rises.

**25. What is underfitting?**
When a model is too simple to capture the real relationship — here, the
degree-1 (straight-line) model underfits, since it can't represent the
diminishing-returns curve visible in the data, and has the worst test RMSE.

**26. How do you check whether a curve fits the data well?**
Compare error metrics (MSE/RMSE/MAE/R²) on a held-out test set, and inspect
the residual plot for random scatter around zero rather than a systematic
pattern.

**27. What is a residual?**
The difference between an observed value and the model's predicted value:
`residual = y_actual − y_predicted`.

**28. What does MSE measure?**
The mean of squared residuals — an error measure that penalizes large
mistakes disproportionately.

**29. What is RMSE, and why is it more interpretable than MSE?**
The square root of MSE; unlike MSE it's in the same units as `y` (mpg here),
so "RMSE = 4.62 mpg" is directly meaningful.

**30. What is MAE, and how does it differ from RMSE?**
The mean of absolute (unsquared) residuals; it weighs all errors linearly,
so it's less sensitive to a few large outlier errors than RMSE is.

**31. What does R² mean?**
The proportion of variance in `y` explained by the model, relative to a
baseline of always predicting the mean of `y`. Here R² ≈ 0.66 means the
model explains about two-thirds of the variation in mpg.

**32. Why is R² not sufficient by itself to judge a model?**
A high R² can still hide overfitting, doesn't reveal *where* errors are
large, doesn't indicate whether residuals are systematically biased, and
can be inflated simply by adding more polynomial terms even if they don't
generalize — which is exactly why this project also reports RMSE, MAE, and
a held-out test comparison, not R² alone.

**33. What is the difference between interpolation and curve fitting?**
Curve fitting builds an approximate best-fit function from noisy data (the
function generally does not pass exactly through every point); classical
interpolation constructs a function that passes exactly through every given
point (e.g., polynomial interpolation through n points using degree n−1).

**34. What is the difference between interpolation and extrapolation (as used in your app)?**
Interpolation predicts an `x` value inside the range seen during training
(reliable); extrapolation predicts outside that range, where the polynomial
is unconstrained and can behave unrealistically — the app explicitly flags
which case a user's query falls into.

**35. Why can extrapolation be unreliable, concretely, for your model?**
Because polynomials are not bounded — outside the trained horsepower range
(46–230 hp), a fitted curve can curve upward or downward with no real-world
justification, purely as an artifact of the polynomial's shape.

### Broader / critical-thinking

**36. What are the applications of curve fitting in engineering?**
Trend prediction, sensor calibration curves, forecasting demand or
consumption, correlating design parameters with performance outcomes, and
any situation converting scattered measurements into a usable model.

**37. What are the limitations of your system?**
Single-predictor model (horsepower only); historical 1970s–80s data that may
not generalize to modern vehicles; R² of 0.66 leaves real unexplained
variance; a single fixed train/test split rather than cross-validation.

**38. How would you improve this project for production use?**
Add more predictors (weight, displacement) via multivariate least squares;
use k-fold cross-validation for degree selection instead of one split;
retrain on a current dataset; add prediction confidence intervals.

**39. Why did you choose a train/test split instead of fitting on all the data?**
Fitting and evaluating on the same data always favors higher-degree
polynomials and cannot detect overfitting — a model can achieve near-zero
training error by memorizing noise. A held-out test set answers the
question that actually matters: does the model generalize to new data?

**40. What would happen if you fit a degree-391 polynomial on your 392 training-ish points?**
It could (numerically permitting) pass through nearly every training point
exactly, driving training error toward zero — but it would oscillate wildly
between points and produce extremely poor, often nonsensical, predictions
on any new data, an extreme case of overfitting.
