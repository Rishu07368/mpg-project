"""
preprocessing.py
-----------------
Cleans the raw DataFrame and prepares (x, y) NumPy arrays for curve fitting.

Design choice for this project:
    x = horsepower   (independent variable, hp)
    y = mpg           (dependent variable, miles per US gallon)

Why horsepower -> mpg?
    - There is a well-documented, genuinely non-linear engineering
      relationship: fuel efficiency drops sharply as horsepower rises from
      small engines, then the marginal penalty shrinks at the high end
      (diminishing-returns curve) — exactly the shape a 2nd/3rd degree
      polynomial is suited to capture, and exactly why a straight line
      under-fits this data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    missing_dropped: int
    outliers_flagged: int
    outlier_indices: list


def clean_data(df: pd.DataFrame, x_col: str = "horsepower", y_col: str = "mpg"):
    """Drop missing values in the modeling columns and flag (but do not
    silently delete) outliers using the IQR rule, since curve-fitting
    experiments should let a student SEE outliers rather than have them
    vanish invisibly.

    Returns
    -------
    df_clean : pd.DataFrame  (missing rows removed, outlier flag column added)
    report   : CleaningReport
    """
    rows_before = len(df)

    working = df[[x_col, y_col]].copy()
    working["_orig_index"] = df.index
    missing_mask = working[[x_col, y_col]].isna().any(axis=1)
    missing_dropped = int(missing_mask.sum())
    working = working.loc[~missing_mask].reset_index(drop=True)

    # IQR outlier flagging on the independent variable (horsepower), since
    # a handful of very high-horsepower muscle cars sit far from the bulk
    # of the data and are worth flagging explicitly for discussion.
    q1, q3 = working[x_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (working[x_col] < lower) | (working[x_col] > upper)
    working["is_outlier"] = outlier_mask

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(working),
        missing_dropped=missing_dropped,
        outliers_flagged=int(outlier_mask.sum()),
        outlier_indices=working.loc[outlier_mask, "_orig_index"].tolist(),
    )
    return working, report


def train_test_split_manual(x: np.ndarray, y: np.ndarray, test_fraction: float = 0.2, seed: int = 42):
    """A minimal, dependency-free train/test split (shuffled).

    We hold out a test set specifically so overfitting can be demonstrated
    honestly: a high-degree polynomial can make TRAIN error look great while
    TEST error gets worse — that gap is the whole point of the experiment.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])


def get_xy(df_clean: pd.DataFrame, x_col: str = "horsepower", y_col: str = "mpg", drop_outliers: bool = False):
    d = df_clean if not drop_outliers else df_clean.loc[~df_clean["is_outlier"]]
    return d[x_col].to_numpy(dtype=float), d[y_col].to_numpy(dtype=float)
