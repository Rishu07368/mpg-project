"""
data_loader.py
---------------
Responsible for ONE thing: reading the raw dataset off disk and handing back
a pandas DataFrame. No cleaning, no modeling happens here — that keeps the
loader trivial to test and re-use (e.g. Streamlit's cache decorator wraps
exactly this function).

Dataset: Auto MPG (398 records of 1970s-1980s US cars), sourced from the
seaborn-data GitHub mirror (originally UCI Machine Learning Repository /
StatLib, Ford Motor Co. / Carnegie Mellon 1983 study).
"""

import pandas as pd
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "mpg.csv"


def load_mpg_data(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the raw Auto MPG dataset.

    Parameters
    ----------
    path : Path
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset (may contain missing values).
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected the Auto MPG CSV "
            "(columns: mpg, cylinders, displacement, horsepower, weight, "
            "acceleration, model_year, origin, name)."
        )
    df = pd.read_csv(path)
    expected_cols = {"mpg", "horsepower", "weight", "cylinders"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
    return df


if __name__ == "__main__":
    data = load_mpg_data()
    print(f"Loaded {len(data)} rows, {data.isna().sum().sum()} missing cells")
    print(data.head())
