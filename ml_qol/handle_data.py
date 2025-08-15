import pandas as pd
import numpy as np


def handle_data(
    df: pd.DataFrame, date_col: str | None = None, log_cols: list[str] = []
) -> pd.DataFrame:
    df = df.copy()

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])

    df[cat_cols] = df[cat_cols].fillna("Nan")

    # Ensure dtype is category for LGBM
    df[cat_cols] = df[cat_cols].astype("category")

    df[log_cols] = np.log1p(df[log_cols])

    return df
