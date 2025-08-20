import pandas as pd
import numpy as np


def handle_data(
    df: pd.DataFrame, date_col: str | None = None, log_cols: list[str] = []
) -> pd.DataFrame:
    df = df.copy()

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

    # Ensure dtype is category for accessing .cat and for LGBM training
    df[cat_cols] = df[cat_cols].astype("category")

    # df[cat_cols] = df[cat_cols].fillna("Nan")

    for col in cat_cols:
        df[col] = df[col].cat.add_categories("Nan")
        df[col] = df[col].fillna("Nan")

    df[log_cols] = np.log1p(df[log_cols])

    return df
