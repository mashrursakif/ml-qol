import pandas as pd
import numpy as np
import warnings
from itertools import combinations
from sklearn.model_selection import KFold
from typing import Any

# ignoring this warning since it triggers too many times, cluttering output, and same warning provided in combine_features already
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def expand_date(df: pd.DataFrame, date_col: str = "date_time"):
    df = df.copy()

    if date_col not in df.columns:
        ValueError(f"Column: {date_col} is not found in df")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        warnings.warn(
            f"Column: {date_col} is not a datetime type, automatically converting datetime"
        )
        df[date_col] = pd.to_datetime(df[date_col])

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    df["is_weekend"] = (df[date_col].dt.weekday) > 4
    df["week_of_year"] = df[date_col].dt.isocalendar().week

    return df


def combine_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    num_features: list[str] | None = None,
    cat_features: list[str] | None = None,
    # methods=['divide', 'multiply']
    safety_check: bool = True,
) -> pd.DataFrame:

    # reset index to prevent indexing errors
    df = df.copy()
    df = df.reset_index(drop=True)

    y = None

    if target_col is None:
        warnings.warn("target_col not specified, this may lead to data leakage")
    else:
        y = df[target_col]
        df = df.drop(columns=target_col)

    if not num_features:
        warnings.warn(
            "num_features not provided, using all numerical features available, this may lead to poor performance"
        )
        num_features = df.select_dtypes(include=["number"]).columns.tolist()

    if not cat_features:
        warnings.warn(
            "cat_features not provided, using all categorical features available, this may lead to poor performance"
        )
        cat_features = df.select_dtypes(include=["category", "object"]).columns.tolist()

    if (len(num_features) > 50 or len(cat_features) > 50) and safety_check:
        raise ValueError(
            "More than 50 features selected, this could lead to extremely high memory usage or slow performance. Pass safety_check = False to override this safe check"
        )

    for col1, col2 in combinations(num_features, 2):
        df[f"{col1}_times_{col2}"] = (df[col1] * df[col2]).astype(float)
        df[f"{col1}_div_{col2}"] = (df[col1] / df[col2].replace(0, np.nan)).astype(
            float
        )

    for col1, col2 in combinations(cat_features, 2):
        df[f"{col1}_{col2}"] = (
            df[col1].astype(str) + "_" + df[col2].astype(str)
        ).astype("category")

    if y is not None:
        df[target_col] = y.copy()

    return df


def bin_column(df: pd.DataFrame, col: str, bins: int = 4, labels=None) -> pd.DataFrame:
    df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=labels)
    return df


def target_encode_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "target",
    cols: list[str] | None = None,
) -> pd.DataFrame:

    # reset index to prevent indexing errors
    train_df = train_df.copy()
    train_df = train_df.reset_index(drop=True)

    test_df = test_df.copy()
    test_df = test_df.reset_index(drop=True)

    if not pd.api.types.is_numeric_dtype(train_df[target_col]):
        raise TypeError(f"{target_col} is not numeric")

    if not cols:
        raise ValueError(f"cols not provided")

    for col in cols:
        if train_df[col].dtype.name not in ["object", "category"]:
            # warnings.warn(f"{col} is numerical, therefore will be binned")
            train_df = bin_column(train_df, col)
            test_df = bin_column(test_df, col)

        # agg_list = [np.mean, np.median, np.max, np.min]
        # using Any type as work around for .agg function-string conflict
        agg_list: Any = ["mean", "median", "max", "min"]
        group_df = (
            train_df.groupby(col, observed=False)[target_col]
            .agg(agg_list)
            .reset_index()
        )
        group_df.columns = [col] + [f"{col}_te_{name}" for name in agg_list]

        test_df = pd.merge(test_df, group_df, on=col, how="left")

        # Remove bin features added from bin_column, ignore key error in case _binned columns don't exist
        test_df.drop(columns=[f"{col}_binned"], inplace=True, errors="ignore")

    return test_df


def target_encode(
    df: pd.DataFrame, target_col: str | None = None, cols: list[str] | None = None
) -> pd.DataFrame:

    # reset index to prevent indexing errors
    df = df.copy()
    df = df.reset_index(drop=True)

    if not target_col:
        raise ValueError(f"target_col not provided")

    if not cols:
        raise ValueError(f"cols not provided")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    te_df = pd.DataFrame()

    for train_idx, valid_idx in kf.split(df):
        train_df = df.loc[train_idx].copy()
        valid_df = df.loc[valid_idx].copy()

        te_valid_df = target_encode_test(train_df, valid_df, target_col, cols)
        te_df = pd.concat([te_df, te_valid_df], ignore_index=True)

    return te_df
