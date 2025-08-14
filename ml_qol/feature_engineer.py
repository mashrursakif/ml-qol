import pandas as pd
import warnings
from itertools import combinations
from sklearn.model_selection import KFold


def expand_date(df, date_col='date_time'):
    df = df.copy()

    if date_col not in df.columns:
        ValueError(f'Column: {date_col} is not found in df')

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        warnings.warn(
            f'Column: {date_col} is not a datetime type, automatically converting datetime')
        df[date_col] = pd.to_datetime(df[date_col])

    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['weekday'] = df[date_col].dt.weekday
    df['is_weekend'] = (df[date_col].dt.weekday) > 4
    df['week_of_year'] = df[date_col].dt.isocalendar().week

    return df


def combine_features(
    df,
    num_features=None,
    cat_features=None,
    # methods=['divide', 'multiply']
):
    df = df.copy()

    if not num_features:
        warnings.warn(
            'num_features not provided, using all numerical features')
        num_features = df.select_dtypes(include=['number']).columns.tolist()

    if not cat_features:
        warnings.warn(
            'cat_features not provided, using all categorical features')
        cat_features = df.select_dtypes(
            include=['category', 'object']).columns.tolist()

    for col1, col2 in combinations(num_features):
        df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
        df[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, pd.NA)

    for col1, col2 in combinations(cat_features):
        df[f'{col1}_{col2}'] = df[col1].astype(
            str) + '_' + df[col2].astype(str)

    return df


def bin_column(df, col, bins=4, labels=None):
    df[f'{col}_binned'] = pd.cut(df[col], bins=bins, labels=labels)
    return df


def target_encode_test(train_df, test_df, target_col='target', cols=None):
    test_df = test_df.copy()
    train_df = train_df.copy()

    if not cols:
        raise ValueError(f'cols not provided')

    for col in cols:
        if train_df[col].dtype.name not in ['object', 'category']:
            warnings.warn(f'{col} is numerical, therefore will be binned')
            train_df = bin_column(train_df, col)
            test_df = bin_column(test_df, col)

        agg_list = ['mean', 'median', 'max', 'min', 'nunique']
        group_df = train_df.groupby(
            col)[target_col].agg(agg_list).reset_index()
        group_df.columns = [col] + [f'{col}_te_{name}' for name in agg_list]

        te_test_df = pd.merge(test_df, group_df, on=col, how='left')

        # Remove bin features added from bin_column
        te_test_df.drop(columns=[f'{col}_binned'], inplace=True)

    return te_test_df


def target_encode(df, target_col='target', cols=None):
    df = df.copy()

    if not cols:
        raise ValueError(f'cols not provided')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    te_df = pd.DataFrame(columns=df.columns)

    for train_idx, valid_idx in kf.split(df):
        train_df = df.loc[train_idx].copy()
        valid_df = df.loc[valid_idx].copy()

        te_valid_df = target_encode_test(train_df, valid_df, target_col, cols)
        te_df = pd.concat([te_df, te_valid_df], ignore_index=True)

    return te_df
