from .types import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)
from .model import train_model, default_params
import pandas as pd
import numpy as np
import warnings


def fold_train(
    model_type: ModelNameType = "catboost",
    task: TaskType = "regression",
    params: dict = default_params,
    data: pd.DataFrame | None = None,
    target_col: str = "target",
    metric: str | None = None,
    verbose: int = 100,
    early_stop: int = 500,
    random_state: int | None = 42,
):
    if data is None:
        raise ValueError("dataset not found")

    data = data.reset_index(drop=True)

    if task == "classification":
        X = data.drop(columns=[target_col])
        y = data[target_col]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split = skf.split(X, y)

        # Warn if invalid metric provided and set classification default
        if metric in ("mse", "mae"):
            warnings.warn(
                f"{metric} is not a valid metric, using default metric 'accuracy' instead"
            )
            metric = "accuracy"

        # Set classification default metric
        if metric is None:
            metric = "accuracy"

    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        split = kf.split(data)

        # Warn if invalid metric provided and set regression default
        if metric in ("accuracy", "f1"):
            warnings.warn(
                f"{metric} is not a valid metric, using default metric 'mse' instead"
            )
            metric = "accuracy"

        # Set regression default
        if metric is None:
            metric = "mse"

    metric_map = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "accuracy": accuracy_score,
        "f1": f1_score,
    }

    metric_fn = metric_map[metric]

    model_list = []
    eval_scores = []

    for train_idx, valid_idx in split:
        train_df = data.loc[train_idx].copy()
        valid_df = data.loc[valid_idx].copy()

        model = train_model(
            model_type=model_type,
            task=task,
            params=params,
            train_data=train_df,
            valid_data=valid_df,
            target_col=target_col,
            verbose=verbose,
            early_stop=early_stop,
            random_state=random_state,
        )

        valid_preds = model.predict(valid_df.drop(columns=target_col))
        eval_score = metric_fn(valid_df[target_col], valid_preds)

        model_list.append(model)
        eval_scores.append(eval_score)

    mean_score = np.array(eval_scores).mean()

    print(f"\nMean validation {metric} score: {mean_score}")

    return model_list


def get_fold_preds(models, test_df: pd.DataFrame) -> np.ndarray:
    all_preds = []

    for model in models:
        preds = model.predict(test_df)
        all_preds.append(preds)

    mean_preds = np.array(all_preds).mean(axis=0)

    return mean_preds


def get_ensemble_preds(preds_list, weights: list[float] | None = None):
    if weights is not None and len(preds_list) != len(weights):
        raise ValueError(f"number of preds_list and number of weights do not match")

    preds_list = np.array(preds_list)
    weighted_preds = np.average(preds_list, weights=weights)

    return weighted_preds
