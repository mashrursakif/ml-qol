from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import warnings

from .model import train_model, default_params

from typing import Literal
from .types import *


def fold_train(
    model_type: ModelNameType = "catboost",
    task: TaskType = "regression",
    params: dict = default_params,
    data: pd.DataFrame | None = None,
    target_col: str = "target",
    n_splits: int = 5,
    metric: Literal["mse", "mae", "accuracy", "f1"] | None = None,
    verbose: int = 100,
    early_stop: int = 500,
    random_state: int | None = 42,
):
    if data is None:
        raise ValueError("dataset not found")

    # reset index to prevent indexing errors
    data = data.copy()
    data = data.reset_index(drop=True)

    if task == "classification":
        X = data.drop(columns=[target_col])
        y = data[target_col]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split = skf.split(X, y)

    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split = kf.split(data)

    model_list = []
    eval_scores = []

    for fold_num, (train_idx, valid_idx) in enumerate(split):
        print(f"Running Fold: {fold_num + 1}")

        train_df = data.loc[train_idx].copy()
        valid_df = data.loc[valid_idx].copy()

        model = train_model(
            model_type=model_type,
            task=task,
            params=params,
            train_data=train_df,
            valid_data=valid_df,
            target_col=target_col,
            metric=metric,
            verbose=verbose,
            early_stop=early_stop,
            random_state=random_state,
        )

        model_list.append(model)
        eval_scores.append(model.eval_score)

    mean_score = np.array(eval_scores).mean()

    print(f"\nMean validation score: {mean_score}")

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
    weighted_preds = np.average(preds_list, weights=weights, axis=0)

    return weighted_preds
