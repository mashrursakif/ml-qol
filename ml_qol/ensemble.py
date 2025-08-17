from .types import *
from sklearn.model_selection import KFold, StratifiedKFold
from .model import train_model, default_params
import pandas as pd
import numpy as np

print("DP::", default_params)


def fold_train(
    model_type: ModelNameType = "catboost",
    task: TaskType = "regression",
    params: dict = default_params,
    dataset: pd.DataFrame | None = None,
    target_col: str = "target",
    verbose: int = 100,
    early_stop: int = 500,
    random_state: int | None = 42,
):
    if dataset is None:
        raise ValueError("dataset not found")

    if task == "classification":
        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split = skf.split(X, y)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        split = kf.split(dataset)

    model_list = []

    for train_idx, valid_idx in split:
        train_df = dataset.loc[train_idx].copy()
        valid_df = dataset.loc[valid_idx].copy()

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

        model_list.append(model)

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
