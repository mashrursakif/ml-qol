import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)

import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import seaborn as sns

# Models
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb

# import xgboost as xgb

from typing import Literal
from .types import *

all_models = {
    "catboost": {
        "regression": CatBoostRegressor,
        "classification": CatBoostClassifier,
    },
    "lightgbm": {
        "regression": LGBMRegressor,
        "classification": LGBMClassifier,
    },
}

param_map = {
    "catboost": {
        "depth": "depth",
        "learning_rate": "learning_rate",
        "iterations": "iterations",
        "loss_function": "loss_function",
        "metric": "eval_metric",
        "reg_2": "l2_leaf_reg",
        "border_count": "border_count",
        "subsample": "subsample",
        "device": "task_type",
        "seed": "random_seed",
    },
    "lightgbm": {
        "depth": "max_depth",
        "learning_rate": "learning_rate",
        "iterations": "num_iterations",
        "loss_function": "objective",
        "metric": "metric",
        "reg_2": "lambda_l2",
        # 'border_count': 'max_bin',
        "subsample": "bagging_fraction",
        "device": "device",
        "seed": "seed",
    },
}


def map_params(model_type: str, user_params: dict) -> dict:
    model_param_map = param_map[model_type]

    mapped_params = {}
    for key, value in user_params.items():
        if key == "loss_function":
            value = map_loss_functions(value, model_type)

        mapped_key = model_param_map.get(key)
        if mapped_key:
            mapped_params[mapped_key] = value
        else:
            warnings.warn(f"Parameter {key} not recognized, and will be ignored")

    return mapped_params


loss_fn_map = {
    "catboost": {
        "RMSE": "RMSE",
        "MAE": "MAE",
        "binary": "Logloss",
        "cross_entropy": "CrossEntropy",
        "multi_class": "MultiClass",
        "MAPE": "MAPE",
        "quantile": "Quantile",
    },
    "lightgbm": {
        "RMSE": "rmse",
        "MAE": "l1",
        "binary": "binary",
        "cross_entropy": "cross_entropy",
        "multi_class": "multiclass",
        "MAPE": "mape",
        "quantile": "quantile",
    },
}


def map_loss_functions(loss_fn, model_type):
    model_loss_map = loss_fn_map[model_type]

    mapped_loss = model_loss_map.get(loss_fn)

    if mapped_loss:
        return mapped_loss
    else:
        warnings.warn(
            f"mapping for loss function: {loss_fn} not found, will be used as is"
        )
        return loss_fn


default_params = {
    "iterations": 1000,
    "learning_rate": 1e-2,
    "loss_function": "RMSE",
    "device": "CPU",
}


def get_model(
    model_type: ModelNameType = "catboost",
    task: TaskType = "regression",
    params: dict = default_params,
):
    model_category = all_models.get(model_type)

    if model_category is None:
        raise ValueError(
            f"Model: {model_type} is not recognized, use one of following: {', '.join(param_map.keys())}"
        )

    model_instance = model_category.get(task)
    if model_instance is None:
        raise ValueError(
            f"Model: {model_type} is not recognized, use one of following: {', '.join(model_category.keys())}"
        )

    mapped_params = map_params(model_type, params)

    if model_type == "lightgbm":
        # Remove Info Logs for LGBM
        mapped_params = {**mapped_params, "verbosity": -1}

    model = model_instance(**mapped_params)

    return model


class ModelWrapper:

    def __init__(
        self,
        model,
        model_type: ModelNameType,
        task: TaskType,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        valid_preds: np.ndarray,
    ):
        self.model = model
        self.model_type = model_type
        self.task = task
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.valid_preds = valid_preds

    def plot_importance(
        self, max_num_features: int = 20, figsize: tuple[float, float] = (10, 6)
    ):
        if self.model_type == "catboost":
            importance = self.model.get_feature_importance()
        elif self.model_type == "lightgbm":
            importance = self.model.feature_importances_
        else:
            raise ValueError(f"{self.model_type} plot_importance is not supported")

        feature_names = self.X_valid.columns.tolist()

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values(by="importance", ascending=True)

        importance_df = importance_df.head(max_num_features)

        plt.figure(figsize=figsize)
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.show()

    def confusion_matrix(self):
        y_valid, valid_preds = self.y_valid, self.valid_preds
        cm = confusion_matrix(y_valid, valid_preds)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(values_format="d")
        plt.title("Confusion Matrix")
        plt.show()

    def __getattr__(self, attr):
        return getattr(self.model, attr)


metric_map = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "accuracy": accuracy_score,
    "f1": f1_score,
}


def train_model(
    model_type: ModelNameType = "catboost",
    task: TaskType = "regression",
    params: dict = default_params,
    train_data: pd.DataFrame | None = None,
    valid_data: pd.DataFrame | None = None,
    target_col: str = "target",
    metric: Literal["mse", "mae", "accuracy", "f1"] | None = None,
    verbose: int = 100,
    early_stop: int = 500,
    random_state: int | None = 42,
    test_size: float | None = 0.2,
):
    model = get_model(model_type, task, params)

    if train_data is None:
        raise ValueError("train_data is not found")

    if train_data is not None and valid_data is not None:
        X_train = train_data.drop(columns=target_col)
        y_train = train_data[target_col]
        X_valid = valid_data.drop(columns=target_col)
        y_valid = valid_data[target_col]

    else:
        print(
            "valid_data is not explicitly passed, hence performing random split on train_data"
        )
        X = train_data.drop(columns=[target_col])
        y = train_data[target_col]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    cat_cols = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    if model_type == "catboost":
        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

        model.fit(
            train_pool,
            eval_set=[train_pool, valid_pool],
            use_best_model=True,
            verbose=verbose,
            early_stopping_rounds=early_stop,
        )

    elif model_type == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop),
                lgb.log_evaluation(period=verbose),
            ],
        )

    if task == "classification":
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
        # Warn if invalid metric provided and set regression default
        if metric in ("accuracy", "f1"):
            warnings.warn(
                f"{metric} is not a valid metric, using default metric 'mse' instead"
            )
            metric = "mse"

        # Set regression default
        if metric is None:
            metric = "mse"

    metric_fn = metric_map[metric]

    valid_preds = model.predict(X_valid)
    eval_score = metric_fn(y_valid, valid_preds)

    print(f"\nValidation {metric} score: {eval_score}")

    model = ModelWrapper(model, model_type, task, X_valid, y_valid, valid_preds)

    return model
