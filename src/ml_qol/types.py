from typing import Literal, Union
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# from xgboost import XGBRegressor, XGBClassifier

ModelNameType = Literal["catboost", "lightgbm", "xgboost"]
# ModelType = (
#     CatBoostRegressor
#     | CatBoostClassifier
#     | LGBMRegressor
#     | LGBMClassifier
#     # | XGBRegressor
#     # | XGBClassifier
# )


TaskType = Literal["regression", "classification"]
