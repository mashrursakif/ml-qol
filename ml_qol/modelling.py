import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
import matplotlib.pyplot as plt
# import seaborn as sns

# Models
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
# import xgboost as xgb

all_models = {
  'catboost': {
		'regression': CatBoostRegressor,
		'classification': CatBoostClassifier,
	},
	'lightgbm': {
   	'regression': LGBMRegressor,
		'classification': LGBMClassifier,
	}
}

param_map = {
	'catboost': {
		'depth': 'depth',
		'learning_rate': 'learning_rate',
		'iterations': 'iterations',
		'loss_function': 'loss_function',
		'metric': 'eval_metric',
		'reg_2': 'l2_leaf_reg',
  	# 'border_count': 'border_count',
   	'subsample': 'subsample',
		'device': 'task_type',
		'seed': 'random_seed',
	},
	'lightgbm': {
		'depth': 'max_depth',
		'learning_rate': 'learning_rate',
		'iterations': 'num_iterations',
		'loss_function': 'objective',
		'metric': 'metric',
		'reg_2': 'lambda_l2',
  	# 'border_count': 'max_bin',
   	'subsample': 'bagging_fraction',
		'device': 'device',
		'seed': 'seed',
	}
}

def map_params(model_type, user_params):
  model_param_map = param_map[model_type]
  
  mapped_params = {}
  for key, value in user_params.items():
    mapped_key = model_param_map.get(key)
    if mapped_key:
      mapped_params[mapped_key] = value
    else:
      warnings.warn(f"Parameter {key} not recognized, and will be ignored")
   
  return mapped_params



default_params = {
	'iterations': 1000,
	'learning_rate': 1e-2,
	'loss_function': 'RMSE',
	'device': 'CPU',
}

def get_model(model_type='catboost', task='regression', params=default_params):
  model_category = all_models.get(model_type)
  if model_category is None:
    raise ValueError(f"Model: {model_type} is not recognized, use one of following: {', '.join(param_map.keys())}")
  
  model_instance = model_category.get(task)
  if model_instance is None:
    raise ValueError(f"Model: {model_type} is not recognized, use one of following: {', '.join(model_category.keys())}")
  
  mapped_params = map_params(model_type, params)
  
  if model_type == 'lightgbm':
    # Remove Info Logs for LGBM
    mapped_params = { **mapped_params, 'verbosity': -1 }
  
  model = model_instance(**mapped_params)
  
  return model



class ModelWrapper:
  def __init__(self, model, model_type, task, X_valid, y_valid):
    self.model = model
    self.model_type = model_type
    self.X_valid = X_valid
    self.y_valid = y_valid
  
  def plot_importance(self, max_num_features=20, figsize=(10, 6)):
    if self.model_type == 'catboost': importance = self.model.get_feature_importance()
    elif self.model_type == 'lightgbm': importance = self.model.feature_importances_
    
    feature_names = self.X_valid.columns.tolist()
    
    importance_df = pd.DataFrame({
      'feature': feature_names,
			'importance': importance
		}).sort_values(by='importance', ascending=True)
    
    importance_df = importance_df.head(max_num_features)
    
    plt.figure(figsize=figsize)
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()
  
  def __getattr__(self, attr):
    return getattr(self.model, attr)
  


def train_model(model_type='catboost', task='regression', params=default_params, dataset=None, target='target', verbose=100, early_stop=500):
  model = get_model(model_type, task, params)
  
  X = dataset.drop(columns=[target])
  y = dataset[target]
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
  
  cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
  
  if model_type == 'catboost':
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)
    
    model.fit(train_pool, eval_set=[train_pool, valid_pool], use_best_model=True, verbose=verbose, early_stopping_rounds=early_stop)
    
  elif model_type == 'lightgbm':
    model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        callbacks=[
		      lgb.early_stopping(stopping_rounds=early_stop),
		      lgb.log_evaluation(period=verbose)
	      ])
  
  model = ModelWrapper(model, model_type, task, X_valid, y_valid)
  return model