# ML QOL

**ML QOL** is a Python package that provides helper functions and quality-of-life features for machine learning tasks

## Features

- Automated hyperparameter mapping for different models such as CatBoost and LightGBM
- Data handling functions for managing dates, NaN values, and more
- Feature engineering functions such as combining features together or adding target encoded features
- Fast and easy way to train and compare different models and their performance, e.g, feature importance, confusion matrix
- Perform folded training and gathering averaged predictions
- Perform weighted ensembling with different types of models

## Dependencies

This package relies on the following Python libraries:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [lightgbm](https://github.com/microsoft/LightGBM)
- [catboost](https://catboost.ai/)
- [xgboost](https://github.com/dmlc/xgboost)
- [matplotlib](https://matplotlib.org/)
- [seabron](https://seaborn.pydata.org/)

You can install them via pip:

```bash
pip install pandas numpy scikit-learn lightgbm catboost xgboost matplotlib seaborn
```

## Installation

**Using pip**

```bash
pip install ml-qol
```

## Quick Start

```python
from ml_qol import train_model

# Train a model
model, model_info = train_model('lightgbm', 'regression', train_data=train_df, target_col='price')

# Show feature importances
model.plot_importance()

# Use for inference
predictions = model.predict(test_df)
print(predictions)
```

## Resources

- GitHub repository: (https://github.com/mashrursakif/ml-qol)
<!-- - Documentation: Currently unavailable -->

### License

#### MIT

## Author

Developed by Mashrur Sakif Souherdo - [GitHub](https://github.com/mashrursakif)
