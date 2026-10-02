# Week 5 — Machine Learning Algorithms and Model Evaluation

## Problem
Predict continuous **life expectancy** using:
- year
- population
- log GDP per capita
- continent

`development_group` is excluded to avoid target leakage because it is derived from GDP per capita.

## Models
1. Linear Regression — transparent baseline
2. Random Forest Regression — nonlinear candidate with GridSearchCV tuning

## Evaluation
- MAE
- RMSE
- R²
- 5-fold cross-validation
- Actual vs predicted plot
- Residual plot
- Residual distribution
- Permutation importance

## Hyperparameter tuning
Grid search varies:
- n_estimators: 100, 200
- max_depth: None, 10, 20
- min_samples_leaf: 1, 2, 4
- max_features: sqrt, 1.0

## Included
- `data/ml_model_dataset.csv`
- `notebook/week5_machine_learning.ipynb`
- `scripts/train_and_evaluate_models.py`
- `outputs/model_comparison.csv`
- `outputs/cross_validation_summary.csv`
- `outputs/best_parameters.csv`
- `outputs/grid_search_results.csv`
- `outputs/test_predictions.csv`
- `outputs/permutation_importance.csv`
- `visualizations/` — five PNGs
- `report/Week_5_Machine_Learning_Report.docx`
- `requirements.txt`

## Source
Gapminder: https://www.gapminder.org/data/
