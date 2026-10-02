import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/ml_model_dataset.csv")
target = "lifeexp"
features = ["year", "pop", "log_gdp_per_capita", "continent"]

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

num_features = ["year", "pop", "log_gdp_per_capita"]
cat_features = ["continent"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_features)
])

# Baseline
linear = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

# Tuned Random Forest
rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 1.0]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    rf, param_grid=param_grid, cv=cv,
    scoring="neg_root_mean_squared_error", n_jobs=-1
)
grid.fit(X_train, y_train)

rf_best = grid.best_estimator_
rf_pred = rf_best.predict(X_test)

def evaluate(y_true, pred):
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, pred)),
        "R2": r2_score(y_true, pred)
    }

print("Linear Regression:", evaluate(y_test, linear_pred))
print("Tuned Random Forest:", evaluate(y_test, rf_pred))
print("Best parameters:", grid.best_params_)
print("Best CV RMSE:", -grid.best_score_)
