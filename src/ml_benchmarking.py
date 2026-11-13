"""FloodGuard AI - ML Model Benchmarking.

Trains and compares scikit-learn regression models (Linear Regression,
Random Forest, Gradient Boosting) for flood-level prediction and extracts
feature importances from tree-based models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RANDOM_SEED


def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model.

    Args:
        X_train: Training feature array of shape (n_samples, n_features).
        y_train: Training target array of shape (n_samples,).

    Returns:
        A fitted ``LinearRegression`` estimator.

    Raises:
        RuntimeError: If training fails.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise RuntimeError(f"Linear Regression training failed: {e}") from e


def train_random_forest(X_train, y_train, n_estimators=100):
    """Train a Random Forest Regressor.

    Args:
        X_train: Training feature array of shape (n_samples, n_features).
        y_train: Training target array of shape (n_samples,).
        n_estimators: Number of trees in the forest. Defaults to 100.

    Returns:
        A fitted ``RandomForestRegressor`` estimator.

    Raises:
        RuntimeError: If training fails.
    """
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise RuntimeError(f"Random Forest training failed: {e}") from e


def train_gradient_boosting(X_train, y_train, n_estimators=100):
    """Train a Gradient Boosting Regressor.

    Args:
        X_train: Training feature array of shape (n_samples, n_features).
        y_train: Training target array of shape (n_samples,).
        n_estimators: Number of boosting stages. Defaults to 100.

    Returns:
        A fitted ``GradientBoostingRegressor`` estimator.

    Raises:
        RuntimeError: If training fails.
    """
    try:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise RuntimeError(f"Gradient Boosting training failed: {e}") from e


def evaluate_regression(y_true, y_pred) -> dict:
    """Compute regression evaluation metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        A dict with keys ``'mae'`` (Mean Absolute Error) and ``'rmse'``
        (Root Mean Squared Error).

    Raises:
        ValueError: If inputs have mismatched lengths or are empty.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot evaluate on empty arrays")

    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {'mae': mae, 'rmse': rmse}
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}") from e


def compare_models(X_train, y_train, X_test, y_test):
    """Train all three sklearn models, evaluate each, and return results.

    Models compared:
        1. Linear Regression
        2. Random Forest (100 trees)
        3. Gradient Boosting (100 trees)

    Args:
        X_train: Training feature array.
        y_train: Training target array.
        X_test: Test feature array.
        y_test: Test target array.

    Returns:
        A tuple of:
            - ``pd.DataFrame`` with columns [Model, MAE, RMSE] sorted by RMSE
              ascending.
            - ``dict`` mapping model name to ``(model, predictions)`` for
              further analysis.

    Raises:
        RuntimeError: If any model training or evaluation fails.
    """
    trainers = {
        'Linear Regression': train_linear_regression,
        'Random Forest': train_random_forest,
        'Gradient Boosting': train_gradient_boosting,
    }

    results = []
    model_dict = {}

    for name, train_fn in trainers.items():
        try:
            model = train_fn(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_regression(y_test, preds)
            results.append({
                'Model': name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
            })
            model_dict[name] = (model, preds)
        except Exception as e:
            raise RuntimeError(f"Comparison failed for {name}: {e}") from e

    df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
    return df, model_dict


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from a tree-based model.

    Supports ``RandomForestRegressor`` and ``GradientBoostingRegressor``
    (or any estimator exposing a ``feature_importances_`` attribute).

    Args:
        model: A fitted tree-based estimator.
        feature_names: List of feature column names matching the model's
            input order.

    Returns:
        A ``pd.DataFrame`` with columns [Feature, Importance] sorted by
        importance descending.

    Raises:
        AttributeError: If the model does not expose ``feature_importances_``.
        ValueError: If the length of *feature_names* does not match the number
            of importances.
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError(
            f"{type(model).__name__} does not have feature_importances_. "
            "Only tree-based models (RandomForest, GradientBoosting) are supported."
        )

    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"number of importances ({len(importances)})"
        )

    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return df
