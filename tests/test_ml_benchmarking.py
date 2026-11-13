"""Tests for src/ml_benchmarking.py."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ml_benchmarking import (
    compare_models,
    evaluate_regression,
    train_gradient_boosting,
    train_linear_regression,
    train_random_forest,
)


# ---------------------------------------------------------------------------
# Small fixture data
# ---------------------------------------------------------------------------

def _small_regression_data():
    np.random.seed(42)
    X_train = np.random.rand(60, 4)
    y_train = X_train[:, 0] * 2 + np.random.rand(60) * 0.1
    X_test = np.random.rand(20, 4)
    y_test = X_test[:, 0] * 2 + np.random.rand(20) * 0.1
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainLinearRegression:
    def test_train_linear_regression(self):
        X_tr, y_tr, X_te, _ = _small_regression_data()
        model = train_linear_regression(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (X_te.shape[0],)


class TestTrainRandomForest:
    def test_train_random_forest(self):
        X_tr, y_tr, X_te, _ = _small_regression_data()
        model = train_random_forest(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (X_te.shape[0],)


class TestTrainGradientBoosting:
    def test_train_gradient_boosting(self):
        X_tr, y_tr, X_te, _ = _small_regression_data()
        model = train_gradient_boosting(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (X_te.shape[0],)


class TestEvaluateRegression:
    def test_evaluate_regression(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1])

        result = evaluate_regression(y_true, y_pred)

        assert "mae" in result
        assert "rmse" in result
        assert result["mae"] >= 0
        assert result["rmse"] >= 0

    def test_evaluate_regression_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate_regression(y, y)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)


class TestCompareModels:
    def test_compare_models(self):
        X_tr, y_tr, X_te, y_te = _small_regression_data()

        results_df, model_dict = compare_models(X_tr, y_tr, X_te, y_te)

        assert isinstance(results_df, pd.DataFrame)
        assert list(results_df.columns) == ["Model", "MAE", "RMSE"]
        assert len(results_df) == 3

        assert isinstance(model_dict, dict)
        assert len(model_dict) == 3
