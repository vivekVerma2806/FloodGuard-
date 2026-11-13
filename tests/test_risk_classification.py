"""Tests for src/risk_classification.py."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RISK_CATEGORIES
from src.risk_classification import (
    assign_risk_labels,
    evaluate_classification,
    get_confusion_matrix_df,
    train_risk_classifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_risk_df() -> pd.DataFrame:
    """Create a small DataFrame spanning all 4 risk categories."""
    np.random.seed(42)
    # Deliberately span all thresholds: Low <3, Moderate 3-5, High 5-7, Critical >=7
    levels = [1.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.0, 2.0, 4.0, 6.0, 8.0]
    return pd.DataFrame(
        {
            "river_level_m": levels,
            "rainfall_mm": np.random.uniform(0, 50, len(levels)),
            "temperature_c": np.random.uniform(20, 40, len(levels)),
            "humidity_pct": np.random.uniform(40, 100, len(levels)),
        }
    )


def _classification_data():
    """Return X_train, y_train, X_test, y_test for classification."""
    np.random.seed(42)
    n = 80
    X = np.random.rand(n, 4)
    # Simple mapping to 4 classes based on first feature quartile
    labels = pd.cut(
        X[:, 0],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=RISK_CATEGORIES,
    ).astype(str)
    split = int(n * 0.75)
    return X[:split], labels[:split], X[split:], labels[split:]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssignRiskLabels:
    def test_assign_risk_labels(self):
        """Check all 4 categories are possible."""
        df = _make_risk_df()
        result = assign_risk_labels(df)

        assert "flood_risk" in result.columns
        present = set(result["flood_risk"].unique())
        # All 4 categories should appear given the crafted levels
        assert present == set(RISK_CATEGORIES)


class TestTrainRiskClassifier:
    def test_train_risk_classifier(self):
        """Train on small data, check it can predict."""
        X_tr, y_tr, X_te, _ = _classification_data()
        model = train_risk_classifier(X_tr, y_tr)

        preds = model.predict(X_te)
        assert len(preds) == len(X_te)
        # Predictions should be valid categories
        assert set(preds).issubset(set(RISK_CATEGORIES))


class TestEvaluateClassification:
    def test_evaluate_classification(self):
        """Check output dict has expected keys."""
        X_tr, y_tr, X_te, y_te = _classification_data()
        model = train_risk_classifier(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)

        result = evaluate_classification(
            y_te, y_pred, y_prob, labels=RISK_CATEGORIES
        )

        assert isinstance(result, dict)
        # Should contain at minimum: precision, recall, f1
        for key in ("precision", "recall", "f1"):
            assert key in result, f"Missing key: {key}"


class TestConfusionMatrixDf:
    def test_confusion_matrix_df(self):
        """Check shape and labels."""
        X_tr, y_tr, X_te, y_te = _classification_data()
        model = train_risk_classifier(X_tr, y_tr)
        y_pred = model.predict(X_te)

        cm_df = get_confusion_matrix_df(y_te, y_pred, labels=RISK_CATEGORIES)

        assert isinstance(cm_df, pd.DataFrame)
        assert cm_df.shape[0] == cm_df.shape[1]  # square
        assert cm_df.shape[0] == len(RISK_CATEGORIES)
