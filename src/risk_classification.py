"""FloodGuard AI - Flood Risk Classification Pipeline.

Provides functions to assign risk labels based on river water levels,
train a RandomForest classifier, and evaluate classification performance
with precision-recall AUC and confusion matrices.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve, auc)
from sklearn.preprocessing import label_binarize
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RISK_CATEGORIES, RISK_THRESHOLDS, RANDOM_SEED


def assign_risk_labels(df: pd.DataFrame, level_col: str = 'river_level_m') -> pd.DataFrame:
    """Assign flood_risk labels based on river water level using config thresholds.

    Maps continuous river-level values to categorical risk labels:
        - < 3.0 m  → Low
        - 3.0–5.0 m → Moderate
        - 5.0–7.0 m → High
        - ≥ 7.0 m  → Critical

    Args:
        df: DataFrame containing at least the ``level_col`` column.
        level_col: Name of the column with river-level measurements in metres.

    Returns:
        A copy of *df* with an added ``flood_risk`` column (categorical,
        ordered by severity).

    Raises:
        KeyError: If ``level_col`` is not present in *df*.
        ValueError: If ``level_col`` contains non-numeric data.
    """
    try:
        if level_col not in df.columns:
            raise KeyError(
                f"Column '{level_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()

        if not pd.api.types.is_numeric_dtype(df[level_col]):
            raise ValueError(
                f"Column '{level_col}' must be numeric, got {df[level_col].dtype}"
            )

        conditions = [
            df[level_col] < RISK_THRESHOLDS['Low'][1],
            (df[level_col] >= RISK_THRESHOLDS['Moderate'][0]) & (df[level_col] < RISK_THRESHOLDS['Moderate'][1]),
            (df[level_col] >= RISK_THRESHOLDS['High'][0]) & (df[level_col] < RISK_THRESHOLDS['High'][1]),
            df[level_col] >= RISK_THRESHOLDS['Critical'][0],
        ]
        choices = RISK_CATEGORIES  # ['Low', 'Moderate', 'High', 'Critical']

        df['flood_risk'] = np.select(conditions, choices, default='Low')
        df['flood_risk'] = pd.Categorical(
            df['flood_risk'], categories=RISK_CATEGORIES, ordered=True
        )
        return df

    except (KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to assign risk labels: {e}") from e


def train_risk_classifier(X_train, y_train, n_estimators: int = 200):
    """Train a RandomForestClassifier for flood-risk prediction.

    Args:
        X_train: Feature matrix (array-like of shape ``(n_samples, n_features)``).
        y_train: Target labels (array-like of shape ``(n_samples,)``).
        n_estimators: Number of trees in the forest (default 200).

    Returns:
        A fitted ``RandomForestClassifier`` instance.

    Raises:
        ValueError: If *X_train* and *y_train* have incompatible shapes.
        RuntimeError: On any unexpected training failure.
    """
    try:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_SEED,
            class_weight='balanced',
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to train risk classifier: {e}") from e


def evaluate_classification(y_true, y_pred, y_prob, labels=None) -> dict:
    """Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted class probabilities (output of ``predict_proba``),
                shape ``(n_samples, n_classes)``.
        labels: List of label names in the order matching *y_prob* columns.
                Defaults to ``RISK_CATEGORIES``.

    Returns:
        A dict with keys:
            - ``precision`` – macro-averaged precision
            - ``recall``    – macro-averaged recall
            - ``f1``        – macro-averaged F1
            - ``pr_auc``    – dict mapping each class (and ``'macro_avg'``)
              to its precision-recall AUC
            - ``confusion_matrix`` – ``numpy.ndarray``
            - ``classification_report`` – human-readable report string

    Raises:
        RuntimeError: On any unexpected evaluation failure.
    """
    try:
        if labels is None:
            labels = RISK_CATEGORIES

        precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        recall_val = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        f1_val = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)

        # Precision-Recall AUC per class
        y_true_bin = label_binarize(y_true, classes=labels)
        pr_auc_dict = {}
        auc_values = []

        for i, label in enumerate(labels):
            if y_true_bin.shape[1] > i:
                prec_curve, rec_curve, _ = precision_recall_curve(
                    y_true_bin[:, i], y_prob[:, i]
                )
                pr_auc_val = auc(rec_curve, prec_curve)
                pr_auc_dict[label] = round(pr_auc_val, 4)
                auc_values.append(pr_auc_val)

        pr_auc_dict['macro_avg'] = round(float(np.mean(auc_values)), 4) if auc_values else 0.0

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

        return {
            'precision': round(precision, 4),
            'recall': round(recall_val, 4),
            'f1': round(f1_val, 4),
            'pr_auc': pr_auc_dict,
            'confusion_matrix': cm,
            'classification_report': report,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to evaluate classification: {e}") from e


def get_classification_report_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Return ``classification_report`` as a tidy DataFrame.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional list of label names.

    Returns:
        A ``pd.DataFrame`` with per-class precision, recall, f1-score,
        and support, plus macro/weighted averages.

    Raises:
        RuntimeError: On any unexpected failure.
    """
    try:
        if labels is None:
            labels = RISK_CATEGORIES

        report_dict = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        df = pd.DataFrame(report_dict).transpose()
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to build classification report DataFrame: {e}") from e


def get_confusion_matrix_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Return the confusion matrix as a labelled DataFrame.

    Rows represent *actual* classes and columns represent *predicted* classes.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional list of label names.

    Returns:
        A square ``pd.DataFrame`` indexed and columned by *labels*.

    Raises:
        RuntimeError: On any unexpected failure.
    """
    try:
        if labels is None:
            labels = RISK_CATEGORIES

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        df = pd.DataFrame(cm, index=labels, columns=labels)
        df.index.name = 'Actual'
        df.columns.name = 'Predicted'
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to build confusion matrix DataFrame: {e}") from e
