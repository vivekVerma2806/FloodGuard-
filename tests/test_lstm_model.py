"""Tests for src/lstm_model.py."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lstm_model import (
    build_lstm_model,
    load_saved_model,
    predict_lstm,
    save_model,
    train_lstm_model,
)


# ---------------------------------------------------------------------------
# Tiny synthetic data helpers
# ---------------------------------------------------------------------------

SEQ_LEN = 5
N_FEATURES = 3
N_SAMPLES = 40  # keep small for speed


def _tiny_data():
    """Return (X_train, y_train, X_val, y_val) with correct LSTM shapes."""
    np.random.seed(42)
    X = np.random.rand(N_SAMPLES, SEQ_LEN, N_FEATURES).astype(np.float32)
    y = np.random.rand(N_SAMPLES).astype(np.float32)
    split = int(N_SAMPLES * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildLSTMModel:
    def test_build_lstm_model(self):
        """Check that the model has the expected layers and output shape."""
        model = build_lstm_model((SEQ_LEN, N_FEATURES))

        # Should have LSTM, Dropout, LSTM, Dropout, Dense layers
        layer_types = [type(l).__name__ for l in model.layers]
        assert "LSTM" in layer_types
        assert "Dense" in layer_types
        assert "Dropout" in layer_types

        # Output shape: (batch, 1)
        assert model.output_shape == (None, 1)

    def test_build_lstm_model_bad_shape(self):
        with pytest.raises(ValueError):
            build_lstm_model((10,))


class TestTrainLSTMModel:
    def test_train_lstm_model(self):
        """Quick train (2 epochs) on tiny data; check history has loss."""
        X_tr, y_tr, X_val, y_val = _tiny_data()
        model = build_lstm_model((SEQ_LEN, N_FEATURES))

        history = train_lstm_model(
            model, X_tr, y_tr, X_val, y_val, epochs=2, batch_size=8
        )

        assert "loss" in history.history
        assert len(history.history["loss"]) == 2


class TestPredictLSTM:
    def test_predict_lstm(self):
        """Check prediction shape matches input batch size."""
        X_tr, y_tr, X_val, y_val = _tiny_data()
        model = build_lstm_model((SEQ_LEN, N_FEATURES))
        # Train briefly so weights are not random-init zeros
        train_lstm_model(model, X_tr, y_tr, X_val, y_val, epochs=1, batch_size=8)

        preds = predict_lstm(model, X_val)

        assert preds.ndim == 1
        assert preds.shape[0] == X_val.shape[0]


class TestSaveLoadModel:
    def test_save_load_model(self, tmp_path):
        """Save model, reload it, and verify predictions match."""
        model = build_lstm_model((SEQ_LEN, N_FEATURES))
        X_tr, y_tr, X_val, y_val = _tiny_data()
        train_lstm_model(model, X_tr, y_tr, X_val, y_val, epochs=1, batch_size=8)

        model_path = str(tmp_path / "test_model.keras")
        save_model(model, path=model_path)

        loaded = load_saved_model(path=model_path)
        assert loaded is not None

        orig_preds = predict_lstm(model, X_val)
        loaded_preds = predict_lstm(loaded, X_val)
        np.testing.assert_allclose(orig_preds, loaded_preds, atol=1e-6)

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_saved_model(path=str(tmp_path / "nope.keras"))
        assert result is None
