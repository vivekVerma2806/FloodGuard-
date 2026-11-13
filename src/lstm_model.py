"""FloodGuard AI - LSTM Flood Forecasting Model.

Provides functions to build, train, predict, save, and load a 2-layer LSTM
model for river-level forecasting.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import LSTM_UNITS, LSTM_EPOCHS, LSTM_BATCH_SIZE, MODEL_DIR, RANDOM_SEED

# Set seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

_DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_flood_model.keras')


def build_lstm_model(input_shape: tuple, units: int = None) -> keras.Model:
    """Build a 2-layer LSTM with dropout for flood-level regression.

    Architecture:
        LSTM(units, return_sequences=True) -> Dropout(0.2) ->
        LSTM(units // 2) -> Dropout(0.2) -> Dense(1)

    Args:
        input_shape: Tuple of (sequence_length, n_features).
        units: Number of LSTM units in the first layer. Defaults to
            ``LSTM_UNITS`` from config.

    Returns:
        A compiled Keras Sequential model (optimizer='adam', loss='mse',
        metrics=['mae']).

    Raises:
        ValueError: If *input_shape* does not have exactly 2 dimensions.
    """
    if len(input_shape) != 2:
        raise ValueError(
            f"input_shape must be (seq_length, n_features), got {input_shape}"
        )

    units = units or LSTM_UNITS

    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units, return_sequences=True),
            Dropout(0.2),
            LSTM(units // 2),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to build LSTM model: {e}") from e


def train_lstm_model(model, X_train, y_train, X_val, y_val,
                     epochs=None, batch_size=None):
    """Train the LSTM model with early stopping.

    Uses ``EarlyStopping`` on validation loss with patience=10 and
    ``restore_best_weights=True``.

    Args:
        model: A compiled Keras model.
        X_train: Training input array of shape (n, seq_length, n_features).
        y_train: Training target array of shape (n,).
        X_val: Validation input array.
        y_val: Validation target array.
        epochs: Maximum training epochs. Defaults to ``LSTM_EPOCHS``.
        batch_size: Batch size. Defaults to ``LSTM_BATCH_SIZE``.

    Returns:
        A ``keras.callbacks.History`` object containing training metrics.

    Raises:
        RuntimeError: If training fails for any reason.
    """
    epochs = epochs or LSTM_EPOCHS
    batch_size = batch_size or LSTM_BATCH_SIZE

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )
        return history
    except Exception as e:
        raise RuntimeError(f"LSTM training failed: {e}") from e


def predict_lstm(model, X: np.ndarray) -> np.ndarray:
    """Generate predictions and return a 1-D array.

    Args:
        model: A trained Keras model.
        X: Input array of shape (n, seq_length, n_features).

    Returns:
        A 1-D numpy array of predicted values.

    Raises:
        RuntimeError: If prediction fails.
    """
    try:
        predictions = model.predict(X, verbose=0)
        return predictions.flatten()
    except Exception as e:
        raise RuntimeError(f"LSTM prediction failed: {e}") from e


def save_model(model, path: str = None):
    """Persist a Keras model to disk.

    Args:
        model: The Keras model to save.
        path: Destination file path. Defaults to
            ``MODEL_DIR/lstm_flood_model.keras``.

    Raises:
        RuntimeError: If saving fails.
    """
    path = path or _DEFAULT_MODEL_PATH

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        print(f"Model saved to {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {path}: {e}") from e


def load_saved_model(path: str = None):
    """Load a previously saved Keras model.

    Args:
        path: File path to load from. Defaults to
            ``MODEL_DIR/lstm_flood_model.keras``.

    Returns:
        The loaded Keras model, or ``None`` if the file does not exist.

    Raises:
        RuntimeError: If the file exists but cannot be loaded.
    """
    path = path or _DEFAULT_MODEL_PATH

    if not os.path.exists(path):
        print(f"No saved model found at {path}")
        return None

    try:
        model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e
