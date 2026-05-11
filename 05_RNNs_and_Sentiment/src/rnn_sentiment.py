"""
rnn_sentiment.py — RNN / LSTM for Sentiment Analysis (IMDB).

Covers:
  - SimpleRNN baseline
  - Stacked LSTM with dropout
  - Bidirectional LSTM
  - Deep RNN (multiple layers)
  - Embedding layer training
  - Text preprocessing pipeline
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path("results")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


MAX_FEATURES = 10_000   # vocabulary size
MAX_LEN      = 200      # sequence length


def load_imdb(max_features: int = MAX_FEATURES, max_len: int = MAX_LEN):
    """Load IMDB dataset, pad/truncate to max_len."""
    if not _HAS_TF:
        raise ImportError("TensorFlow required: pip install tensorflow")
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
        num_words=max_features
    )
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
    X_test  = keras.preprocessing.sequence.pad_sequences(X_test,  maxlen=max_len)
    return X_train, y_train, X_test, y_test


def build_simple_rnn(max_features: int, embed_dim: int = 32) -> "keras.Model":
    """Baseline: single SimpleRNN layer."""
    model = keras.Sequential([
        layers.Embedding(max_features, embed_dim, input_length=MAX_LEN),
        layers.SimpleRNN(32, return_sequences=False),
        layers.Dense(1, activation="sigmoid"),
    ], name="SimpleRNN")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(max_features: int, embed_dim: int = 64) -> "keras.Model":
    """Stacked Bidirectional LSTM with dropout."""
    model = keras.Sequential([
        layers.Embedding(max_features, embed_dim, input_length=MAX_LEN),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
        layers.Bidirectional(layers.LSTM(32, dropout=0.2)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ], name="BiLSTM")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_deep_rnn(max_features: int, n_layers: int = 3, embed_dim: int = 32) -> "keras.Model":
    """Deep stacked LSTM (n_layers layers)."""
    if not _HAS_TF:
        raise ImportError("TensorFlow required")
    inp = keras.Input(shape=(MAX_LEN,))
    x = layers.Embedding(max_features, embed_dim)(inp)
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        x = layers.LSTM(32, return_sequences=return_seq, dropout=0.2)(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, x, name=f"DeepRNN_{n_layers}layers")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history, title: str, filename: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], lw=2, label="train")
    ax1.plot(history.history["val_accuracy"], lw=2, ls="--", label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(history.history["loss"], lw=2, label="train")
    ax2.plot(history.history["val_loss"], lw=2, ls="--", label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()
    fig.suptitle(title); plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150); plt.close()
    print(f"  Saved: {RESULTS / filename}")
