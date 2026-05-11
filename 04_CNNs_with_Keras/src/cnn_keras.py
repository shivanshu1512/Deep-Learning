"""
cnn_keras.py — CNN demonstrations with Keras/TensorFlow.

Covers:
  - Conv2D with padding='same' vs 'valid'
  - MaxPooling2D vs AveragePooling2D
  - Batch Normalisation effect on training
  - Functional API for multi-input / multi-output models
  - Age & Gender prediction CNN (multi-task learning)
  - Model summary and parameter counts
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_tf():
    if not _HAS_TF:
        raise ImportError("TensorFlow/Keras required: pip install tensorflow")


def plot_history(history, metrics: list[str], title: str, filename: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        ax.plot(history.history[metric], lw=2, label=f"train {metric}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], lw=2, ls="--", label=f"val {metric}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(metric); ax.legend()
    fig.suptitle(title); plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150); plt.close()
    print(f"  Saved: {RESULTS / filename}")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_padding_demo_model(padding: str = "same") -> "keras.Model":
    """Simple CNN showing effect of padding='same' vs 'valid'."""
    _require_tf()
    inp = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, padding=padding, activation="relu")(inp)
    x = layers.Conv2D(64, 3, padding=padding, activation="relu")(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inp, x, name=f"CNN_padding_{padding}")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_pooling_demo_model(pool_type: str = "max") -> "keras.Model":
    """Shows MaxPooling2D vs AveragePooling2D."""
    _require_tf()
    Pool = layers.MaxPooling2D if pool_type == "max" else layers.AveragePooling2D
    inp = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = Pool(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = Pool(2)(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inp, x, name=f"CNN_{pool_type}pool")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_batchnorm_model(use_bn: bool = True) -> "keras.Model":
    """CNN with / without Batch Normalisation on MNIST."""
    _require_tf()
    inp = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, padding="same")(inp)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inp, x, name=f"CNN_{'with' if use_bn else 'no'}_BN")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_functional_api_model() -> "keras.Model":
    """
    Functional API demo: ResNet-style skip connection on MNIST.
    Shows how to build non-sequential architectures.
    """
    _require_tf()
    inp = keras.Input(shape=(28, 28, 1), name="image")
    # Main path
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(32, 3, padding="same")(x)
    # Skip connection
    skip = layers.Conv2D(32, 1, padding="same")(inp)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)
    x = layers.GlobalMaxPooling2D()(x)
    out = layers.Dense(10, activation="softmax", name="digit")(x)
    model = keras.Model(inp, out, name="ResNet_like_MNIST")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_age_gender_model(img_size: int = 64) -> "keras.Model":
    """
    Multi-task CNN for age (regression) + gender (classification).
    Uses Keras Functional API with shared backbone.
    """
    _require_tf()
    inp = keras.Input(shape=(img_size, img_size, 3), name="face_image")

    # Shared backbone
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # Task-specific heads
    age_out    = layers.Dense(1, activation="linear",  name="age")(x)
    gender_out = layers.Dense(1, activation="sigmoid", name="gender")(x)

    model = keras.Model(inp, [age_out, gender_out], name="Age_Gender_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"age": "mse", "gender": "binary_crossentropy"},
        metrics={"age": "mae", "gender": "accuracy"},
    )
    return model
