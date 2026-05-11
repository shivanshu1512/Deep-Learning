"""
demo.py — CNN with Keras: MNIST training demo.

Usage
-----
python demo.py --demo all          # all CNN experiments
python demo.py --demo batchnorm    # BatchNorm vs no-BatchNorm
python demo.py --demo padding      # same vs valid padding
python demo.py --demo pooling      # max vs avg pooling
python demo.py --demo functional   # ResNet-style skip connection
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
from src.cnn_keras import (
    build_padding_demo_model, build_pooling_demo_model,
    build_batchnorm_model, build_functional_api_model, plot_history
)

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


def load_mnist():
    if not _HAS_TF:
        raise ImportError("TensorFlow required: pip install tensorflow")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train[..., None].astype("float32") / 255.0
    X_test  = X_test[..., None].astype("float32")  / 255.0
    return X_train, y_train, X_test, y_test


def parse_args():
    p = argparse.ArgumentParser(description="CNN with Keras Demo",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--demo", choices=["batchnorm","padding","pooling","functional","all"], default="all")
    p.add_argument("--epochs", type=int, default=5)
    return p.parse_args()


def run_batchnorm(X_train, y_train, X_test, y_test, epochs):
    print("\n=== Batch Normalisation: With vs Without ===")
    for use_bn in [False, True]:
        model = build_batchnorm_model(use_bn)
        print(f"\n  {'With' if use_bn else 'Without'} BatchNorm — {model.count_params():,} params")
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=128,
                         validation_split=0.1, verbose=0)
        val_acc = hist.history["val_accuracy"][-1]
        print(f"  Val Accuracy: {val_acc:.4f}")
        plot_history(hist, ["accuracy", "loss"],
                     f"BatchNorm={'Yes' if use_bn else 'No'}",
                     f"batchnorm_{'yes' if use_bn else 'no'}.png")


def run_padding(X_train, y_train, X_test, y_test, epochs):
    print("\n=== Padding: same vs valid ===")
    for p in ["same", "valid"]:
        model = build_padding_demo_model(p)
        print(f"\n  Padding={p}")
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=128,
                         validation_split=0.1, verbose=0)
        print(f"  Val Accuracy: {hist.history['val_accuracy'][-1]:.4f}")
        plot_history(hist, ["accuracy"], f"Padding={p}", f"padding_{p}.png")


def run_pooling(X_train, y_train, X_test, y_test, epochs):
    print("\n=== Pooling: max vs average ===")
    for pool in ["max", "average"]:
        model = build_pooling_demo_model(pool)
        print(f"\n  Pool={pool}")
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=128,
                         validation_split=0.1, verbose=0)
        print(f"  Val Accuracy: {hist.history['val_accuracy'][-1]:.4f}")
        plot_history(hist, ["accuracy"], f"{pool.capitalize()} Pooling", f"pooling_{pool}.png")


def run_functional(X_train, y_train, X_test, y_test, epochs):
    print("\n=== Functional API: ResNet-style skip connection ===")
    model = build_functional_api_model()
    model.summary()
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=128,
                     validation_split=0.1, verbose=1)
    print(f"  Val Accuracy: {hist.history['val_accuracy'][-1]:.4f}")
    plot_history(hist, ["accuracy", "loss"], "Functional API (Skip Connection)", "functional_api.png")


def main():
    args = parse_args()
    if not _HAS_TF:
        print("TensorFlow not installed. Run: pip install tensorflow")
        return
    X_train, y_train, X_test, y_test = load_mnist()
    if args.demo in ("batchnorm", "all"):
        run_batchnorm(X_train, y_train, X_test, y_test, args.epochs)
    if args.demo in ("padding", "all"):
        run_padding(X_train, y_train, X_test, y_test, args.epochs)
    if args.demo in ("pooling", "all"):
        run_pooling(X_train, y_train, X_test, y_test, args.epochs)
    if args.demo in ("functional", "all"):
        run_functional(X_train, y_train, X_test, y_test, args.epochs)
    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
