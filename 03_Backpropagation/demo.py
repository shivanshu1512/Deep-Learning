"""
demo.py — Backpropagation demo: regression & classification.

Usage
-----
python demo.py                        # both tasks
python demo.py --task classification
python demo.py --task regression
"""
from __future__ import annotations

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from src.backprop import MLP, plot_loss

RESULTS = Path("results")


def make_classification_data(n: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-2, -2], 1.0, (n // 2, 2))
    X1 = rng.normal([ 2,  2], 1.0, (n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float).reshape(-1, 1)
    return X, y


def make_regression_data(n: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (n, 1))
    y = np.sin(X) + rng.normal(0, 0.2, (n, 1))
    return X, y


def parse_args():
    p = argparse.ArgumentParser(description="Backpropagation Demo",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", choices=["classification", "regression", "all"], default="all")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    return p.parse_args()


def run_classification(epochs, activation):
    print("\n=== Backprop Classification (2-D Gaussian clusters) ===")
    X, y = make_classification_data()
    model = MLP([2, 16, 8, 1], activation=activation, task="classification",
                lr=0.05, momentum=0.9)
    model.fit(X, y, epochs=epochs, batch_size=32)
    preds = (model.predict(X) >= 0.5).astype(int).flatten()
    acc = np.mean(preds == y.flatten())
    print(f"  Final BCE Loss : {model.history_[-1]:.4f}")
    print(f"  Accuracy       : {acc * 100:.1f}%")
    plot_loss(model.history_, "Backprop Classification Loss", "classification_loss.png")

    # Decision boundary plot
    RESULTS.mkdir(exist_ok=True)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = (model.predict(grid) >= 0.5).reshape(xx.shape)
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap="RdBu")
    plt.scatter(X[y.flatten()==0,0], X[y.flatten()==0,1], c="red",  s=20, alpha=0.6, label="Class 0")
    plt.scatter(X[y.flatten()==1,0], X[y.flatten()==1,1], c="blue", s=20, alpha=0.6, label="Class 1")
    plt.title("MLP Decision Boundary (Backprop)"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "classification_boundary.png", dpi=150); plt.close()
    print(f"  Saved: {RESULTS / 'classification_boundary.png'}")


def run_regression(epochs, activation):
    print("\n=== Backprop Regression (sin curve) ===")
    X, y = make_regression_data()
    model = MLP([1, 32, 16, 1], activation=activation, task="regression",
                lr=0.01, momentum=0.9)
    model.fit(X, y, epochs=epochs, batch_size=32)
    print(f"  Final MSE: {model.history_[-1]:.4f}")
    plot_loss(model.history_, "Backprop Regression Loss (MSE)", "regression_loss.png")

    RESULTS.mkdir(exist_ok=True)
    X_sort = np.sort(X, axis=0)
    y_pred = model.predict(X_sort)
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, s=10, alpha=0.5, label="Data")
    plt.plot(X_sort, y_pred, lw=2, color="red", label="MLP fit")
    plt.title("MLP Regression (sin curve)"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "regression_fit.png", dpi=150); plt.close()
    print(f"  Saved: {RESULTS / 'regression_fit.png'}")


def main():
    args = parse_args()
    if args.task in ("classification", "all"):
        run_classification(args.epochs, args.activation)
    if args.task in ("regression", "all"):
        run_regression(args.epochs, args.activation)
    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
