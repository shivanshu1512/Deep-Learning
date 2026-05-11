"""
gradient_descent.py — Batch GD vs. SGD vs. Mini-Batch GD from scratch.

Covers:
  - Batch Gradient Descent (exact gradient, slow)
  - Stochastic Gradient Descent (noisy, fast)
  - Mini-Batch Gradient Descent (balance of both)
  - Vanishing Gradient problem demonstration
  - Learning rate effect visualisation
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")


# ---------------------------------------------------------------------------
# Loss & gradient for simple linear regression (MSE)
# ---------------------------------------------------------------------------

def mse_loss(w: float, b: float, X: np.ndarray, y: np.ndarray) -> float:
    preds = X * w + b
    return float(np.mean((preds - y) ** 2))


def mse_gradient(w: float, b: float, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    preds = X * w + b
    err   = preds - y
    dw = float(2 * np.mean(err * X))
    db = float(2 * np.mean(err))
    return dw, db


# ---------------------------------------------------------------------------
# Gradient descent variants
# ---------------------------------------------------------------------------

class LinearRegressor:
    """Simple 1-D linear regressor trained with different GD strategies."""

    def __init__(self, lr: float = 0.01, epochs: int = 100, seed: int = 42) -> None:
        self.lr     = lr
        self.epochs = epochs
        self.rng    = np.random.default_rng(seed)
        self.w_: float = 0.0
        self.b_: float = 0.0
        self.history_: list[float] = []

    def _init_weights(self):
        self.w_ = float(self.rng.normal(0, 0.1))
        self.b_ = 0.0
        self.history_ = []

    # ---- Batch GD --------------------------------------------------------
    def fit_batch(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        self._init_weights()
        for _ in range(self.epochs):
            dw, db = mse_gradient(self.w_, self.b_, X, y)
            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db
            self.history_.append(mse_loss(self.w_, self.b_, X, y))
        return self

    # ---- Stochastic GD ---------------------------------------------------
    def fit_sgd(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        self._init_weights()
        for _ in range(self.epochs):
            idx = self.rng.integers(0, len(X))
            xi, yi = X[idx : idx + 1], y[idx : idx + 1]
            dw, db = mse_gradient(self.w_, self.b_, xi, yi)
            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db
            self.history_.append(mse_loss(self.w_, self.b_, X, y))
        return self

    # ---- Mini-Batch GD ---------------------------------------------------
    def fit_minibatch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 16) -> "LinearRegressor":
        self._init_weights()
        N = len(X)
        for _ in range(self.epochs):
            idx = self.rng.choice(N, batch_size, replace=False)
            dw, db = mse_gradient(self.w_, self.b_, X[idx], y[idx])
            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db
            self.history_.append(mse_loss(self.w_, self.b_, X, y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.w_ * X + self.b_


# ---------------------------------------------------------------------------
# Vanishing gradient demo
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def simulate_vanishing_gradient(n_layers: int = 10) -> np.ndarray:
    """Simulate gradient magnitude through n_layers of sigmoid activations."""
    x = np.array([0.0])           # pre-activation at each layer
    gradients = []
    grad = np.array([1.0])        # start with gradient = 1
    for _ in range(n_layers):
        grad = grad * sigmoid_derivative(x)
        gradients.append(float(np.abs(grad)))
    return np.array(gradients)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_gd_comparison(histories: dict[str, list[float]], filename: str = "gd_comparison.png") -> None:
    RESULTS.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 5))
    for name, hist in histories.items():
        plt.plot(hist, lw=2, label=name)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.yscale("log")
    plt.title("Batch GD vs SGD vs Mini-Batch GD"); plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150); plt.close()
    print(f"  Saved: {RESULTS / filename}")


def plot_vanishing_gradient(filename: str = "vanishing_gradient.png") -> None:
    RESULTS.mkdir(exist_ok=True)
    grads = simulate_vanishing_gradient(n_layers=15)
    plt.figure(figsize=(8, 4))
    plt.semilogy(range(1, len(grads) + 1), grads, "o-", lw=2, color="purple")
    plt.xlabel("Layer (from output)"); plt.ylabel("Gradient Magnitude (log scale)")
    plt.title("Vanishing Gradient through Sigmoid Activations")
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150); plt.close()
    print(f"  Saved: {RESULTS / filename}")


def plot_lr_effect(X: np.ndarray, y: np.ndarray, filename: str = "lr_effect.png") -> None:
    RESULTS.mkdir(exist_ok=True)
    lrs = [0.001, 0.01, 0.1, 0.5]
    plt.figure(figsize=(9, 5))
    for lr in lrs:
        m = LinearRegressor(lr=lr, epochs=100).fit_batch(X, y)
        plt.plot(m.history_, lw=2, label=f"lr={lr}")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.yscale("log")
    plt.title("Effect of Learning Rate"); plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150); plt.close()
    print(f"  Saved: {RESULTS / filename}")
