"""
perceptron.py — From-scratch Perceptron implementation with visualisation.

Covers:
  - The Perceptron learning algorithm (Rosenblatt 1958)
  - The Perceptron Trick (weight update rule)
  - Hinge loss
  - Why perceptrons fail on non-linearly-separable data (XOR problem)
  - Decision boundary animation
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

RESULTS = Path("results")


class Perceptron:
    """
    Binary Perceptron classifier trained with the Perceptron learning rule.

    Parameters
    ----------
    learning_rate : float
        Step size for the weight update.
    max_epochs : int
        Maximum number of passes over the dataset.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_epochs: int = 100,
        random_state: int | None = 42,
    ) -> None:
        self.lr           = learning_rate
        self.max_epochs   = max_epochs
        self.rng          = np.random.default_rng(random_state)
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.errors_: list[int] = []

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train the perceptron on dataset (X, y).

        Parameters
        ----------
        X : (N, D) float array
        y : (N,) array of {-1, +1} labels

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_    = 0.0
        self.errors_  = []

        for epoch in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict_one(xi)
                if prediction != yi:
                    # Perceptron trick: move boundary towards misclassified point
                    update = self.lr * yi
                    self.weights_ += update * xi
                    self.bias_    += update
                    errors        += 1
            self.errors_.append(errors)
            if errors == 0:
                print(f"  Converged at epoch {epoch + 1}")
                break
        return self

    def predict_one(self, x: np.ndarray) -> int:
        return 1 if np.dot(self.weights_, x) + self.bias_ >= 0 else -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(x) for x in X])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# Hinge loss
# ---------------------------------------------------------------------------

def hinge_loss(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Element-wise hinge loss: max(0, 1 - y * score)."""
    return np.maximum(0.0, 1.0 - y_true * scores)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_decision_boundary(
    model: Perceptron,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Perceptron Decision Boundary",
    filename: str = "decision_boundary.png",
) -> None:
    RESULTS.mkdir(exist_ok=True)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, zz, alpha=0.3, cmap="RdBu")
    ax.scatter(X[y == 1, 0],  X[y == 1, 1],  c="blue",  label="+1", edgecolors="k", s=60)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="red",   label="-1", edgecolors="k", s=60)
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS / filename}")


def plot_errors(errors: list[int], filename: str = "training_errors.png") -> None:
    RESULTS.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(errors, "o-", lw=2, color="crimson")
    plt.xlabel("Epoch"); plt.ylabel("Misclassifications")
    plt.title("Perceptron Training Errors per Epoch")
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS / filename}")


def plot_hinge_loss(filename: str = "hinge_loss.png") -> None:
    RESULTS.mkdir(exist_ok=True)
    scores = np.linspace(-2.5, 2.5, 300)
    pos_loss = hinge_loss(np.ones_like(scores), scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scores, pos_loss, lw=2, label="Hinge Loss (y=+1)")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Score (w·x + b)"); ax.set_ylabel("Loss")
    ax.set_title("Hinge Loss: max(0, 1 − y·score)")
    ax.legend(); plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS / filename}")
