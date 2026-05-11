"""
backprop.py — Manual backpropagation from scratch (NumPy only).

Implements a fully-connected neural network with:
  - Forward pass with arbitrary depth
  - Analytical backpropagation (chain rule)
  - ReLU and Sigmoid activations
  - MSE loss (regression) and Binary Cross-Entropy (classification)
  - Mini-batch training with momentum SGD

Educational goal: understand exactly how gradients flow backwards.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)

ACTIVATIONS = {
    "relu":    (relu,    relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
}


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class MLP:
    """
    A multi-layer perceptron trained with manual backpropagation.

    Parameters
    ----------
    layer_sizes : list[int]
        e.g. [2, 8, 4, 1] → 2-D input, two hidden layers (8, 4), 1 output
    activation : str
        Hidden layer activation: 'relu' | 'sigmoid'
    task : str
        'regression' (MSE loss) | 'classification' (BCE loss)
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient (0 = vanilla SGD).
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        task: str = "classification",
        lr: float = 0.01,
        momentum: float = 0.9,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.task        = task
        self.lr          = lr
        self.momentum    = momentum
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.history_: list[float] = []

        # He initialisation for ReLU, Xavier for Sigmoid
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.vW: list[np.ndarray] = []  # momentum buffers
        self.vb: list[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            if activation == "relu":
                std = np.sqrt(2.0 / fan_in)          # He init
            else:
                std = np.sqrt(1.0 / fan_in)          # Xavier init
            self.W.append(rng.normal(0, std, (fan_in, layer_sizes[i + 1])))
            self.b.append(np.zeros(layer_sizes[i + 1]))
            self.vW.append(np.zeros_like(self.W[-1]))
            self.vb.append(np.zeros_like(self.b[-1]))

    # ------------------------------------------------------------------
    def _forward(self, X: np.ndarray) -> tuple[list, list]:
        """Return (pre-activations Z, activations A) for all layers."""
        Z_list, A_list = [], [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A_list[-1] @ W + b
            Z_list.append(Z)
            if i < len(self.W) - 1:          # hidden layers
                A_list.append(self.act(Z))
            else:                             # output layer
                if self.task == "classification":
                    A_list.append(sigmoid(Z))
                else:
                    A_list.append(Z)
        return Z_list, A_list

    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.task == "classification":
            eps = 1e-8
            return float(-np.mean(
                y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
            ))
        else:
            return float(np.mean((y_pred - y_true) ** 2))

    def _backward(self, Z_list, A_list, y_true: np.ndarray) -> None:
        """Compute gradients via backprop and update weights."""
        m = y_true.shape[0]
        y_pred = A_list[-1]

        # Output layer delta
        if self.task == "classification":
            delta = (y_pred - y_true) / m
        else:
            delta = 2 * (y_pred - y_true) / m

        # Backpropagate through layers
        for i in reversed(range(len(self.W))):
            dW = A_list[i].T @ delta
            db = delta.sum(axis=0)

            # Momentum SGD update
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW
            self.vb[i] = self.momentum * self.vb[i] - self.lr * db
            self.W[i] += self.vW[i]
            self.b[i]  += self.vb[i]

            if i > 0:
                delta = (delta @ self.W[i].T) * self.act_grad(Z_list[i - 1])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        batch_size: int = 32,
        rng: np.random.Generator | None = None,
    ) -> "MLP":
        if rng is None:
            rng = np.random.default_rng(42)
        self.history_ = []
        N = X.shape[0]
        for epoch in range(epochs):
            idx = rng.permutation(N)
            for start in range(0, N, batch_size):
                batch = idx[start : start + batch_size]
                Z_list, A_list = self._forward(X[batch])
                self._backward(Z_list, A_list, y[batch])
            _, A = self._forward(X)
            self.history_.append(self._compute_loss(A[-1], y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, A = self._forward(X)
        return A[-1]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_loss(history: list[float], title: str, filename: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history, lw=2, color="steelblue")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150); plt.close()
    print(f"  Saved: {RESULTS / filename}")
