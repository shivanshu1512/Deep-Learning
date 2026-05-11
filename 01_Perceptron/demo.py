"""
demo.py — Interactive Perceptron demo.

Demonstrates:
  1. Linearly-separable data → Perceptron converges
  2. XOR data → Perceptron fails (the fundamental limitation)
  3. Hinge loss curve

Usage
-----
python demo.py                    # runs all demos
python demo.py --dataset linear   # only linearly separable
python demo.py --dataset xor      # only XOR (shows failure)
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

sys.path.insert(0, ".")
from src.perceptron import (
    Perceptron, plot_decision_boundary, plot_errors, plot_hinge_loss
)


def make_linear(n: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-2, -2], 0.8, (n // 2, 2))
    X1 = rng.normal([ 2,  2], 0.8, (n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([-1] * (n // 2) + [1] * (n // 2))
    return X, y


def make_xor(n: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    quads = [([-2,-2],1),([ 2, 2],1),([-2, 2],-1),([ 2,-2],-1)]
    Xs, ys = [], []
    for center, label in quads:
        Xs.append(rng.normal(center, 0.4, (n // 4, 2)))
        ys.extend([label] * (n // 4))
    return np.vstack(Xs), np.array(ys)


def run_linear():
    print("\n=== Linearly Separable Data ===")
    X, y = make_linear()
    p = Perceptron(learning_rate=0.1, max_epochs=200)
    p.fit(X, y)
    print(f"  Accuracy: {p.accuracy(X, y) * 100:.1f}%")
    plot_decision_boundary(p, X, y, "Linear Data — Perceptron Converges", "linear_boundary.png")
    plot_errors(p.errors_, "linear_errors.png")


def run_xor():
    print("\n=== XOR Data (Perceptron Limitation) ===")
    X, y = make_xor()
    p = Perceptron(learning_rate=0.1, max_epochs=200)
    p.fit(X, y)
    print(f"  Accuracy: {p.accuracy(X, y) * 100:.1f}%  ← never reaches 100% on XOR")
    plot_decision_boundary(p, X, y, "XOR Data — Perceptron Cannot Converge", "xor_boundary.png")
    plot_errors(p.errors_, "xor_errors.png")


def parse_args():
    p = argparse.ArgumentParser(description="Perceptron Demo")
    p.add_argument("--dataset", choices=["linear", "xor", "all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    plot_hinge_loss()
    if args.dataset in ("linear", "all"):
        run_linear()
    if args.dataset in ("xor", "all"):
        run_xor()
    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
