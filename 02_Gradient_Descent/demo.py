"""
demo.py — Gradient Descent comparison demo.

Usage
-----
python demo.py                   # all demos
python demo.py --demo vanishing  # only vanishing gradient
python demo.py --demo compare    # only GD variant comparison
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

sys.path.insert(0, ".")
from src.gradient_descent import (
    LinearRegressor, plot_gd_comparison,
    plot_vanishing_gradient, plot_lr_effect
)


def make_data(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, n)
    y = 2.5 * X + 1.0 + rng.normal(0, 0.8, n)
    return X, y


def parse_args():
    p = argparse.ArgumentParser(description="Gradient Descent Demo",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--demo", choices=["compare", "vanishing", "lr", "all"], default="all")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    X, y = make_data()

    if args.demo in ("compare", "all"):
        print("\n=== Comparing GD Variants ===")
        batch = LinearRegressor(lr=args.lr, epochs=args.epochs).fit_batch(X, y)
        sgd   = LinearRegressor(lr=args.lr, epochs=args.epochs).fit_sgd(X, y)
        mini  = LinearRegressor(lr=args.lr, epochs=args.epochs).fit_minibatch(X, y)
        plot_gd_comparison({
            "Batch GD":      batch.history_,
            "SGD":           sgd.history_,
            "Mini-Batch GD": mini.history_,
        })
        print(f"  Batch GD   final loss: {batch.history_[-1]:.4f}")
        print(f"  SGD        final loss: {sgd.history_[-1]:.4f}")
        print(f"  Mini-Batch final loss: {mini.history_[-1]:.4f}")

    if args.demo in ("vanishing", "all"):
        print("\n=== Vanishing Gradient Demo ===")
        plot_vanishing_gradient()

    if args.demo in ("lr", "all"):
        print("\n=== Learning Rate Effect ===")
        plot_lr_effect(X, y)

    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
