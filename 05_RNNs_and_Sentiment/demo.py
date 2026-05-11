"""
demo.py — RNN / LSTM Sentiment Analysis on IMDB.

Usage
-----
python demo.py --model simple_rnn           # SimpleRNN baseline
python demo.py --model bilstm               # Bidirectional LSTM (best)
python demo.py --model deep_rnn             # Deep stacked LSTM
python demo.py --compare                    # Run all models and compare
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
from src.rnn_sentiment import (
    load_imdb, build_simple_rnn, build_lstm, build_deep_rnn,
    plot_history, MAX_FEATURES, MAX_LEN, RESULTS
)

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


def parse_args():
    p = argparse.ArgumentParser(description="RNN Sentiment Analysis (IMDB)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", choices=["simple_rnn","bilstm","deep_rnn"], default="bilstm")
    p.add_argument("--compare", action="store_true", help="Run & compare all models")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def train_and_report(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    print(f"\n  Model: {model.name} | Params: {model.count_params():,}")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5, verbose=0),
    ]
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=0.1, callbacks=callbacks, verbose=1)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")
    plot_history(hist, model.name, f"{model.name.lower()}_training.png")
    return acc


def main():
    args = parse_args()
    if not _HAS_TF:
        print("TensorFlow not installed. Run: pip install tensorflow")
        return

    print("Loading IMDB dataset …")
    X_train, y_train, X_test, y_test = load_imdb()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Vocab: {MAX_FEATURES:,}")

    models_to_run = {
        "simple_rnn": lambda: build_simple_rnn(MAX_FEATURES),
        "bilstm":     lambda: build_lstm(MAX_FEATURES),
        "deep_rnn":   lambda: build_deep_rnn(MAX_FEATURES, n_layers=3),
    }

    results = {}
    if args.compare:
        for name, factory in models_to_run.items():
            m = factory()
            acc = train_and_report(m, X_train, y_train, X_test, y_test,
                                   args.epochs, args.batch_size)
            results[name] = acc
        print("\n=== Model Comparison ===")
        for name, acc in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name:<20}: {acc:.4f}")
    else:
        m = models_to_run[args.model]()
        train_and_report(m, X_train, y_train, X_test, y_test,
                         args.epochs, args.batch_size)

    print("\nPlots saved to results/")


if __name__ == "__main__":
    main()
