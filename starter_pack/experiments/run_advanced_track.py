#!/usr/bin/env python3
"""
Track B: confidence and reliability analysis on digits.
Computes max-prob confidence, predictive entropy, and
generates 5-bin reliability diagrams for both models.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data_loader import load_digits
from src.softmax_regression import SoftmaxRegression
from src.neural_network import NeuralNetwork
from src.optimizers import SGD, Adam
from src.train import train_model
from src.evaluate import compute_metrics
from src.visualize import plot_confidence_reliability


def analyze(model, X_test, y_test, name):
    """Print confidence/entropy stats and make reliability plot."""
    met = compute_metrics(model, X_test, y_test)
    P = met["probabilities"]
    preds = met["predictions"]
    correct = (preds == y_test).astype(float)

    conf = np.max(P, axis=1)
    ent = -np.sum(P * np.log(np.clip(P, 1e-12, 1.0)), axis=1)

    n_ok = int(correct.sum())
    n_bad = len(correct) - n_ok

    print(f"\n  --- {name} ---")
    print(f"  Accuracy: {met['accuracy']:.4f}")
    print(f"  CE:       {met['cross_entropy']:.4f}")
    print(f"  Correct: {n_ok}   Wrong: {n_bad}")
    print(f"  Avg confidence (correct):   {conf[correct == 1].mean():.4f}")
    if n_bad > 0:
        print(f"  Avg confidence (incorrect): {conf[correct == 0].mean():.4f}")
    print(f"  Avg entropy    (correct):   {ent[correct == 1].mean():.4f}")
    if n_bad > 0:
        print(f"  Avg entropy    (incorrect): {ent[correct == 0].mean():.4f}")

    tag = name.lower().replace(" ", "_")
    rel = plot_confidence_reliability(conf, correct, name,
                                      filename=f"reliability_{tag}.png")

    # print reliability table
    print(f"\n  Reliability bins ({name}):")
    print(f"  {'Bin':>15}  {'Count':>6}  {'Conf':>9}  {'Acc':>9}")
    for i in range(len(rel["bin_accs"])):
        lo, hi = rel["bin_edges"][i], rel["bin_edges"][i+1]
        print(f"  [{lo:.2f}, {hi:.2f}]  {rel['bin_counts'][i]:>6}  "
              f"{rel['bin_confs'][i]:>9.4f}  {rel['bin_accs'][i]:>9.4f}")


def main():
    print("Loading digits ...")
    data = load_digits()
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"],   data["y_val"]
    Xte, yte = data["X_test"],  data["y_test"]

    # train both models
    print("\n  Training softmax regression ...")
    sr = SoftmaxRegression(64, 10)
    _, sr_best, _ = train_model(sr, SGD(lr=0.05), Xtr, ytr, Xva, yva,
                                 epochs=200, verbose=False)
    sr.set_params(sr_best)

    print("  Training neural network ...")
    nn = NeuralNetwork(64, 32, 10)
    _, nn_best, _ = train_model(nn, Adam(lr=0.001), Xtr, ytr, Xva, yva,
                                 epochs=200, verbose=False)
    nn.set_params(nn_best)

    print("\n" + "="*60)
    print("  TRACK B: CONFIDENCE & RELIABILITY")
    print("="*60)
    analyze(sr, Xte, yte, "Softmax Regression")
    analyze(nn, Xte, yte, "Neural Network")


if __name__ == "__main__":
    main()
    print("\nDone.")