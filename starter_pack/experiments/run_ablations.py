#!/usr/bin/env python3
"""
Ablation experiments:
  1) Capacity ablation on moons (hidden widths 2, 8, 32)
  2) Optimizer comparison on digits (SGD / Momentum / Adam)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data_loader import load_synthetic, load_digits
from src.neural_network import NeuralNetwork
from src.optimizers import SGD, Momentum, Adam
from src.train import train_model, evaluate
from src.visualize import plot_capacity_ablation, plot_optimizer_comparison


def capacity_ablation():
    """Try hidden widths {2, 8, 32} on moons."""
    print("\n" + "="*60)
    print("  CAPACITY ABLATION -- Moons")
    print("="*60)

    data = load_synthetic("moons")
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]
    X_all = np.vstack([Xtr, Xva, Xte])
    y_all = np.concatenate([ytr, yva, yte])

    widths = [2, 8, 32]
    models = []
    for w in widths:
        print(f"\n  width = {w}")
        nn = NeuralNetwork(2, w, 2)
        # use Adam so the NN actually learns the nonlinear boundary
        opt = Adam(lr=0.005)
        _, best, _ = train_model(nn, opt, Xtr, ytr, Xva, yva, epochs=300)
        nn.set_params(best)
        acc, ce = evaluate(nn, Xte, yte)
        print(f"  Test -- Acc: {acc:.4f}  CE: {ce:.4f}")
        models.append(nn)

    plot_capacity_ablation(models, widths, X_all, y_all,
                           filename="moons_capacity_ablation.png")


def optimizer_study():
    """SGD vs Momentum vs Adam on digits (NN hidden=32)."""
    print("\n" + "="*60)
    print("  OPTIMIZER STUDY -- Digits NN")
    print("="*60)

    data = load_digits()
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]

    configs = [
        ("SGD",      SGD(lr=0.05)),
        ("Momentum", Momentum(lr=0.05, mu=0.9)),
        ("Adam",     Adam(lr=0.001)),
    ]

    all_hist = []
    all_names = []
    for name, opt in configs:
        print(f"\n  {name}")
        nn = NeuralNetwork(64, 32, 10)
        hist, best, ep = train_model(nn, opt, Xtr, ytr, Xva, yva, epochs=200)
        nn.set_params(best)
        acc, ce = evaluate(nn, Xte, yte)
        print(f"  Test -- Acc: {acc:.4f}  CE: {ce:.4f}  (best @ epoch {ep})")
        all_hist.append(hist)
        all_names.append(name)

    plot_optimizer_comparison(all_hist, all_names,
                              filename="digits_optimizer_comparison.png")


if __name__ == "__main__":
    capacity_ablation()
    optimizer_study()
    print("\nDone.")