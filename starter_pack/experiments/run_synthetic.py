#!/usr/bin/env python3
"""
Synthetic dataset experiments.
Trains both models on linear_gaussian and moons, saves decision boundaries
and training curves.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data_loader import load_synthetic
from src.softmax_regression import SoftmaxRegression
from src.neural_network import NeuralNetwork
from src.optimizers import SGD, Adam
from src.train import train_model, evaluate
from src.visualize import plot_decision_boundary, plot_training_curves


def run(name, num_classes=2, epochs=200):
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"{'='*60}")

    data = load_synthetic(name)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]
    d = Xtr.shape[1]

    X_all = np.vstack([Xtr, Xva, Xte])
    y_all = np.concatenate([ytr, yva, yte])

    # -- softmax regression --
    print("\n  Softmax Regression")
    sr = SoftmaxRegression(d, num_classes)
    opt_sr = SGD(lr=0.05)
    sr_hist, sr_best, _ = train_model(sr, opt_sr, Xtr, ytr, Xva, yva,
                                       epochs=epochs)
    sr.set_params(sr_best)
    sr_acc, sr_ce = evaluate(sr, Xte, yte)
    print(f"  Test -- Acc: {sr_acc:.4f}  CE: {sr_ce:.4f}")

    # -- neural network (use Adam for better convergence on moons) --
    print(f"\n  Neural Network (hidden=32)")
    nn = NeuralNetwork(d, 32, num_classes)
    opt_nn = Adam(lr=0.005)
    nn_hist, nn_best, _ = train_model(nn, opt_nn, Xtr, ytr, Xva, yva,
                                       epochs=epochs)
    nn.set_params(nn_best)
    nn_acc, nn_ce = evaluate(nn, Xte, yte)
    print(f"  Test -- Acc: {nn_acc:.4f}  CE: {nn_ce:.4f}")

    # -- save figures --
    plot_decision_boundary(sr, X_all, y_all,
                           title=f"Softmax Regression — {name}",
                           filename=f"{name}_softmax_boundary.png")
    plot_decision_boundary(nn, X_all, y_all,
                           title=f"Neural Network — {name}",
                           filename=f"{name}_nn_boundary.png")
    plot_training_curves(sr_hist, title=f"Softmax — {name}",
                         filename=f"{name}_softmax_curves.png")
    plot_training_curves(nn_hist, title=f"Neural Network — {name}",
                         filename=f"{name}_nn_curves.png")


if __name__ == "__main__":
    run("linear_gaussian")
    run("moons")
    print("\nDone — figures saved to starter_pack/figures/")