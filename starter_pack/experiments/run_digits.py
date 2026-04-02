#!/usr/bin/env python3
"""
Digits benchmark: both models with default protocol settings.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_digits
from src.softmax_regression import SoftmaxRegression
from src.neural_network import NeuralNetwork
from src.optimizers import SGD, Adam
from src.train import train_model, evaluate
from src.visualize import plot_training_curves


def main():
    print("Loading digits ...")
    data = load_digits()
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]
    d, k = Xtr.shape[1], 10
    print(f"  Train {Xtr.shape}  Val {Xva.shape}  Test {Xte.shape}")

    # softmax baseline
    print("\n  Softmax Regression (SGD lr=0.05)")
    sr = SoftmaxRegression(d, k)
    sr_hist, sr_best, _ = train_model(sr, SGD(lr=0.05),
                                       Xtr, ytr, Xva, yva, epochs=200)
    sr.set_params(sr_best)
    sr_acc, sr_ce = evaluate(sr, Xte, yte)
    print(f"  Test -- Acc: {sr_acc:.4f}  CE: {sr_ce:.4f}")

    # neural network
    print("\n  Neural Network (Adam lr=0.001, hidden=32)")
    nn = NeuralNetwork(d, 32, k)
    nn_hist, nn_best, _ = train_model(nn, Adam(lr=0.001),
                                       Xtr, ytr, Xva, yva, epochs=200)
    nn.set_params(nn_best)
    nn_acc, nn_ce = evaluate(nn, Xte, yte)
    print(f"  Test -- Acc: {nn_acc:.4f}  CE: {nn_ce:.4f}")

    plot_training_curves(sr_hist, title="Softmax — Digits",
                         filename="digits_softmax_curves.png")
    plot_training_curves(nn_hist, title="Neural Network — Digits",
                         filename="digits_nn_curves.png")


if __name__ == "__main__":
    main()
    print("\nDone.")