#!/usr/bin/env python3
"""
Repeated-seed evaluation on digits (5 seeds).
Reports mean +/- 95% CI for both models.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_digits
from src.softmax_regression import SoftmaxRegression
from src.neural_network import NeuralNetwork
from src.optimizers import SGD, Adam
from src.train import train_model
from src.evaluate import repeated_seed_evaluation


def main():
    print("Loading digits ...")
    data = load_digits()
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"],   data["y_val"]
    Xte, yte = data["X_test"],  data["y_test"]

    seeds = [0, 1, 2, 3, 4]

    # softmax regression
    print("\n" + "="*60)
    print("  REPEATED SEEDS -- Softmax Regression")
    print("="*60)
    sr_res = repeated_seed_evaluation(
        model_class=SoftmaxRegression,
        model_kwargs={"input_dim": 64, "num_classes": 10},
        optimizer_class=SGD,
        optimizer_kwargs={"lr": 0.05},
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        X_test=Xte, y_test=yte,
        train_fn=train_model,
        train_kwargs={"epochs": 200, "batch_size": 64},
        seeds=seeds,
    )
    print(f"\n  Acc: {sr_res['acc_mean']:.4f} +/- {sr_res['acc_ci']:.4f}")
    print(f"  CE:  {sr_res['ce_mean']:.4f} +/- {sr_res['ce_ci']:.4f}")

    # neural network
    print("\n" + "="*60)
    print("  REPEATED SEEDS -- Neural Network (Adam, hidden=32)")
    print("="*60)
    nn_res = repeated_seed_evaluation(
        model_class=NeuralNetwork,
        model_kwargs={"input_dim": 64, "hidden_dim": 32, "num_classes": 10},
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 0.001},
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        X_test=Xte, y_test=yte,
        train_fn=train_model,
        train_kwargs={"epochs": 200, "batch_size": 64},
        seeds=seeds,
    )
    print(f"\n  Acc: {nn_res['acc_mean']:.4f} +/- {nn_res['acc_ci']:.4f}")
    print(f"  CE:  {nn_res['ce_mean']:.4f} +/- {nn_res['ce_ci']:.4f}")

    # summary
    print("\n" + "="*60)
    print("  SUMMARY (5 seeds, 95% CI)")
    print("="*60)
    print(f"  {'Model':<25} {'Accuracy':>20} {'Cross-Entropy':>20}")
    print(f"  {'-'*65}")
    print(f"  {'Softmax Regression':<25} "
          f"{sr_res['acc_mean']:.4f} +/- {sr_res['acc_ci']:.4f}     "
          f"{sr_res['ce_mean']:.4f} +/- {sr_res['ce_ci']:.4f}")
    print(f"  {'Neural Network':<25} "
          f"{nn_res['acc_mean']:.4f} +/- {nn_res['acc_ci']:.4f}     "
          f"{nn_res['ce_mean']:.4f} +/- {nn_res['ce_ci']:.4f}")


if __name__ == "__main__":
    main()
    print("\nDone.")