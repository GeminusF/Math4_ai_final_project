"""Evaluation helpers — metrics + repeated-seed runner."""

import numpy as np
from .math_utils import one_hot, cross_entropy_loss


def compute_metrics(model, X, y):
    """Get accuracy, CE, probabilities, and predictions for a dataset."""
    P = model.forward(X)
    preds = np.argmax(P, axis=1)
    acc = float(np.mean(preds == y))
    Y = one_hot(y, P.shape[1])
    ce = float(cross_entropy_loss(model._S, Y))  # use cached logits
    return {
        "accuracy": acc,
        "cross_entropy": ce,
        "probabilities": P,
        "predictions": preds,
    }


def repeated_seed_evaluation(model_class, model_kwargs,
                             optimizer_class, optimizer_kwargs,
                             X_train, y_train, X_val, y_val,
                             X_test, y_test,
                             train_fn, train_kwargs,
                             seeds=(0, 1, 2, 3, 4)):
    """
    Train with 5 different seeds, then report mean +/- 95% CI.
    We use the t-distribution with df=4 -> t_crit = 2.776.
    """
    accs, ces = [], []

    for s in seeds:
        mk = {**model_kwargs, "seed": s}
        model = model_class(**mk)
        opt = optimizer_class(**optimizer_kwargs)

        tk = {**train_kwargs, "seed": s, "verbose": False}
        _, best_p, _ = train_fn(model, opt, X_train, y_train, X_val, y_val, **tk)

        model.set_params(best_p)
        m = compute_metrics(model, X_test, y_test)
        accs.append(m["accuracy"])
        ces.append(m["cross_entropy"])
        print(f"    Seed {s}: Test Acc = {m['accuracy']:.4f}, "
              f"Test CE = {m['cross_entropy']:.4f}")

    accs = np.array(accs)
    ces = np.array(ces)
    n = len(seeds)
    t_crit = 2.776  # 95% CI, df=4

    def stats(arr):
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1))
        ci = t_crit * sd / np.sqrt(n)
        return mu, sd, ci

    a_mean, a_std, a_ci = stats(accs)
    c_mean, c_std, c_ci = stats(ces)

    return {
        "acc_mean": a_mean, "acc_std": a_std, "acc_ci": a_ci,
        "ce_mean": c_mean, "ce_std": c_std, "ce_ci": c_ci,
        "all_accs": accs, "all_ces": ces,
    }