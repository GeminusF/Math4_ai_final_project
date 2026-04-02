"""Training loop — handles mini-batches, validation, and checkpointing."""

import numpy as np
from .data_loader import make_batches
from .math_utils import one_hot, cross_entropy_loss


def evaluate(model, X, y):
    """Quick eval: returns (accuracy, cross-entropy) on a given split."""
    P = model.forward(X)
    preds = np.argmax(P, axis=1)
    acc = np.mean(preds == y)
    # use the cached logits from forward() for stable CE computation
    Y = one_hot(y, P.shape[1])
    ce = cross_entropy_loss(model._S, Y)
    return float(acc), float(ce)


def train_model(model, optimizer, X_train, y_train, X_val, y_val,
                epochs=200, batch_size=64, seed=42, verbose=True):
    """
    Train for `epochs` epochs using mini-batches.
    Keeps a checkpoint of the best model (lowest val CE).
    Returns (history, best_params, best_epoch).
    """
    rng = np.random.default_rng(seed)

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    best_val = float("inf")
    best_params = model.get_params()
    best_epoch = 0

    for ep in range(1, epochs + 1):
        # mini-batch training
        for Xb, yb in make_batches(X_train, y_train, batch_size, rng):
            model.forward(Xb)
            grads = model.backward(yb)
            optimizer.step(model, grads)

        # epoch-level eval
        tr_acc, tr_loss = evaluate(model, X_train, y_train)
        va_acc, va_loss = evaluate(model, X_val, y_val)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # checkpoint best
        if va_loss < best_val:
            best_val = va_loss
            best_params = model.get_params()
            best_epoch = ep

        if verbose and ep % 20 == 0:
            print(f"  Epoch {ep:3d}  |  "
                  f"Train Loss {tr_loss:.4f}  Acc {tr_acc:.4f}  |  "
                  f"Val Loss {va_loss:.4f}  Acc {va_acc:.4f}")

    if verbose:
        print(f"  Best val loss: {best_val:.4f} at epoch {best_epoch}")

    return history, best_params, best_epoch