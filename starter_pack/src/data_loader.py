"""Loading the three datasets + mini-batch iterator."""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_synthetic(name):
    """Load linear_gaussian or moons. Returns dict with train/val/test splits."""
    d = np.load(DATA_DIR / f"{name}.npz")
    return {k: d[k] for k in d.files}


def load_digits():
    """Load the digits benchmark with our fixed train/val/test split."""
    data = np.load(DATA_DIR / "digits_data.npz")
    idx = np.load(DATA_DIR / "digits_split_indices.npz")
    X, y = data["X"], data["y"]
    return {
        "X_train": X[idx["train_idx"]], "y_train": y[idx["train_idx"]],
        "X_val":   X[idx["val_idx"]],   "y_val":   y[idx["val_idx"]],
        "X_test":  X[idx["test_idx"]],  "y_test":  y[idx["test_idx"]],
    }


def make_batches(X, y, batch_size, rng=None):
    """Yield (X_batch, y_batch) tuples. Shuffles if rng is given."""
    n = X.shape[0]
    order = np.arange(n)
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, n, batch_size):
        sel = order[i : i + batch_size]
        yield X[sel], y[sel]