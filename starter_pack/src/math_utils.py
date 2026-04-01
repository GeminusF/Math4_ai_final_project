"""
Shared math helpers used across both models.
Softmax, cross-entropy, one-hot, etc.
"""

import numpy as np


def one_hot(y, k):
    """Turn integer labels into a one-hot matrix (n x k)."""
    n = y.shape[0]
    out = np.zeros((n, k))
    out[np.arange(n), y] = 1.0
    return out


def softmax(logits):
    """
    Numerically stable softmax.
    We subtract the row-max first so exp() doesn't blow up.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(logits):
    """
    Log-sum-exp trick: compute log(softmax(z)) directly from logits.

    Instead of doing log(exp(z_j) / sum(exp(z)))  -- which loses precision --
    we use: log_softmax(z)_j = z_j - max(z) - log(sum(exp(z - max(z))))

    This avoids the double rounding that happens if you first compute
    softmax and then take log of the result.
    """
    m = logits.max(axis=1, keepdims=True)
    shifted = logits - m
    lse = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - lse


def cross_entropy_loss(logits, y_onehot):
    """
    Mean cross-entropy using the log-sum-exp trick.
    Takes raw logits (not probabilities!) so we can compute log-softmax
    in a numerically stable way.
    """
    log_probs = log_softmax(logits)
    # only the entries where y_onehot == 1 matter
    return -np.sum(y_onehot * log_probs) / logits.shape[0]