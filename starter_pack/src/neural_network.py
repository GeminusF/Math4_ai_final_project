"""
One-hidden-layer neural network.

Architecture:
    h = tanh(W1 x + b1)
    s = W2 h + b2
    p = softmax(s)

We use Xavier init for the weights since tanh works best with that.
"""

import numpy as np
from .math_utils import softmax, cross_entropy_loss, one_hot


class NeuralNetwork:
    """Single hidden-layer network with tanh + softmax output."""

    def __init__(self, input_dim, hidden_dim, num_classes,
                 reg_lambda=1e-4, seed=42):
        rng = np.random.default_rng(seed)

        # Xavier init — scale based on fan-in + fan-out
        s1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        s2 = np.sqrt(2.0 / (hidden_dim + num_classes))

        self.W1 = rng.normal(0, s1, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, s2, (num_classes, hidden_dim))
        self.b2 = np.zeros(num_classes)

        self.reg_lambda = reg_lambda

        # forward pass cache
        self._X = None
        self._Z1 = None   # pre-activation
        self._H = None     # tanh output
        self._S = None     # output logits
        self._P = None     # softmax probs

    def forward(self, X):
        """Forward pass. Returns probabilities (n x k)."""
        Z1 = X @ self.W1.T + self.b1
        H = np.tanh(Z1)
        S = H @ self.W2.T + self.b2
        P = softmax(S)

        # stash everything for backprop
        self._X = X
        self._Z1 = Z1
        self._H = H
        self._S = S
        self._P = P
        return P

    def loss(self, y):
        """
        Cross-entropy with L2 on both weight matrices.
        Uses log-sum-exp trick through the logits.
        Call forward() first!
        """
        Y = one_hot(y, self.W2.shape[0])
        ce = cross_entropy_loss(self._S, Y)  # logits, not probs
        reg = (self.reg_lambda / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return ce + reg

    def backward(self, y):
        """
        Backprop through the whole network.

        Chain rule step by step:
         dL/dS  = (P - Y) / n
         dL/dW2 = dS^T @ H   + lambda * W2
         dL/db2 = sum(dS)
         dL/dZ1 = (dS @ W2) * (1 - H^2)   <-- tanh derivative
         dL/dW1 = dZ1^T @ X  + lambda * W1
         dL/db1 = sum(dZ1)
        """
        n = self._X.shape[0]
        Y = one_hot(y, self.W2.shape[0])

        # output layer
        dS = (self._P - Y) / n
        dW2 = dS.T @ self._H + self.reg_lambda * self.W2
        db2 = dS.sum(axis=0)

        # hidden layer
        dH = dS @ self.W2                          # (n, hidden)
        dZ1 = dH * (1.0 - self._H ** 2)            # tanh'(z) = 1 - tanh(z)^2
        dW1 = dZ1.T @ self._X + self.reg_lambda * self.W1
        db1 = dZ1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def parameters(self):
        return [("W1", self.W1), ("b1", self.b1),
                ("W2", self.W2), ("b2", self.b2)]

    def predict(self, X):
        self.forward(X)
        return np.argmax(self._P, axis=1)

    def get_params(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def set_params(self, p):
        self.W1 = p["W1"].copy()
        self.b1 = p["b1"].copy()
        self.W2 = p["W2"].copy()
        self.b2 = p["b2"].copy()