"""
Softmax regression — our linear baseline.
Basically just a score function s = Wx + b followed by softmax.
"""

import numpy as np
from .math_utils import softmax, cross_entropy_loss, one_hot


class SoftmaxRegression:
    """Linear softmax classifier. No hidden layer, no nonlinearity."""

    def __init__(self, input_dim, num_classes, reg_lambda=1e-4, seed=42):
        # zero-init for weights; bias also zero
        self.W = np.zeros((num_classes, input_dim))
        self.b = np.zeros(num_classes)
        self.reg_lambda = reg_lambda

        # we cache stuff for backward()
        self._X = None
        self._S = None  # raw scores (logits)
        self._P = None  # probabilities after softmax

    def forward(self, X):
        """Compute class probabilities for input batch X (n x d)."""
        S = X @ self.W.T + self.b
        P = softmax(S)
        # save for backward
        self._X = X
        self._S = S
        self._P = P
        return P

    def loss(self, y):
        """
        Cross-entropy + L2 reg.
        Uses the log-sum-exp trick internally (via math_utils).
        Must call forward() first.
        """
        Y = one_hot(y, self.W.shape[0])
        ce = cross_entropy_loss(self._S, Y)  # pass logits, not probs
        reg = (self.reg_lambda / 2.0) * np.sum(self.W ** 2)
        return ce + reg

    def backward(self, y):
        """Gradient of loss w.r.t. W and b. Pretty standard chain rule stuff."""
        n = self._X.shape[0]
        Y = one_hot(y, self.W.shape[0])

        dS = (self._P - Y) / n
        dW = dS.T @ self._X + self.reg_lambda * self.W
        db = dS.sum(axis=0)
        return {"W": dW, "b": db}

    def parameters(self):
        return [("W", self.W), ("b", self.b)]

    def predict(self, X):
        self.forward(X)
        return np.argmax(self._P, axis=1)

    def get_params(self):
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_params(self, p):
        self.W = p["W"].copy()
        self.b = p["b"].copy()