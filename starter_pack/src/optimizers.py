"""Optimizers — SGD, Momentum, Adam."""

import numpy as np


class SGD:
    """Plain mini-batch SGD. Nothing fancy."""

    def __init__(self, lr=0.05):
        self.lr = lr

    def step(self, model, grads):
        for name, param in model.parameters():
            param -= self.lr * grads[name]


class Momentum:
    """SGD + momentum. Keeps a velocity buffer for each param."""

    def __init__(self, lr=0.05, mu=0.9):
        self.lr = lr
        self.mu = mu
        self._v = {}

    def step(self, model, grads):
        for name, param in model.parameters():
            if name not in self._v:
                self._v[name] = np.zeros_like(param)
            v = self._v[name]
            v[:] = self.mu * v + grads[name]
            param -= self.lr * v


class Adam:
    """
    Adam (Kingma & Ba 2015).
    Keeps running first and second moment estimates per parameter,
    with bias correction.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = {}
        self._v = {}
        self.t = 0

    def step(self, model, grads):
        self.t += 1
        for name, param in model.parameters():
            g = grads[name]
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)

            # update biased moments
            self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * g
            self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g**2

            # bias correction
            m_hat = self._m[name] / (1 - self.beta1 ** self.t)
            v_hat = self._v[name] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)