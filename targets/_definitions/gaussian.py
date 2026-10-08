"""Gaussian target (tail-weight family): exponential tails."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS

# ===========================================================================
# Gaussian:  U = 1/2 (x - mu)^T Sigma^{-1} (x - mu)
# ===========================================================================
class GaussianULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, mu=None, Sigma=None, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.mu = np.zeros(d) if mu is None else np.array(mu)
        self.Sigma = np.eye(d) if Sigma is None else np.array(Sigma)
        # Sigma = I (the default): None selects the O(d) form instead of a d x d product.
        self.Sigma_inv = None if Sigma is None else np.linalg.inv(self.Sigma)

    def grad_f(self, x):
        if self.Sigma_inv is None:
            return x - self.mu
        return self.Sigma_inv @ (x - self.mu)


class GaussianMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, mu=None, Sigma=None, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.mu = np.zeros(d) if mu is None else np.array(mu)
        self.Sigma = np.eye(d) if Sigma is None else np.array(Sigma)
        # Sigma = I (the default): None selects the O(d) form instead of a d x d product.
        self.Sigma_inv = None if Sigma is None else np.linalg.inv(self.Sigma)

    def f(self, x):
        diff = x - self.mu
        if self.Sigma_inv is None:
            return 0.5 * diff @ diff
        return 0.5 * diff @ self.Sigma_inv @ diff

    def grad_f(self, x):
        if self.Sigma_inv is None:
            return x - self.mu
        return self.Sigma_inv @ (x - self.mu)


class GaussianBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, mu=None, Sigma=None,
                 x0=None, v0=None, seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.mu = np.zeros(d) if mu is None else np.array(mu)
        self.Sigma = np.eye(d) if Sigma is None else np.array(Sigma)
        # Sigma = I (the default): None selects the O(d) form instead of a d x d product.
        self.Sigma_inv = None if Sigma is None else np.linalg.inv(self.Sigma)

    def f(self, x):
        diff = x - self.mu
        if self.Sigma_inv is None:
            return 0.5 * diff @ diff
        return 0.5 * diff @ self.Sigma_inv @ diff

    def grad_f(self, x):
        if self.Sigma_inv is None:
            return x - self.mu
        return self.Sigma_inv @ (x - self.mu)


class GaussianDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, mu, Sigma, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.mu = np.array(mu)
        self.Sigma = np.array(Sigma)
        self.std = np.sqrt(np.diag(self.Sigma))

    def quantile(self, p, dim=None):
        if dim is None:
            return self.mu + self.std * norm.ppf(p)
        return self.mu[dim] + self.std[dim] * norm.ppf(p)

    def cdf(self, x, dim):
        return norm.cdf(x, loc=self.mu[dim], scale=self.std[dim])

    def theoretical_mean(self, dim):
        return float(self.mu[dim])

    def theoretical_variance(self, dim):
        return float(self.Sigma[dim, dim])


def exact_samples(d, n, rng):
    return rng.normal(0.0, 1.0, size=(n, d))
