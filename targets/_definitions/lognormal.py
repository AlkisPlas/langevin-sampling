"""Lognormal target. Demo-only target: used by the examples/ scripts, not part of the experiment grid.

Supported on the positive orthant; see the boundary-handling note in common.py.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA

from .common import QUANTILE_LEVELS, _BOUNDARY_GRAD, _as_vector

# ===========================================================================
# Lognormal:  log X ~ N(mu, sigma^2), so on x > 0
#             U = sum_i [ log x_i + (log x_i - mu_i)^2 / (2 sigma_i^2) ]
# ===========================================================================
class LognormalULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, mu=None, sigma=None, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.mu = np.zeros(self.d) if mu is None else np.asarray(mu, dtype=float)
        self.sigma = _as_vector(sigma, self.d)

    def grad_f(self, x):
        if np.any(x <= 0):
            return np.ones(self.d) * _BOUNDARY_GRAD
        log_x = np.log(x)
        return (1 / x) * (1 + (log_x - self.mu) / self.sigma**2)


class LognormalMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, mu=None, sigma=None, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.mu = np.zeros(self.d) if mu is None else np.asarray(mu, dtype=float)
        self.sigma = _as_vector(sigma, self.d)

    def f(self, x):
        if np.any(x <= 0):
            return np.inf
        log_x = np.log(x)
        return float(np.sum(log_x) + 0.5 * np.sum(((log_x - self.mu) / self.sigma) ** 2))

    def grad_f(self, x):
        if np.any(x <= 0):
            return np.ones(self.d) * _BOUNDARY_GRAD
        log_x = np.log(x)
        return (1 / x) * (1 + (log_x - self.mu) / self.sigma**2)


class LognormalDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, mu=None, sigma=None, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.mu = np.zeros(self.d) if mu is None else np.asarray(mu, dtype=float)
        self.sigma = _as_vector(sigma, self.d)

    def quantile(self, p, dim=None):
        if dim is None:
            return np.exp(self.mu + self.sigma * norm.ppf(p))
        return float(np.exp(self.mu[dim] + self.sigma[dim] * norm.ppf(p)))

    def cdf(self, x, dim):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(np.where(x > 0, x, np.nan)) - self.mu[dim]) / self.sigma[dim]
        return np.where(x > 0, norm.cdf(z), 0.0)

    def theoretical_mean(self, dim):
        return float(np.exp(self.mu[dim] + 0.5 * self.sigma[dim] ** 2))

    def theoretical_variance(self, dim):
        s2 = self.sigma[dim] ** 2
        return float((np.exp(s2) - 1.0) * np.exp(2 * self.mu[dim] + s2))
