"""Exponential target. Demo-only target: used by the examples/ scripts, not part of the experiment grid.

Supported on the positive orthant; see the boundary-handling note in common.py.
"""

from __future__ import annotations

import numpy as np

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA

from .common import QUANTILE_LEVELS, _BOUNDARY_GRAD, _as_vector

# ===========================================================================
# Exponential:  U = sum_i rate_i * x_i  on x > 0
# ===========================================================================
class ExponentialULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, rate=None, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.rate = _as_vector(rate, self.d)
        if x0 is None:
            self.x0 = 1.0 / self.rate      # start at the mean, inside the support

    def grad_f(self, x):
        if np.any(x <= 0):
            return np.ones(self.d) * _BOUNDARY_GRAD
        return self.rate


class ExponentialMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, rate=None, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.rate = _as_vector(rate, self.d)
        if x0 is None:
            self.x0 = 1.0 / self.rate

    def f(self, x):
        if np.any(x <= 0):
            return np.inf                  # exact rejection outside the support
        return float(np.sum(self.rate * x))

    def grad_f(self, x):
        if np.any(x <= 0):
            return np.ones(self.d) * _BOUNDARY_GRAD
        return self.rate


class ExponentialDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, rate=None, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.rate = _as_vector(rate, self.d)

    def quantile(self, p, dim=None):
        if dim is None:
            return -np.log(1 - p) / self.rate
        return -np.log(1 - p) / self.rate[dim]

    def cdf(self, x, dim):
        return np.where(x > 0, 1.0 - np.exp(-self.rate[dim] * np.asarray(x)), 0.0)

    def theoretical_mean(self, dim):
        return float(1.0 / self.rate[dim])

    def theoretical_variance(self, dim):
        return float(1.0 / self.rate[dim] ** 2)
