"""Anisotropic (ill-conditioned) Gaussian target (geometric family): conditioning."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS

def variances_for_kappa(d, kappa):
    """Log-spaced eigenvalues spanning condition number kappa (variances 1..kappa)."""
    if d == 1:
        return np.array([1.0])
    return np.logspace(0.0, np.log10(kappa), d)


# ===========================================================================
# Anisotropic (ill-conditioned) Gaussian:  U = 1/2 sum_i x_i^2 / lambda_i
# ===========================================================================
class AnisotropicGaussianULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, kappa=100.0, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.inv_var = 1.0 / variances_for_kappa(d, kappa)

    def grad_f(self, x):
        return self.inv_var * x


class AnisotropicGaussianMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, kappa=100.0, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.inv_var = 1.0 / variances_for_kappa(d, kappa)

    def f(self, x):
        return 0.5 * np.sum(self.inv_var * x**2)

    def grad_f(self, x):
        return self.inv_var * x


class AnisotropicGaussianBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, kappa=100.0, x0=None, v0=None, seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.inv_var = 1.0 / variances_for_kappa(d, kappa)

    def f(self, x):
        return 0.5 * np.sum(self.inv_var * x**2)

    def grad_f(self, x):
        return self.inv_var * x


class AnisotropicGaussianDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, kappa=100.0, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.variances = variances_for_kappa(self.d, kappa)
        self.std = np.sqrt(self.variances)

    def quantile(self, p, dim=None):
        if dim is None:
            return self.std * norm.ppf(p)
        return self.std[dim] * norm.ppf(p)

    def cdf(self, x, dim):
        return norm.cdf(x, loc=0.0, scale=self.std[dim])

    def theoretical_mean(self, dim):
        return 0.0

    def theoretical_variance(self, dim):
        return float(self.variances[dim])


def exact_samples(d, n, rng, kappa=100.0):
    std = np.sqrt(variances_for_kappa(d, kappa))
    return rng.normal(0.0, 1.0, size=(n, d)) * std
