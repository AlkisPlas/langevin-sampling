"""Cauchy target (tail-weight family): polynomial tails, no finite moments."""

from __future__ import annotations

import numpy as np

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS

# ===========================================================================
# Cauchy:  U = (d+1)/2 * log(scale^2 + ||x - loc||^2)
# ===========================================================================
class CauchyULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, location=None, scale=1.0, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.location = np.zeros(d) if location is None else np.array(location)
        self.scale = scale

    def grad_f(self, x):
        diff = x - self.location
        return (self.d + 1) * diff / (self.scale ** 2 + np.sum(diff ** 2))


class CauchyMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, location=None, scale=1.0, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.location = np.zeros(d) if location is None else np.array(location)
        self.scale = scale

    def f(self, x):
        diff = x - self.location
        return ((self.d + 1) / 2) * np.log(self.scale ** 2 + np.sum(diff ** 2))

    def grad_f(self, x):
        diff = x - self.location
        return (self.d + 1) * diff / (self.scale ** 2 + np.sum(diff ** 2))


class CauchyBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, location=None, scale=1.0,
                 x0=None, v0=None, seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.location = np.zeros(d) if location is None else np.array(location)
        self.scale = scale

    def f(self, x):
        diff = x - self.location
        return ((self.d + 1) / 2) * np.log(self.scale ** 2 + np.sum(diff ** 2))

    def grad_f(self, x):
        diff = x - self.location
        return (self.d + 1) * diff / (self.scale ** 2 + np.sum(diff ** 2))


class CauchyDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, location, scale, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.location = np.array(location)
        self.scale = float(scale)

    def quantile(self, p, dim=None):
        q = np.tan(np.pi * (p - 0.5))
        if dim is None:
            return self.location + self.scale * q
        return self.location[dim] + self.scale * q

    def cdf(self, x, dim):
        z = (x - self.location[dim]) / self.scale
        return 0.5 + np.arctan(z) / np.pi

    def theoretical_mean(self, dim):
        return float("nan")

    def theoretical_variance(self, dim):
        return float("nan")


def exact_samples(d, n, rng):
    # Isotropic multivariate Cauchy: a Gaussian scale mixture with an
    # inverse-chi-square(1) mixing variable, matching U = (d+1)/2 log(1+|x|^2).
    z = rng.normal(0.0, 1.0, size=(n, d))
    w = rng.chisquare(df=1.0, size=(n, 1))
    return z / np.sqrt(w)
