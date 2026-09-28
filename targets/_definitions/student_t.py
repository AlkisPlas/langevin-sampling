"""Student-t target (tail-weight family): polynomial tails, finite variance for nu > 2."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS, _broadcast_scale

# ===========================================================================
# Student-t:  U = (nu+d)/2 * log(1 + ||(x - loc)/scale||^2 / nu)
# ===========================================================================
class StudentTULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, nu, location=None, scale=None,
                 x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.nu = nu
        self.location = np.zeros(d) if location is None else np.array(location)
        self.scale = _broadcast_scale(scale, d)
        self.scale_sq = self.scale ** 2

    def grad_f(self, x):
        diff = x - self.location
        standardized = diff / self.scale
        mahalanobis_sq = np.sum(standardized ** 2)
        denominator = 1 + mahalanobis_sq / self.nu
        return ((self.nu + self.d) / self.nu) * (diff / self.scale_sq) / denominator


class StudentTMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, nu, location=None, scale=None,
                 x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.nu = nu
        self.location = np.zeros(d) if location is None else np.array(location)
        self.scale = _broadcast_scale(scale, d)
        self.scale_sq = self.scale ** 2

    def f(self, x):
        diff = (x - self.location) / self.scale
        mahalanobis_sq = np.sum(diff ** 2)
        return ((self.nu + self.d) / 2) * np.log(1 + mahalanobis_sq / self.nu)

    def grad_f(self, x):
        diff = x - self.location
        standardized = diff / self.scale
        mahalanobis_sq = np.sum(standardized ** 2)
        denominator = 1 + mahalanobis_sq / self.nu
        return ((self.nu + self.d) / self.nu) * (diff / self.scale_sq) / denominator


class StudentTBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, nu, location=None, scale=None,
                 x0=None, v0=None, seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.nu = nu
        self.location = np.zeros(d) if location is None else np.array(location)
        self.scale = _broadcast_scale(scale, d)
        self.scale_sq = self.scale ** 2

    def f(self, x):
        diff = (x - self.location) / self.scale
        mahalanobis_sq = np.sum(diff ** 2)
        return ((self.nu + self.d) / 2) * np.log(1 + mahalanobis_sq / self.nu)

    def grad_f(self, x):
        diff = x - self.location
        standardized = diff / self.scale
        mahalanobis_sq = np.sum(standardized ** 2)
        denominator = 1 + mahalanobis_sq / self.nu
        return ((self.nu + self.d) / self.nu) * (diff / self.scale_sq) / denominator


class StudentTDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, location, scale, nu, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.nu = nu
        self.location = np.array(location)
        self.scale = _broadcast_scale(scale, samples_post.shape[1])

    def quantile(self, p, dim=None):
        q = student_t.ppf(p, df=self.nu)
        if dim is None:
            return self.location + self.scale * q
        return self.location[dim] + self.scale[dim] * q

    def cdf(self, x, dim):
        return student_t.cdf(x, df=self.nu, loc=self.location[dim], scale=self.scale[dim])

    def theoretical_mean(self, dim):
        if self.nu <= 1:
            return float("nan")
        return float(self.location[dim])

    def theoretical_variance(self, dim):
        if self.nu <= 2:
            return float("nan")
        return float(self.scale[dim] ** 2 * self.nu / (self.nu - 2))


def exact_samples(d, n, rng, nu=5.0):
    return rng.standard_t(df=nu, size=(n, d))
