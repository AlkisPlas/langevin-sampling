"""Gaussian mixture target (geometric family): multimodality."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS

MIX_CENTERS = (-4.0, 0.0, 4.0)  # three equally-weighted modes along axis 0


def mixture_cdf(t, centers):
    """CDF of the equally-weighted unit-variance 1D mixture with given centers."""
    centers = np.asarray(centers, dtype=float)
    return np.mean(norm.cdf(np.asarray(t)[..., None] - centers), axis=-1)


def mixture_quantile(p, centers, lo=-50.0, hi=50.0, n_iter=100):
    """Invert the 1D mixture CDF by bisection."""
    centers = np.asarray(centers, dtype=float)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if np.mean(norm.cdf(mid - centers)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ===========================================================================
# Gaussian mixture: 3 equally-weighted unit-var modes along axis 0.
#    U = -log sum_k exp(-||x - mu_k||^2 / 2)
# ===========================================================================
def _mixture_means(centers, d):
    means = np.zeros((len(centers), d))
    means[:, 0] = np.asarray(centers, dtype=float)
    return means


def _mixture_e(x, means):
    r = np.sum((x - means) ** 2, axis=1)
    m = r.min()
    return m, np.exp(-(r - m) / 2.0)


def _mixture_grad(x, means):
    _, e = _mixture_e(x, means)
    return (e[:, None] * (x - means)).sum(axis=0) / e.sum()


class GaussianMixtureULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, centers=MIX_CENTERS, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.means = _mixture_means(centers, d)

    def grad_f(self, x):
        return _mixture_grad(x, self.means)


class GaussianMixtureMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, centers=MIX_CENTERS, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.means = _mixture_means(centers, d)

    def f(self, x):
        m, e = _mixture_e(x, self.means)
        return m / 2.0 - np.log(e.sum())

    def grad_f(self, x):
        return _mixture_grad(x, self.means)


class GaussianMixtureBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, centers=MIX_CENTERS,
                 x0=None, v0=None, seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.means = _mixture_means(centers, d)

    def f(self, x):
        m, e = _mixture_e(x, self.means)
        return m / 2.0 - np.log(e.sum())

    def grad_f(self, x):
        return _mixture_grad(x, self.means)


class GaussianMixtureDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, centers=MIX_CENTERS, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.centers = np.asarray(centers, dtype=float)
        self.K = len(self.centers)

    def quantile(self, p, dim=None):
        if dim is None:
            return np.nan
        if dim == 0:
            return mixture_quantile(p, self.centers)
        return norm.ppf(p)

    def cdf(self, x, dim):
        if dim == 0:
            return mixture_cdf(x, self.centers)
        return norm.cdf(x)

    def theoretical_mean(self, dim):
        return 0.0  # symmetric centers, zero elsewhere

    def theoretical_variance(self, dim):
        if dim == 0:
            between = float(np.mean(self.centers**2) - np.mean(self.centers)**2)
            return 1.0 + between
        return 1.0

    def b2_group(self, dim):
        """Group of the coordinate for b2_grouped: coordinates with the same
        distribution share a group (see experiments/run_experiments.py)."""
        return 0 if dim == 0 else 1            # 0 = mode coordinate, 1 = unit normals

    def b2_functions(self, dim):
        """Functions f = x_dim**p of the b^2 metric, as (p, E[f], Var[f]).

        See Hoffman & Sountsov (2022); an empty list excludes the coordinate.
        """
        if dim != 0:                           # unit normal
            return [(1, 0.0, 1.0), (2, 1.0, 2.0)]
        c = self.centers                       # x = c + z, c uniform over centers
        mean = float(np.mean(c))
        m2 = 1.0 + float(np.mean(c**2))
        m4 = float(np.mean(c**4)) + 6.0 * float(np.mean(c**2)) + 3.0
        return [(1, mean, m2 - mean**2), (2, m2, m4 - m2**2)]

    def mode_occupancy(self):
        """Occupancy error and transition rate based on the axis-0 mode label."""
        x0 = self.samples[:, 0]
        labels = np.argmin(np.abs(x0[:, None] - self.centers), axis=1)
        fracs = np.array([np.mean(labels == k) for k in range(self.K)])
        occ_err = float(np.mean(np.abs(fracs - 1.0 / self.K)))
        trans = float(np.mean(labels[1:] != labels[:-1])) if len(labels) > 1 else float("nan")
        return {"occupancy_error": occ_err, "transition_rate": trans}


def exact_samples(d, n, rng, centers=MIX_CENTERS):
    centers = np.asarray(centers, dtype=float)
    x = rng.normal(0.0, 1.0, size=(n, d))
    x[:, 0] += centers[rng.integers(len(centers), size=n)]
    return x
