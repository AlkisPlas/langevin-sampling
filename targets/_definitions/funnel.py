"""Neal's funnel target (geometric family): state-dependent scale."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS

FUNNEL_SIGMA_V = 3.0   # std of the funnel neck coordinate


# ===========================================================================
# Neal's funnel:  v=x0 ~ N(0, sigma_v^2),  x_i|v ~ N(0, e^v)  (i>=1)
#    U = v^2/(2 sigma_v^2) + (d-1)/2 * v + 1/2 e^{-v} sum_{i>=1} x_i^2
# ===========================================================================
def _funnel_grad(x, sigma_v):
    v = x[0]
    rest = x[1:]
    g = np.empty_like(x)
    d = x.shape[0]
    g[0] = v / sigma_v**2 + (d - 1) / 2.0 - 0.5 * np.exp(-v) * np.sum(rest**2)
    g[1:] = rest * np.exp(-v)
    return g


def _funnel_f(x, sigma_v):
    v = x[0]
    rest = x[1:]
    d = x.shape[0]
    return v**2 / (2.0 * sigma_v**2) + (d - 1) / 2.0 * v + 0.5 * np.exp(-v) * np.sum(rest**2)


class FunnelULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, sigma_v=FUNNEL_SIGMA_V, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.sigma_v = sigma_v

    def grad_f(self, x):
        return _funnel_grad(x, self.sigma_v)


class FunnelMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, sigma_v=FUNNEL_SIGMA_V, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.sigma_v = sigma_v

    def f(self, x):
        return _funnel_f(x, self.sigma_v)

    def grad_f(self, x):
        return _funnel_grad(x, self.sigma_v)


class FunnelBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, sigma_v=FUNNEL_SIGMA_V,
                 x0=None, v0=None, seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.sigma_v = sigma_v

    def f(self, x):
        return _funnel_f(x, self.sigma_v)

    def grad_f(self, x):
        return _funnel_grad(x, self.sigma_v)


class FunnelDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, sigma_v=FUNNEL_SIGMA_V, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.sigma_v = sigma_v
        self.var_xi = float(np.exp(sigma_v**2 / 2.0))  # E[e^v] for the conditioned coords

    def quantile(self, p, dim=None):
        if dim is None:
            return np.nan
        if dim == 0:
            return self.sigma_v * norm.ppf(p)
        return np.nan

    def cdf(self, x, dim):
        if dim == 0:
            return norm.cdf(x, loc=0.0, scale=self.sigma_v)
        return np.nan

    def theoretical_mean(self, dim):
        return 0.0

    def theoretical_variance(self, dim):
        return float(self.sigma_v**2) if dim == 0 else self.var_xi


def exact_samples(d, n, rng, sigma_v=FUNNEL_SIGMA_V):
    v = rng.normal(0.0, sigma_v, size=n)
    x = rng.normal(0.0, 1.0, size=(n, d)) * np.exp(v / 2.0)[:, None]
    x[:, 0] = v
    return x
