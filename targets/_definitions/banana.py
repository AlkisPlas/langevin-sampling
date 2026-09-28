"""Banana target (geometric family): non-linear correlation."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS

BANANA_V0 = 100.0      # variance of the wide coordinate in each 2D block
BANANA_B = 0.03        # curvature of the bend


# ===========================================================================
# Banana: product of d/2 independent 2D Rosenbrock blocks (requires even d).
#    Block (even=wide, odd=bent):  u = x_odd - b*x_even^2 + b*V0
#    U = 1/2 sum_blocks [ x_even^2 / V0 + u^2 ]
#
#    NOTE: this is the product-of-blocks form mandated by the approved
#    specification (proposed_targets_for_approval.md, section 2), NOT a single
#    2D block padded with unit normals. Every even coordinate is wide
#    (var V0) and every odd coordinate is bent (var 1 + 2 b^2 V0^2).
# ===========================================================================
def _banana_grad(x, V0, b):
    a = x[0::2]                      # wide coords
    c = x[1::2]                      # bent coords
    u = c - b * a**2 + b * V0
    g = np.empty_like(x)
    g[0::2] = a / V0 - 2.0 * b * a * u
    g[1::2] = u
    return g


def _banana_f(x, V0, b):
    a = x[0::2]
    c = x[1::2]
    u = c - b * a**2 + b * V0
    return 0.5 * np.sum(a**2 / V0 + u**2)


class BananaULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, V0=BANANA_V0, b=BANANA_B, x0=None, seed=None):
        assert d % 2 == 0, "banana requires even d (product of 2D blocks)"
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.V0, self.b = V0, b

    def grad_f(self, x):
        return _banana_grad(x, self.V0, self.b)


class BananaMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, V0=BANANA_V0, b=BANANA_B, x0=None, seed=None):
        assert d % 2 == 0, "banana requires even d (product of 2D blocks)"
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.V0, self.b = V0, b

    def f(self, x):
        return _banana_f(x, self.V0, self.b)

    def grad_f(self, x):
        return _banana_grad(x, self.V0, self.b)


class BananaBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, V0=BANANA_V0, b=BANANA_B,
                 x0=None, v0=None, seed=None):
        assert d % 2 == 0, "banana requires even d (product of 2D blocks)"
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.V0, self.b = V0, b

    def f(self, x):
        return _banana_f(x, self.V0, self.b)

    def grad_f(self, x):
        return _banana_grad(x, self.V0, self.b)


class BananaDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, V0=BANANA_V0, b=BANANA_B, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self.V0, self.b = V0, b
        # Var of the bent coordinate: Var[x_odd] = 1 + 2 b^2 V0^2.
        self.var_bent = 1.0 + 2.0 * b**2 * V0**2

    def quantile(self, p, dim=None):
        if dim is None:
            return np.nan
        if dim % 2 == 0:                      # wide coordinate ~ N(0, V0)
            return np.sqrt(self.V0) * norm.ppf(p)
        return np.nan                          # bent coordinate: no closed form

    def cdf(self, x, dim):
        if dim % 2 == 0:
            return norm.cdf(x, loc=0.0, scale=np.sqrt(self.V0))
        return np.nan

    def theoretical_mean(self, dim):
        return 0.0

    def theoretical_variance(self, dim):
        return float(self.V0) if dim % 2 == 0 else float(self.var_bent)


def exact_samples(d, n, rng, V0=BANANA_V0, b=BANANA_B):
    if d % 2 != 0:
        raise ValueError("banana requires even d")
    # Per 2D block: y_wide ~ N(0, V0), y_bent ~ N(0, 1), then bend.
    y = rng.normal(0.0, 1.0, size=(n, d))
    y[:, 0::2] *= np.sqrt(V0)
    x = y.copy()
    x[:, 1::2] = y[:, 1::2] + b * (y[:, 0::2] ** 2 - V0)
    return x
