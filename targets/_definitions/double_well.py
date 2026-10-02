"""Double-well target (geometric family): a barrier between wells and a non-smooth potential."""

from __future__ import annotations

import numpy as np

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS, Numerical1D

# ===========================================================================
# Double-well (non-smooth):  U = beta * sum_i (x_i^2 - |x_i|), wells at +/-1/2.
#    beta is the barrier height factor (barrier = beta/4 per coordinate).
#    Separable -> identical 1D marginal across coordinates (numerical).
# ===========================================================================
def double_well_marginal(beta=1.0):
    """1D marginal of the double well with barrier factor beta."""
    return Numerical1D(lambda t: -beta * (t**2 - np.abs(t)), lo=-6.0, hi=6.0)


DOUBLE_WELL_MARGINAL = double_well_marginal(1.0)


class DoubleWellULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, beta=1.0, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.beta = beta

    def grad_f(self, x):
        return self.beta * (2.0 * x - np.sign(x))


class DoubleWellMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, beta=1.0, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.beta = beta

    def f(self, x):
        return self.beta * np.sum(x**2 - np.abs(x))

    def grad_f(self, x):
        return self.beta * (2.0 * x - np.sign(x))


class DoubleWellBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, beta=1.0, x0=None, v0=None, seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.beta = beta

    def f(self, x):
        return self.beta * np.sum(x**2 - np.abs(x))

    def grad_f(self, x):
        return self.beta * (2.0 * x - np.sign(x))


class DoubleWellDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, beta=1.0, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self._m = DOUBLE_WELL_MARGINAL if beta == 1.0 else double_well_marginal(beta)

    def quantile(self, p, dim=None):
        q = self._m.quantile(p)
        if dim is None:
            return np.full(self.d, q)
        return q

    def cdf(self, x, dim):
        return self._m.cdf(x)

    def theoretical_mean(self, dim):
        return self._m.mean

    def theoretical_variance(self, dim):
        return self._m.var

    def b2_functions(self, dim):
        """Functions f = x_dim**p of the b^2 metric, as (p, E[f], Var[f]).

        See Hoffman & Sountsov (2022); an empty list excludes the coordinate.
        """
        m = self._m
        return [(1, m.mean, m.var), (2, m.m2, m.m4 - m.m2**2)]

    def mode_occupancy(self):
        """Per-coordinate sign occupancy (each well has weight 1/2) and flip rate."""
        pos = self.samples > 0
        frac_pos = pos.mean(axis=0)
        occ_err = float(np.mean(np.abs(frac_pos - 0.5)))
        if self.n_samples > 1:
            flips = (pos[1:] != pos[:-1]).mean()
            trans = float(flips)
        else:
            trans = float("nan")
        return {"occupancy_error": occ_err, "transition_rate": trans}


def exact_samples(d, n, rng, beta=1.0):
    # Separable: inverse-CDF sampling from the shared 1D marginal.
    u = rng.uniform(0.0, 1.0, size=(n, d))
    m = DOUBLE_WELL_MARGINAL if beta == 1.0 else double_well_marginal(beta)
    return m.quantile(u)
