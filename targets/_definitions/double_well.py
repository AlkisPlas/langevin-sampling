"""Double-well target (geometric family): 2^d modes and a non-smooth potential."""

from __future__ import annotations

import numpy as np

from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from samplers.overdamped.ula.ula_runner import ULA
from samplers.overdamped.mala.mala_runner import MALA
from samplers.underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

from .common import QUANTILE_LEVELS, Numerical1D

# ===========================================================================
# Double-well (non-smooth):  U = sum_i (x_i^2 - |x_i|), wells at +/-1/2.
#    Separable -> identical 1D marginal across coordinates (numerical).
# ===========================================================================
DOUBLE_WELL_MARGINAL = Numerical1D(lambda t: -(t**2 - np.abs(t)), lo=-6.0, hi=6.0)


class DoubleWellULA(ULA):
    def grad_f(self, x):
        return 2.0 * x - np.sign(x)


class DoubleWellMALA(MALA):
    def f(self, x):
        return np.sum(x**2 - np.abs(x))

    def grad_f(self, x):
        return 2.0 * x - np.sign(x)


class DoubleWellBAOAB(BAOAB):
    def f(self, x):
        return np.sum(x**2 - np.abs(x))

    def grad_f(self, x):
        return 2.0 * x - np.sign(x)


class DoubleWellDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=QUANTILE_LEVELS,
                         acceptance_rate=acceptance_rate)
        self._m = DOUBLE_WELL_MARGINAL

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


def exact_samples(d, n, rng):
    # Separable: inverse-CDF sampling from the shared 1D marginal.
    u = rng.uniform(0.0, 1.0, size=(n, d))
    return DOUBLE_WELL_MARGINAL.quantile(u)
