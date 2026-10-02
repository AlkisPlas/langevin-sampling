"""Helpers shared by several target modules."""

from __future__ import annotations

import numpy as np

QUANTILE_LEVELS = [0.025, 0.5, 0.975]


def _trapz(y, x):
    """Trapezoidal integration, compatible with NumPy 1.x and 2.x."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x)


def _broadcast_scale(scale, d):
    """Expand a scalar or None scale into a length-d vector."""
    if scale is None:
        return np.ones(d)
    arr = np.array(scale)
    if arr.shape == ():
        return np.ones(d) * float(arr)
    return arr


class Numerical1D:
    """Normalised 1D distribution from an unnormalised log-density, on a grid.

    Provides cdf, quantile, mean, variance and the raw moments m2 = E[x^2],
    m4 = E[x^4] by quadrature. Used for the separable double-well marginal
    (identical across coordinates).
    """

    def __init__(self, logpdf, lo, hi, n=8001):
        self.grid = np.linspace(lo, hi, n)
        logp = logpdf(self.grid)
        logp = logp - np.max(logp)
        pdf = np.exp(logp)
        pdf = pdf / _trapz(pdf, self.grid)
        self.pdf = pdf
        dx = np.diff(self.grid)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)])
        self.cdf_grid = cdf / cdf[-1]
        self.mean = float(_trapz(self.grid * pdf, self.grid))
        self.var = float(_trapz((self.grid - self.mean) ** 2 * pdf, self.grid))
        self.m2 = float(_trapz(self.grid ** 2 * pdf, self.grid))
        self.m4 = float(_trapz(self.grid ** 4 * pdf, self.grid))

    def cdf(self, x):
        return np.interp(x, self.grid, self.cdf_grid)

    def quantile(self, p):
        q = np.interp(p, self.cdf_grid, self.grid)
        return float(q) if np.ndim(p) == 0 else q


# Boundary handling for the two positive-orthant demo targets (exponential,
# lognormal). MALA returns f = +inf outside the support, so the Metropolis test
# rejects the move exactly, whereas ULA has no such test and instead takes a
# large restoring gradient. That ULA fallback introduces a bias which the
# corresponding demo is meant to expose -- do not "fix" it.
_BOUNDARY_GRAD = 1e10


def _as_vector(value, d, default=1.0):
    if value is None:
        return np.ones(d) * default
    arr = np.asarray(value, dtype=float)
    return np.ones(d) * float(arr) if arr.shape == () else arr
