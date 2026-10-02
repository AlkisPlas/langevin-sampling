"""
Target distributions for the Langevin sampler study -- the single source of truth.

This module is the public interface of the `targets` package. Import targets from
here only (`from targets.targets import ...` or `from targets import targets`).
Each target is defined in its own module under `targets/_definitions/`, which is
private: its layout may change without notice.

Two families of targets, eight in total.

Tail-weight family (the original study; these vary ONLY in tail decay):
  - gaussian             : exponential tails
  - student_t            : polynomial tails, finite variance for nu > 2
  - cauchy               : polynomial tails, no finite moments

Geometric family (the extension; each probes one distinct difficulty axis):
  - anisotropic_gaussian : conditioning (ill-conditioned covariance)
  - banana               : non-linear correlation (product of 2D Rosenbrock blocks)
  - gaussian_mixture     : multimodality (3 modes along axis 0)
  - funnel               : state-dependent scale (Neal's funnel)
  - double_well          : barrier + non-smoothness (beta * (x^2 - |x|))

Demo-only targets (used by the examples/ scripts, not by the experiment grid):
  - exponential, lognormal : supported on the positive orthant

Each target provides ULA / MALA / BAOAB sampler subclasses and a Diagnostics
subclass exposing the interface used by experiments/run_experiments.py:
  quantile(p, dim), cdf(x, dim), theoretical_mean(dim), theoretical_variance(dim)
plus, for multimodal targets, mode_occupancy() returning occupancy/transition stats.
`exact_samples()` draws i.i.d. reference samples from any of the eight targets.

No module in this package executes code at import time, so importing is safe.
Every potential is defined exactly once: run_experiments.py and the examples/
demo scripts all import from here rather than redefining potentials. The demos
previously carried their own copies, which had silently diverged -- their banana
was a single 2D block plus unit normals, not the product-of-blocks form that the
approved specification calls for.

Gradient convention: grad_f(x) = grad U(x) where pi(x) ~ exp(-U(x)) and f(x) = U(x).
All gradients and marginals are verified in targets/test_targets.py.
"""

from __future__ import annotations

from targets._definitions import (
    anisotropic_gaussian as _anisotropic_gaussian,
    banana as _banana,
    cauchy as _cauchy,
    double_well as _double_well,
    funnel as _funnel,
    gaussian as _gaussian,
    gaussian_mixture as _gaussian_mixture,
    student_t as _student_t,
)
from targets._definitions.anisotropic_gaussian import (
    AnisotropicGaussianBAOAB,
    AnisotropicGaussianDiagnostics,
    AnisotropicGaussianMALA,
    AnisotropicGaussianULA,
    variances_for_kappa,
)
from targets._definitions.banana import (
    BANANA_B,
    BANANA_V0,
    BananaBAOAB,
    BananaDiagnostics,
    BananaMALA,
    BananaULA,
)
from targets._definitions.cauchy import CauchyBAOAB, CauchyDiagnostics, CauchyMALA, CauchyULA
from targets._definitions.common import QUANTILE_LEVELS, Numerical1D
from targets._definitions.double_well import (
    DOUBLE_WELL_MARGINAL,
    DoubleWellBAOAB,
    DoubleWellDiagnostics,
    DoubleWellMALA,
    DoubleWellULA,
    double_well_marginal,
)
from targets._definitions.exponential import ExponentialDiagnostics, ExponentialMALA, ExponentialULA
from targets._definitions.funnel import (
    FUNNEL_SIGMA_V,
    FunnelBAOAB,
    FunnelDiagnostics,
    FunnelMALA,
    FunnelULA,
)
from targets._definitions.gaussian import GaussianBAOAB, GaussianDiagnostics, GaussianMALA, GaussianULA
from targets._definitions.gaussian_mixture import (
    MIX_CENTERS,
    GaussianMixtureBAOAB,
    GaussianMixtureDiagnostics,
    GaussianMixtureMALA,
    GaussianMixtureULA,
    mixture_cdf,
    mixture_quantile,
)
from targets._definitions.lognormal import LognormalDiagnostics, LognormalMALA, LognormalULA
from targets._definitions.student_t import (
    StudentTBAOAB,
    StudentTDiagnostics,
    StudentTMALA,
    StudentTULA,
)

# The eight target names understood by exact_samples() and by the factories in
# run_experiments.py.
TAIL_WEIGHT_TARGETS = ("gaussian", "student_t", "cauchy")
GEOMETRIC_TARGETS = ("anisotropic_gaussian", "banana", "gaussian_mixture",
                     "funnel", "double_well")
ALL_TARGETS = TAIL_WEIGHT_TARGETS + GEOMETRIC_TARGETS


def exact_samples(distribution, d, n, rng, nu=5.0, kappa=100.0,
                  V0=BANANA_V0, b=BANANA_B, centers=MIX_CENTERS,
                  sigma_v=FUNNEL_SIGMA_V, beta=1.0):
    """Draw `n` i.i.d. samples of dimension `d` from the named target.

    Every one of the eight targets admits exact sampling, which gives the
    diagnostics a ground truth independent of any Markov chain. Used by
    test_targets.py to verify that each potential, its declared marginals and
    its theoretical moments describe one and the same distribution.

    Returns an (n, d) array.
    """
    if distribution == "gaussian":
        return _gaussian.exact_samples(d, n, rng)
    if distribution == "student_t":
        return _student_t.exact_samples(d, n, rng, nu=nu)
    if distribution == "cauchy":
        return _cauchy.exact_samples(d, n, rng)
    if distribution == "anisotropic_gaussian":
        return _anisotropic_gaussian.exact_samples(d, n, rng, kappa=kappa)
    if distribution == "banana":
        return _banana.exact_samples(d, n, rng, V0=V0, b=b)
    if distribution == "gaussian_mixture":
        return _gaussian_mixture.exact_samples(d, n, rng, centers=centers)
    if distribution == "funnel":
        return _funnel.exact_samples(d, n, rng, sigma_v=sigma_v)
    if distribution == "double_well":
        return _double_well.exact_samples(d, n, rng, beta=beta)
    raise ValueError(f"Unknown distribution: {distribution}")


__all__ = [
    # target names
    "TAIL_WEIGHT_TARGETS", "GEOMETRIC_TARGETS", "ALL_TARGETS",
    # fixed parameters
    "BANANA_V0", "BANANA_B", "MIX_CENTERS", "FUNNEL_SIGMA_V", "QUANTILE_LEVELS",
    # helpers
    "variances_for_kappa", "mixture_cdf", "mixture_quantile", "Numerical1D",
    "DOUBLE_WELL_MARGINAL", "double_well_marginal", "exact_samples",
    # tail-weight family
    "GaussianULA", "GaussianMALA", "GaussianBAOAB", "GaussianDiagnostics",
    "StudentTULA", "StudentTMALA", "StudentTBAOAB", "StudentTDiagnostics",
    "CauchyULA", "CauchyMALA", "CauchyBAOAB", "CauchyDiagnostics",
    # geometric family
    "AnisotropicGaussianULA", "AnisotropicGaussianMALA", "AnisotropicGaussianBAOAB",
    "AnisotropicGaussianDiagnostics",
    "BananaULA", "BananaMALA", "BananaBAOAB", "BananaDiagnostics",
    "GaussianMixtureULA", "GaussianMixtureMALA", "GaussianMixtureBAOAB",
    "GaussianMixtureDiagnostics",
    "FunnelULA", "FunnelMALA", "FunnelBAOAB", "FunnelDiagnostics",
    "DoubleWellULA", "DoubleWellMALA", "DoubleWellBAOAB", "DoubleWellDiagnostics",
    # demo-only targets
    "ExponentialULA", "ExponentialMALA", "ExponentialDiagnostics",
    "LognormalULA", "LognormalMALA", "LognormalDiagnostics",
]
