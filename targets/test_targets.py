"""
Consistency tests for targets/targets.py -- the single source of truth for
all eight target distributions.

Each target is described in three independent places: the potential U (used to
drive the chain), the Diagnostics marginals (cdf / quantile, used for KS and
q-MAE) and the theoretical moments (used for mean/var bias). Nothing forces
those three to agree, and when they silently disagree every downstream metric
is quietly wrong -- the sampler explores one distribution while the diagnostics
score it against another. These tests pin all three together.

The strongest check is test_potential_matches_reference_logpdf: it compares U(x)
against an independently written closed-form log-density and asserts the
difference is constant. That catches errors in normalisation-dependent terms
which a gradient check cannot see -- for instance Neal's funnel, where the
(d-1)/2 * v term comes from the log-determinant of the conditional Gaussians
and does not show up in any finite-difference test of grad U against U.

Run directly (no pytest needed):
    python -m targets.test_targets
or under pytest if available:
    pytest targets/test_targets.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp
from scipy.stats import kstest, norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets import targets
from experiments.run_experiments import (
    ExperimentConfig,
    build_diagnostics,
    build_sampler,
)

SEED = 12345


# ---------------------------------------------------------------------------
# Target specifications: which dimensions to exercise, and the extra knobs
# each target needs. `even_only` marks the banana (product of 2D blocks).
# ---------------------------------------------------------------------------
SPECS = [
    dict(name="gaussian", dims=[1, 3, 6], nu=None, kappa=None),
    dict(name="student_t", dims=[1, 3, 6], nu=5.0, kappa=None),
    dict(name="cauchy", dims=[1, 3, 6], nu=None, kappa=None),
    dict(name="anisotropic_gaussian", dims=[2, 4, 10], nu=None, kappa=100.0),
    dict(name="banana", dims=[2, 4, 10], nu=None, kappa=None),
    dict(name="gaussian_mixture", dims=[1, 3, 6], nu=None, kappa=None),
    dict(name="funnel", dims=[2, 4, 10], nu=None, kappa=None),
    dict(name="double_well", dims=[1, 3, 6], nu=None, kappa=None),
]


def _cfg(algorithm, spec, d, n_steps=1):
    return ExperimentConfig(
        algorithm=algorithm, distribution=spec["name"], d=d, eta=0.05,
        gamma=1.0 if algorithm == "BAOAB" else None, nu=spec["nu"],
        n_steps=n_steps, burn_in=0, seed=None, kappa=spec["kappa"],
    )


def _sampler(algorithm, spec, d):
    """Build a sampler through the real factory, so the wiring is tested too."""
    return build_sampler(_cfg(algorithm, spec, d))


def _diagnostics(spec, d, samples):
    return build_diagnostics(_cfg("ULA", spec, d), None, samples)


def _typical_points(spec, d, n, rng):
    """Points drawn from the target itself -- where gradient accuracy matters."""
    x = targets.exact_samples(spec["name"], d, n, rng, nu=spec["nu"] or 5.0,
                              kappa=spec["kappa"] or 100.0)
    if spec["name"] == "double_well":
        # U = x^2 - |x| has a kink at 0; finite differences are meaningless
        # there, so push points clear of the non-differentiable point.
        x = np.where(np.abs(x) < 0.15, np.sign(x + 1e-12) * 0.15, x)
    return x


# ---------------------------------------------------------------------------
# Independently written reference log-densities (NOT derived from targets.py).
# Each returns log pi(x) up to an additive constant that must be the same for
# every x -- which is exactly what the test asserts.
# ---------------------------------------------------------------------------
def _mvt_logpdf(x, nu, d):
    """Isotropic multivariate Student-t (nu -> 1 gives multivariate Cauchy)."""
    q = np.sum(x**2)
    return (gammaln((nu + d) / 2.0) - gammaln(nu / 2.0) - (d / 2.0) * np.log(nu * np.pi)
            - ((nu + d) / 2.0) * np.log1p(q / nu))


def _reference_logpdf(name, x, nu=None, kappa=None):
    d = x.shape[0]
    if name == "gaussian":
        return float(np.sum(norm.logpdf(x)))
    if name == "student_t":
        return float(_mvt_logpdf(x, nu, d))
    if name == "cauchy":
        return float(_mvt_logpdf(x, 1.0, d))
    if name == "anisotropic_gaussian":
        std = np.sqrt(targets.variances_for_kappa(d, kappa))
        return float(np.sum(norm.logpdf(x, loc=0.0, scale=std)))
    if name == "banana":
        # Pushforward of y ~ N(0, diag(V0,1)) per 2D block under a unit-Jacobian
        # triangular map; invert the map and evaluate the Gaussian density on y.
        V0, b = targets.BANANA_V0, targets.BANANA_B
        y = x.copy()
        y[1::2] = x[1::2] - b * (x[0::2] ** 2 - V0)
        lp = np.sum(norm.logpdf(y[0::2], loc=0.0, scale=np.sqrt(V0)))
        lp += np.sum(norm.logpdf(y[1::2]))
        return float(lp)
    if name == "gaussian_mixture":
        centers = np.asarray(targets.MIX_CENTERS, dtype=float)
        comp = []
        for c in centers:
            mu = np.zeros(d)
            mu[0] = c
            comp.append(np.sum(norm.logpdf(x - mu)))
        return float(logsumexp(comp) - np.log(len(centers)))
    if name == "funnel":
        # Hierarchical: v ~ N(0, sigma_v^2), x_i | v ~ N(0, (e^{v/2})^2).
        sv = targets.FUNNEL_SIGMA_V
        v = x[0]
        lp = norm.logpdf(v, loc=0.0, scale=sv)
        lp += np.sum(norm.logpdf(x[1:], loc=0.0, scale=np.exp(v / 2.0)))
        return float(lp)
    if name == "double_well":
        return float(-np.sum(x**2 - np.abs(x)))
    raise ValueError(name)


# ===========================================================================
# Tests
# ===========================================================================
def test_potential_matches_reference_logpdf():
    """U(x) + log pi_ref(x) must be constant in x, for every target and d.

    This validates the potential against an independent closed form, including
    terms that only affect normalisation (the funnel's log-determinant term).
    """
    rng = np.random.default_rng(SEED)
    failures = []
    for spec in SPECS:
        for d in spec["dims"]:
            s = _sampler("MALA", spec, d)
            xs = _typical_points(spec, d, 40, rng)
            offs = np.array([
                s.f(x) + _reference_logpdf(spec["name"], x.copy(),
                                           nu=spec["nu"], kappa=spec["kappa"])
                for x in xs
            ])
            spread = float(np.max(offs) - np.min(offs))
            scale = max(1.0, float(np.max(np.abs(offs))))
            if spread / scale > 1e-9:
                failures.append(
                    f"{spec['name']} d={d}: U + log pi_ref varies by {spread:.3e} "
                    f"(offsets {np.min(offs):.6f}..{np.max(offs):.6f}) -- the "
                    f"potential does not match the reference density"
                )
    assert not failures, "\n".join(failures)


def test_gradient_matches_potential():
    """grad_f must be the finite-difference gradient of f, for MALA and BAOAB."""
    rng = np.random.default_rng(SEED + 1)
    failures = []
    for spec in SPECS:
        for d in spec["dims"]:
            for alg in ("MALA", "BAOAB"):
                s = _sampler(alg, spec, d)
                for x in _typical_points(spec, d, 6, rng):
                    g = s.grad_f(x)
                    for k in range(d):
                        h = 1e-6 * max(1.0, abs(x[k]))
                        xp, xm = x.copy(), x.copy()
                        xp[k] += h
                        xm[k] -= h
                        fd = (s.f(xp) - s.f(xm)) / (2 * h)
                        denom = max(1.0, abs(g[k]), abs(fd))
                        if abs(fd - g[k]) / denom > 1e-5:
                            failures.append(
                                f"{spec['name']} d={d} {alg} dim={k}: "
                                f"grad={g[k]:.8g} fd={fd:.8g}"
                            )
    assert not failures, "\n".join(failures[:20])


def test_all_three_samplers_share_one_potential():
    """ULA / MALA / BAOAB for a target must use identical gradients."""
    rng = np.random.default_rng(SEED + 2)
    failures = []
    for spec in SPECS:
        for d in spec["dims"]:
            u, m, b = (_sampler(a, spec, d) for a in ("ULA", "MALA", "BAOAB"))
            for x in _typical_points(spec, d, 8, rng):
                gu, gm, gb = u.grad_f(x), m.grad_f(x), b.grad_f(x)
                if not np.allclose(gu, gm, rtol=1e-12, atol=1e-12):
                    failures.append(f"{spec['name']} d={d}: ULA vs MALA gradient differs")
                if not np.allclose(gm, gb, rtol=1e-12, atol=1e-12):
                    failures.append(f"{spec['name']} d={d}: MALA vs BAOAB gradient differs")
                if not np.isclose(m.f(x), b.f(x), rtol=1e-12, atol=1e-12):
                    failures.append(f"{spec['name']} d={d}: MALA vs BAOAB potential differs")
    assert not failures, "\n".join(sorted(set(failures)))


# Estimators whose sampling variance is too large for a moment test at any
# practical n: the funnel's conditioned coordinates are a lognormal scale
# mixture with E[x^4] = 3 e^{2 sigma_v^2} ~ 2e8, and Cauchy has no moments.
_MOMENT_SKIP = {
    "cauchy": "no finite moments",
    "funnel": "conditioned coords are a heavy scale mixture (dim 0 still checked)",
}


def test_exact_samples_match_theoretical_moments():
    """Exact i.i.d. draws must reproduce the declared theoretical mean/variance.

    This is the test that catches a mismatch between the potential's structure
    and the declared moments -- e.g. a banana defined as a single 2D block but
    scored against product-of-blocks variances.
    """
    rng = np.random.default_rng(SEED + 3)
    n = 400_000
    failures = []
    for spec in SPECS:
        name = spec["name"]
        if name == "cauchy":
            continue
        for d in spec["dims"]:
            x = targets.exact_samples(name, d, n, rng, nu=spec["nu"] or 5.0,
                                      kappa=spec["kappa"] or 100.0)
            diag = _diagnostics(spec, d, x)
            emp_mean = x.mean(axis=0)
            emp_var = x.var(axis=0)
            for k in range(d):
                if name == "funnel" and k >= 1:
                    continue  # see _MOMENT_SKIP
                tv = diag.theoretical_variance(k)
                tm = diag.theoretical_mean(k)
                if np.isnan(tv):
                    continue
                # Student-t(5) has a large but finite 4th moment -> loose tol.
                rtol = 0.10 if name == "student_t" else 0.04
                if abs(emp_var[k] - tv) / tv > rtol:
                    failures.append(
                        f"{name} d={d} dim={k}: empirical var {emp_var[k]:.4f} vs "
                        f"theoretical {tv:.4f} (rel {abs(emp_var[k]-tv)/tv:.3f})"
                    )
                tol_mean = 6.0 * np.sqrt(tv / n) + 1e-3
                if abs(emp_mean[k] - tm) > tol_mean:
                    failures.append(
                        f"{name} d={d} dim={k}: empirical mean {emp_mean[k]:.4f} vs "
                        f"theoretical {tm:.4f} (tol {tol_mean:.4f})"
                    )
    assert not failures, "\n".join(failures)


def test_exact_samples_match_declared_cdf():
    """Exact i.i.d. draws must pass a KS test against Diagnostics.cdf per dim."""
    rng = np.random.default_rng(SEED + 4)
    n = 20_000
    failures = []
    for spec in SPECS:
        name = spec["name"]
        for d in spec["dims"]:
            x = targets.exact_samples(name, d, n, rng, nu=spec["nu"] or 5.0,
                                      kappa=spec["kappa"] or 100.0)
            diag = _diagnostics(spec, d, x)
            for k in range(d):
                probe = diag.cdf(np.array([0.0, 1.0]), k)
                if np.any(np.isnan(probe)):
                    continue  # no closed-form marginal for this dim, by design
                stat = kstest(x[:, k], lambda t, kk=k: diag.cdf(t, dim=kk)).statistic
                # 20k samples: the 1e-4 KS critical value is ~0.0139.
                if stat > 0.02:
                    failures.append(f"{name} d={d} dim={k}: KS vs declared cdf = {stat:.4f}")
    assert not failures, "\n".join(failures)


def test_cdf_quantile_roundtrip():
    """cdf(quantile(p)) == p wherever both are defined."""
    rng = np.random.default_rng(SEED + 5)
    ps = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    failures = []
    for spec in SPECS:
        for d in spec["dims"]:
            x = targets.exact_samples(spec["name"], d, 64, rng, nu=spec["nu"] or 5.0,
                                      kappa=spec["kappa"] or 100.0)
            diag = _diagnostics(spec, d, x)
            for k in range(d):
                for p in ps:
                    q = diag.quantile(p, dim=k)
                    if np.any(np.isnan(q)):
                        continue
                    back = diag.cdf(q, k)
                    if np.any(np.isnan(back)):
                        continue
                    if abs(float(back) - p) > 2e-3:
                        failures.append(
                            f"{spec['name']} d={d} dim={k} p={p}: "
                            f"cdf(quantile(p))={float(back):.5f}"
                        )
    assert not failures, "\n".join(failures)


def test_double_well_marginal_by_independent_quadrature():
    """The double-well grid marginal must match adaptive quadrature.

    targets.Numerical1D builds the marginal on a fixed trapezoidal grid; here we
    recompute the same normalising constant, mean and variance with scipy's
    adaptive quadrature, so a grid that is too coarse or too narrow is caught.
    """
    dens = lambda t: np.exp(-(t**2 - np.abs(t)))
    Z = quad(dens, -np.inf, np.inf)[0]
    m1 = quad(lambda t: t * dens(t), -np.inf, np.inf)[0] / Z
    m2 = quad(lambda t: t * t * dens(t), -np.inf, np.inf)[0] / Z
    var = m2 - m1**2

    grid = targets.DOUBLE_WELL_MARGINAL
    assert abs(grid.mean - m1) < 1e-8, f"mean {grid.mean} vs quad {m1}"
    assert abs(grid.var - var) / var < 1e-6, f"var {grid.var} vs quad {var}"

    # And the CDF at a few points, against quadrature of the same density.
    for t in (-1.5, -0.5, 0.0, 0.5, 1.5):
        ref = quad(dens, -np.inf, t)[0] / Z
        got = float(grid.cdf(t))
        assert abs(got - ref) < 1e-6, f"cdf({t}) = {got} vs quad {ref}"


def test_demo_only_targets():
    """Exponential and lognormal: gradients, moments and cdf/quantile agreement.

    These live in targets.py for the tutorial demos rather than the experiment
    grid, so they are not in SPECS (constrained support, and no entry in
    exact_samples). They still need pinning down.
    """
    rng = np.random.default_rng(SEED + 6)
    d, n = 3, 400_000
    failures = []

    cases = [
        ("exponential",
         targets.ExponentialMALA(d=d, eta=0.1, n_steps=1, burn_in=0, rate=1.0),
         targets.ExponentialULA(d=d, eta=0.1, n_steps=1, burn_in=0, rate=1.0),
         lambda s: targets.ExponentialDiagnostics(s, rate=1.0),
         lambda: rng.exponential(1.0, size=(n, d))),
        ("lognormal",
         targets.LognormalMALA(d=d, eta=0.1, n_steps=1, burn_in=0),
         targets.LognormalULA(d=d, eta=0.1, n_steps=1, burn_in=0),
         lambda s: targets.LognormalDiagnostics(s),
         lambda: rng.lognormal(0.0, 1.0, size=(n, d))),
    ]

    for name, mala, ula, make_diag, draw in cases:
        # Gradients, at interior points well away from the x = 0 boundary.
        pts = rng.uniform(0.3, 3.0, size=(6, d))
        for x in pts:
            g = mala.grad_f(x)
            if not np.allclose(g, ula.grad_f(x), rtol=1e-12, atol=1e-12):
                failures.append(f"{name}: ULA and MALA gradients differ")
            for k in range(d):
                h = 1e-7 * max(1.0, abs(x[k]))
                xp, xm = x.copy(), x.copy()
                xp[k] += h
                xm[k] -= h
                fd = (mala.f(xp) - mala.f(xm)) / (2 * h)
                if abs(fd - g[k]) / max(1.0, abs(g[k]), abs(fd)) > 1e-5:
                    failures.append(f"{name} dim={k}: grad={g[k]:.8g} fd={fd:.8g}")

        # Boundary handling: outside the support MALA must reject exactly.
        outside = np.full(d, -0.5)
        if not np.isinf(mala.f(outside)):
            failures.append(f"{name}: f is finite outside the support")

        # Moments and marginal cdf against i.i.d. draws from the true law.
        x = draw()
        diag = make_diag(x)
        for k in range(d):
            tv, tm = diag.theoretical_variance(k), diag.theoretical_mean(k)
            if abs(x[:, k].var() - tv) / tv > 0.05:
                failures.append(f"{name} dim={k}: var {x[:, k].var():.4f} vs {tv:.4f}")
            if abs(x[:, k].mean() - tm) / tm > 0.02:
                failures.append(f"{name} dim={k}: mean {x[:, k].mean():.4f} vs {tm:.4f}")
            stat = kstest(x[:, k], lambda t, kk=k: diag.cdf(t, dim=kk)).statistic
            if stat > 0.02:
                failures.append(f"{name} dim={k}: KS vs declared cdf = {stat:.4f}")
            for p in (0.05, 0.5, 0.95):
                back = float(diag.cdf(diag.quantile(p, dim=k), k))
                if abs(back - p) > 2e-3:
                    failures.append(f"{name} dim={k} p={p}: cdf(quantile(p))={back:.5f}")

    assert not failures, "\n".join(failures)


def test_banana_requires_even_dimension():
    """The product-of-2D-blocks structure is only defined for even d.

    Odd d must fail loudly at every entry point rather than silently producing
    a mis-shaped state or a bare broadcasting error.
    """
    spec = dict(name="banana", nu=None, kappa=None)
    for alg in ("ULA", "MALA", "BAOAB"):
        try:
            _sampler(alg, spec, 3)
        except (AssertionError, ValueError):
            continue
        raise AssertionError(f"banana {alg} accepted odd d=3")

    rng = np.random.default_rng(0)
    try:
        targets.exact_samples("banana", 3, 10, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("exact_samples accepted odd d=3 for banana")


# ===========================================================================
# Standalone runner (pytest is not installed in this environment)
# ===========================================================================
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}")
            for line in str(exc).splitlines():
                print(f"        {line}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
