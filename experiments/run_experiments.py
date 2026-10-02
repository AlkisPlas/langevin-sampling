"""
Run Langevin samplers (ULA, MALA, BAOAB) over a grid of parameters and write
diagnostics to an Excel workbook.

Targets come from targets/targets.py, which is the single source of truth
for every target distribution: the tail-weight family (Gaussian, Student-t,
Cauchy) and the geometric family (anisotropic Gaussian, banana, Gaussian
mixture, funnel, double well). This module only builds samplers/diagnostics
from those definitions and extracts metrics -- it defines no potentials itself.

Requirements: numpy, scipy, pandas, openpyxl

Run from the repo root:
    python -m experiments.run_experiments
    python experiments/run_experiments.py
    python experiments/run_experiments.py --preset small
    python experiments/run_experiments.py --n-steps 200000 --output my.xlsx
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kstest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets import targets


# ---------------------------------------------------------------------------
# Sampler / diagnostics factories.
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    algorithm: str       # "ULA" | "MALA" | "BAOAB"
    distribution: str    # "gaussian" | "cauchy" | "student_t"
    d: int
    eta: float
    gamma: float | None  # only for BAOAB
    nu: float | None     # only for student_t
    n_steps: int
    burn_in: int
    seed: int
    # Constructor parameters of the target, e.g. {"b": 0.03}; empty = target defaults.
    target_params: dict = field(default_factory=dict)

    def params_label(self) -> str:
        return ",".join(f"{k}={v}" for k, v in self.target_params.items())

    def label(self) -> str:
        bits = [self.algorithm, self.distribution, f"d={self.d}", f"eta={self.eta}"]
        if self.gamma is not None:
            bits.append(f"gamma={self.gamma}")
        if self.target_params:
            bits.append(self.params_label())
        if self.nu is not None:
            bits.append(f"nu={self.nu}")
        bits.append(f"seed={self.seed}")
        return " ".join(bits)


DISPERSION = 4.0  # over-dispersion factor for x0 initialisation

# The per_dim sheet is written only for d <= PER_DIM_MAX_D: no analysis reads it,
# and for d > 100 it would exceed the Excel row limit (plan §6, decision 12).
PER_DIM_MAX_D = 100


def sample_initial_state(distribution: str, d: int, seed: int,
                         dispersion: float = DISPERSION, params: dict | None = None):
    """
    Draw an over-dispersed initial position x0 (and equilibrium momentum v0
    for the kinetic scheme) using an independent RNG stream so that x0
    sampling does not consume the global PRNG state used by the chain
    innovations.

    - Gaussian: x0 ~ N(0, dispersion * I)  (target var=1)
    - Student-t(nu=5): x0 ~ 2 * t_5         (heavier than target)
    - Cauchy: x0 ~ dispersion * Cauchy(0,1) (heavier than target)
    - v0 ~ N(0, I)                          (equilibrium momentum)

    `params` are the target's constructor parameters (see ExperimentConfig);
    missing entries fall back to the target defaults.
    """
    params = params or {}
    rng = np.random.default_rng(seed)
    if distribution == "gaussian":
        x0 = rng.normal(0.0, np.sqrt(dispersion), d)
    elif distribution == "student_t":
        x0 = rng.standard_t(df=5.0, size=d) * 2.0
    elif distribution == "cauchy":
        x0 = rng.standard_cauchy(d) * dispersion
    elif distribution == "anisotropic_gaussian":
        variances = targets.variances_for_kappa(d, params.get("kappa", 100.0))
        x0 = rng.normal(0.0, 1.0, d) * np.sqrt(dispersion * variances)
    elif distribution == "banana":
        # Over-dispersed draw in the underlying y-space, then map onto the banana.
        if d % 2 != 0:
            raise ValueError("banana requires even d (product of 2D blocks)")
        V0 = params.get("V0", targets.BANANA_V0)
        b = params.get("b", targets.BANANA_B)
        y = rng.normal(0.0, 1.0, d)
        y[0::2] *= np.sqrt(dispersion * V0)
        y[1::2] *= np.sqrt(dispersion)
        x0 = y.copy()
        x0[1::2] = y[1::2] + b * (y[0::2] ** 2 - V0)
    elif distribution == "gaussian_mixture":
        # Start inside a randomly chosen mode (exposes mode-trapping).
        centers = params.get("centers", targets.MIX_CENTERS)
        x0 = rng.normal(0.0, np.sqrt(dispersion), d)
        x0[0] += centers[rng.integers(len(centers))]
    elif distribution == "funnel":
        # Exact hierarchical draw: v ~ N(0, sigma_v^2), x_i ~ N(0, e^v).
        sv = params.get("sigma_v", targets.FUNNEL_SIGMA_V)
        v = rng.normal(0.0, sv)
        x0 = rng.normal(0.0, np.exp(v / 2.0), d)
        x0[0] = v
    elif distribution == "double_well":
        x0 = rng.normal(0.0, np.sqrt(dispersion), d)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    v0 = rng.normal(0.0, 1.0, d)
    return x0, v0


def build_sampler(cfg: ExperimentConfig):
    p = cfg.target_params
    x0, v0 = sample_initial_state(cfg.distribution, cfg.d, cfg.seed, params=p)
    common = dict(d=cfg.d, eta=cfg.eta, n_steps=cfg.n_steps, burn_in=cfg.burn_in,
                  x0=x0, seed=cfg.seed)
    if cfg.distribution == "anisotropic_gaussian":
        if cfg.algorithm == "ULA":
            return targets.AnisotropicGaussianULA(**p, **common)
        if cfg.algorithm == "MALA":
            return targets.AnisotropicGaussianMALA(**p, **common)
        if cfg.algorithm == "BAOAB":
            return targets.AnisotropicGaussianBAOAB(gamma=cfg.gamma, v0=v0, **p, **common)
    if cfg.distribution == "banana":
        if cfg.algorithm == "ULA":
            return targets.BananaULA(**p, **common)
        if cfg.algorithm == "MALA":
            return targets.BananaMALA(**p, **common)
        if cfg.algorithm == "BAOAB":
            return targets.BananaBAOAB(gamma=cfg.gamma, v0=v0, **p, **common)
    if cfg.distribution == "gaussian_mixture":
        if cfg.algorithm == "ULA":
            return targets.GaussianMixtureULA(**p, **common)
        if cfg.algorithm == "MALA":
            return targets.GaussianMixtureMALA(**p, **common)
        if cfg.algorithm == "BAOAB":
            return targets.GaussianMixtureBAOAB(gamma=cfg.gamma, v0=v0, **p, **common)
    if cfg.distribution == "funnel":
        if cfg.algorithm == "ULA":
            return targets.FunnelULA(**p, **common)
        if cfg.algorithm == "MALA":
            return targets.FunnelMALA(**p, **common)
        if cfg.algorithm == "BAOAB":
            return targets.FunnelBAOAB(gamma=cfg.gamma, v0=v0, **p, **common)
    if cfg.distribution == "double_well":
        if cfg.algorithm == "ULA":
            return targets.DoubleWellULA(**p, **common)
        if cfg.algorithm == "MALA":
            return targets.DoubleWellMALA(**p, **common)
        if cfg.algorithm == "BAOAB":
            return targets.DoubleWellBAOAB(gamma=cfg.gamma, v0=v0, **p, **common)
    if cfg.distribution == "gaussian":
        if cfg.algorithm == "ULA":
            return targets.GaussianULA(**common)
        if cfg.algorithm == "MALA":
            return targets.GaussianMALA(**common)
        if cfg.algorithm == "BAOAB":
            return targets.GaussianBAOAB(gamma=cfg.gamma, v0=v0, **common)
    if cfg.distribution == "cauchy":
        if cfg.algorithm == "ULA":
            return targets.CauchyULA(**common)
        if cfg.algorithm == "MALA":
            return targets.CauchyMALA(**common)
        if cfg.algorithm == "BAOAB":
            return targets.CauchyBAOAB(gamma=cfg.gamma, v0=v0, **common)
    if cfg.distribution == "student_t":
        if cfg.algorithm == "ULA":
            return targets.StudentTULA(nu=cfg.nu, **common)
        if cfg.algorithm == "MALA":
            return targets.StudentTMALA(nu=cfg.nu, **common)
        if cfg.algorithm == "BAOAB":
            return targets.StudentTBAOAB(nu=cfg.nu, gamma=cfg.gamma, v0=v0, **common)
    raise ValueError(f"Unknown algorithm/distribution combo: {cfg.algorithm}/{cfg.distribution}")


def build_diagnostics(cfg: ExperimentConfig, sampler, samples_post):
    acc = getattr(sampler, "acceptance_rate", None) if cfg.algorithm == "MALA" else None
    p = cfg.target_params
    if cfg.distribution == "anisotropic_gaussian":
        return targets.AnisotropicGaussianDiagnostics(samples_post, acceptance_rate=acc, **p)
    if cfg.distribution == "banana":
        return targets.BananaDiagnostics(samples_post, acceptance_rate=acc, **p)
    if cfg.distribution == "gaussian_mixture":
        return targets.GaussianMixtureDiagnostics(samples_post, acceptance_rate=acc, **p)
    if cfg.distribution == "funnel":
        return targets.FunnelDiagnostics(samples_post, acceptance_rate=acc, **p)
    if cfg.distribution == "double_well":
        return targets.DoubleWellDiagnostics(samples_post, acceptance_rate=acc, **p)
    if cfg.distribution == "gaussian":
        return targets.GaussianDiagnostics(samples_post, mu=np.zeros(cfg.d), Sigma=np.eye(cfg.d),
                                   acceptance_rate=acc)
    if cfg.distribution == "cauchy":
        return targets.CauchyDiagnostics(samples_post, location=np.zeros(cfg.d), scale=1.0,
                                 acceptance_rate=acc)
    if cfg.distribution == "student_t":
        return targets.StudentTDiagnostics(samples_post, location=np.zeros(cfg.d),
                                   scale=np.ones(cfg.d), nu=cfg.nu, acceptance_rate=acc)
    raise ValueError(f"Unknown distribution: {cfg.distribution}")


# ---------------------------------------------------------------------------
# Metric extraction.
# ---------------------------------------------------------------------------


def _safe_mean_abs_bias(emp: np.ndarray, theor: np.ndarray) -> float:
    """Mean |emp - theor| over the dimensions with a theoretical value."""
    ok = ~np.isnan(theor)
    if not np.any(ok):
        return float("nan")
    return float(np.mean(np.abs(emp[ok] - theor[ok])))


def extract_metrics(cfg: ExperimentConfig, diagnostics, runtime: float):
    samples = diagnostics.samples
    d = diagnostics.d
    n_post = diagnostics.n_samples

    ess_per_dim = np.array(diagnostics.compute_ess(), dtype=float)
    ess_mean = float(np.mean(ess_per_dim))
    ess_min = float(np.min(ess_per_dim))
    ess_max = float(np.max(ess_per_dim))
    ess_ratio_mean = ess_mean / n_post if n_post else float("nan")
    iat_mean = n_post / ess_mean if ess_mean > 0 else float("nan")
    ess_per_sec = ess_mean / runtime if runtime > 0 else float("nan")

    max_lag = 200
    acf_stack = np.array([
        diagnostics.compute_autocorrelation(samples[:, k], max_lag=max_lag)
        for k in range(d)
    ])
    acf_avg = acf_stack.mean(axis=0)

    def acf_at(lag):
        return float(acf_avg[lag]) if lag < len(acf_avg) else float("nan")

    div_rate = float(diagnostics.compute_divergence_rate())

    qmae = diagnostics.compute_quantile_divergence()
    qmae_mean = float(np.mean(qmae)) if qmae is not None else float("nan")

    qmae_per_level: dict[float, float] = {}
    for q_level in diagnostics.quantile_levels:
        # Only the dimensions with a closed-form marginal.
        theor_qs = np.array([diagnostics.quantile(q_level, dim=k) for k in range(d)])
        ok = ~np.isnan(theor_qs)
        if not np.any(ok):
            qmae_per_level[q_level] = float("nan")
        else:
            emp_qs = np.array([np.percentile(samples[:, k], q_level * 100) for k in range(d)])
            qmae_per_level[q_level] = float(np.mean(np.abs(emp_qs[ok] - theor_qs[ok])))

    tail_summary: dict[float, float] = {}
    tail_stats = diagnostics.compute_tail_coverage()
    for q in (0.01, 0.05, 0.95, 0.99):
        if tail_stats is None:
            tail_summary[q] = float("nan")
        else:
            ratios = []
            for k in range(d):
                if f"dim_{k}" not in tail_stats:   # no closed-form marginal
                    continue
                entry = tail_stats[f"dim_{k}"][f"q_{q}"]
                ratios.append(entry["coverage"] / entry["expected"])
            tail_summary[q] = float(np.mean(ratios))

    emp_mean = samples.mean(axis=0)
    emp_var = samples.var(axis=0, ddof=1) if n_post > 1 else np.full(d, np.nan)
    emp_median = np.median(samples, axis=0)
    emp_std = np.sqrt(emp_var)

    theor_mean = np.array([diagnostics.theoretical_mean(k) for k in range(d)])
    theor_var = np.array([diagnostics.theoretical_variance(k) for k in range(d)])
    theor_median = np.array([diagnostics.quantile(0.5, dim=k) for k in range(d)])

    ks_per_dim: list[float] = []
    for k in range(d):
        try:
            stat = float(kstest(samples[:, k], lambda x, kk=k: diagnostics.cdf(x, dim=kk)).statistic)
        except Exception:
            stat = float("nan")
        ks_per_dim.append(stat)
    ks_mean = float(np.nanmean(ks_per_dim)) if len(ks_per_dim) else float("nan")

    # b^2 (Hoffman & Sountsov 2022): (E_chain[f] - E[f])^2 / Var[f] for the
    # functions f = x_k**p that the target declares; NaN if it declares none.
    # b2_grouped: the mean within each group of identically distributed
    # coordinates (target method b2_group; one group by default), then the
    # maximum over the groups. It does not dilute a difficult group with easy
    # ones, and it does not grow with d (the number of groups is fixed).
    b2_fn = getattr(diagnostics, "b2_functions", None)
    group_fn = getattr(diagnostics, "b2_group", lambda k: 0)
    b2_per_dim = np.full(d, np.nan)
    b2_terms: list[float] = []
    b2_by_group: dict = {}
    if callable(b2_fn):
        for k in range(d):
            terms = [(float(np.mean(samples[:, k] ** p)) - e) ** 2 / v for p, e, v in b2_fn(k)]
            if terms:
                b2_per_dim[k] = float(np.mean(terms))
                b2_terms.extend(terms)
                b2_by_group.setdefault(group_fn(k), []).extend(terms)
    b2_avg = float(np.mean(b2_terms)) if b2_terms else float("nan")
    b2_grouped = (float(max(np.mean(t) for t in b2_by_group.values()))
                  if b2_by_group else float("nan"))

    # Mode-occupancy (only for multimodal targets that expose it).
    mode_fn = getattr(diagnostics, "mode_occupancy", None)
    if callable(mode_fn):
        mode_info = mode_fn()
        mode_occ_err = float(mode_info["occupancy_error"])
        mode_trans = float(mode_info["transition_rate"])
    else:
        mode_occ_err = float("nan")
        mode_trans = float("nan")

    summary_row: dict[str, Any] = {
        "algorithm": cfg.algorithm,
        "distribution": cfg.distribution,
        "d": cfg.d,
        "eta": cfg.eta,
        "gamma": cfg.gamma if cfg.gamma is not None else float("nan"),
        "target_params": cfg.params_label(),
        "nu": cfg.nu if cfg.nu is not None else float("nan"),
        "n_steps": cfg.n_steps,
        "burn_in": cfg.burn_in,
        "n_samples_post": n_post,
        "seed": cfg.seed,
        "runtime_seconds": runtime,
        "acceptance_rate": diagnostics.acceptance_rate
        if diagnostics.acceptance_rate is not None
        else float("nan"),
        "divergence_rate": div_rate,
        "ess_mean": ess_mean,
        "ess_min": ess_min,
        "ess_max": ess_max,
        "ess_ratio_mean": ess_ratio_mean,
        "iat_mean": iat_mean,
        "ess_per_sec": ess_per_sec,
        "acf_lag_1": acf_at(1),
        "acf_lag_10": acf_at(10),
        "acf_lag_50": acf_at(50),
        "acf_lag_100": acf_at(100),
        "empirical_mean_avg": float(np.mean(emp_mean)),
        "theoretical_mean_avg": float(np.mean(theor_mean)) if not np.any(np.isnan(theor_mean)) else float("nan"),
        "mean_abs_bias": _safe_mean_abs_bias(emp_mean, theor_mean),
        "empirical_var_avg": float(np.mean(emp_var)),
        "theoretical_var_avg": float(np.mean(theor_var)) if not np.any(np.isnan(theor_var)) else float("nan"),
        "var_abs_bias": _safe_mean_abs_bias(emp_var, theor_var),
        "empirical_std_avg": float(np.mean(emp_std)),
        "empirical_median_avg": float(np.mean(emp_median)),
        "theoretical_median_avg": float(np.mean(theor_median)) if not np.any(np.isnan(theor_median)) else float("nan"),
        "median_abs_bias": _safe_mean_abs_bias(emp_median, theor_median),
        "quantile_mae_avg": qmae_mean,
        "quantile_mae_q0.025": qmae_per_level.get(0.025, float("nan")),
        "quantile_mae_q0.5": qmae_per_level.get(0.5, float("nan")),
        "quantile_mae_q0.975": qmae_per_level.get(0.975, float("nan")),
        "tail_cov_ratio_q0.01": tail_summary[0.01],
        "tail_cov_ratio_q0.05": tail_summary[0.05],
        "tail_cov_ratio_q0.95": tail_summary[0.95],
        "tail_cov_ratio_q0.99": tail_summary[0.99],
        "ks_stat_mean": ks_mean,
        "mode_occupancy_error": mode_occ_err,
        "mode_transition_rate": mode_trans,
        "b2_avg": b2_avg,
        "b2_grouped": b2_grouped,
    }

    per_dim_rows: list[dict[str, Any]] = []
    for k in range(d):
        per_dim_rows.append({
            "algorithm": cfg.algorithm,
            "distribution": cfg.distribution,
            "d": cfg.d,
            "eta": cfg.eta,
            "gamma": cfg.gamma if cfg.gamma is not None else float("nan"),
            "target_params": cfg.params_label(),
            "nu": cfg.nu if cfg.nu is not None else float("nan"),
            "seed": cfg.seed,
            "dim": k,
            "ess": float(ess_per_dim[k]),
            "ess_ratio": float(ess_per_dim[k] / n_post) if n_post else float("nan"),
            "iat": float(n_post / ess_per_dim[k]) if ess_per_dim[k] > 0 else float("nan"),
            "empirical_mean": float(emp_mean[k]),
            "theoretical_mean": float(theor_mean[k]) if not np.isnan(theor_mean[k]) else float("nan"),
            "empirical_var": float(emp_var[k]),
            "theoretical_var": float(theor_var[k]) if not np.isnan(theor_var[k]) else float("nan"),
            "empirical_std": float(emp_std[k]),
            "empirical_median": float(emp_median[k]),
            "theoretical_median": float(theor_median[k]) if not np.isnan(theor_median[k]) else float("nan"),
            "ks_stat": float(ks_per_dim[k]),
            "b2": float(b2_per_dim[k]),
        })

    return summary_row, per_dim_rows


# ---------------------------------------------------------------------------
# Parameter grid presets.
# ---------------------------------------------------------------------------


PRESETS = {
    "smoke": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian", "cauchy", "student_t"],
        "d_values": [1],
        "eta": {
            "ULA":   {"gaussian": [0.1], "cauchy": [0.1], "student_t": [0.3]},
            "MALA":  {"gaussian": [0.5], "cauchy": [1.0], "student_t": [0.5]},
            "BAOAB": {"gaussian": [0.5], "cauchy": [0.1], "student_t": [0.5]},
        },
        "gamma_values": [1.0],
        "n_steps": 5_000,
        "burn_in": 500,
        "nu": 5.0,
        "seeds": [0, 1, 2],
    },
    "small": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian", "cauchy", "student_t"],
        "d_values": [1, 5],
        "eta": {
            "ULA":   {"gaussian": [0.05, 0.1], "cauchy": [0.05, 0.1], "student_t": [0.1, 0.3]},
            "MALA":  {"gaussian": [0.5, 1.0],  "cauchy": [1.0, 2.0],  "student_t": [0.3, 0.7]},
            "BAOAB": {"gaussian": [0.3, 0.7],  "cauchy": [0.05, 0.1], "student_t": [0.3, 0.6]},
        },
        "gamma_values": [1.0, 2.0],
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": [0, 1, 2, 3, 4],
    },
    "default": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian", "cauchy", "student_t"],
        "d_values": [1, 3, 5, 10],
        "eta": {
            "ULA":   {"gaussian": [0.01, 0.05, 0.1, 0.3],
                       "cauchy":   [0.05, 0.1, 0.2, 0.5],
                       "student_t":[0.05, 0.1, 0.3, 0.6]},
            "MALA":  {"gaussian": [0.1, 0.5, 1.0, 1.5],
                       "cauchy":   [0.5, 1.0, 2.0, 4.0],
                       "student_t":[0.1, 0.3, 0.7, 1.0]},
            "BAOAB": {"gaussian": [0.1, 0.3, 0.5, 1.0],
                       "cauchy":   [0.05, 0.1, 0.2, 0.5],
                       "student_t":[0.1, 0.3, 0.6, 1.0]},
        },
        "gamma_values": [0.5, 1.0, 2.0, 5.0],
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    "large": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian", "cauchy", "student_t"],
        "d_values": [1, 3, 5, 10, 20],
        "eta": {
            "ULA":   {"gaussian": [0.005, 0.01, 0.05, 0.1, 0.3],
                       "cauchy":   [0.02, 0.05, 0.1, 0.2, 0.5],
                       "student_t":[0.02, 0.05, 0.1, 0.3, 0.6]},
            "MALA":  {"gaussian": [0.05, 0.1, 0.5, 1.0, 1.5],
                       "cauchy":   [0.2, 0.5, 1.0, 2.0, 4.0],
                       "student_t":[0.05, 0.1, 0.3, 0.7, 1.0]},
            "BAOAB": {"gaussian": [0.05, 0.1, 0.3, 0.5, 1.0],
                       "cauchy":   [0.02, 0.05, 0.1, 0.2, 0.5],
                       "student_t":[0.05, 0.1, 0.3, 0.6, 1.0]},
        },
        "gamma_values": [0.5, 1.0, 2.0, 5.0],
        "n_steps": 100_000,
        "burn_in": 10_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # High-dimension extension preset. Runs *only* d in {20, 50, 100} on the
    # three targets; the eta grids are anchored on the empirical d=10
    # optima (rather than on Roberts-Rosenthal predictions) and extended in
    # the direction the d=1->d=10 trend suggested. This preset does NOT
    # overlap with the `default` preset's d in {1, 3, 5, 10}, so the two
    # xlsx outputs can be aggregated independently and concatenated at
    # figure-generation time.
    "high_d_only": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian", "student_t", "cauchy"],
        "d_values": [20, 50, 100],
        "eta": {
            "ULA":   {"gaussian":  [0.01, 0.03, 0.10, 0.30],
                       "student_t": [0.01, 0.03, 0.05, 0.15],
                       "cauchy":    [0.03, 0.10, 0.20, 0.50]},
            "MALA":  {"gaussian":  [0.10, 0.30, 0.50, 1.00],
                       "student_t": [0.05, 0.15, 0.30, 0.70],
                       "cauchy":    [0.10, 0.30, 1.00, 2.00]},
            "BAOAB": {"gaussian":  [0.30, 0.50, 1.00, 1.50],
                       "student_t": [0.10, 0.30, 0.60, 1.00],
                       "cauchy":    [0.10, 0.30, 0.50, 1.00]},
        },
        "gamma_values": [0.5, 1.0, 2.0, 5.0],
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # Tiny d=50 variant for smoke-testing the high_d_only grid before
    # committing to the full overnight run.
    "high_d_smoke": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian", "student_t", "cauchy"],
        "d_values": [50],
        "eta": {
            "ULA":   {"gaussian":  [0.01, 0.10],
                       "student_t": [0.01, 0.05],
                       "cauchy":    [0.03, 0.20]},
            "MALA":  {"gaussian":  [0.10, 0.50],
                       "student_t": [0.05, 0.30],
                       "cauchy":    [0.10, 1.00]},
            "BAOAB": {"gaussian":  [0.30, 1.00],
                       "student_t": [0.10, 0.60],
                       "cauchy":    [0.10, 0.50]},
        },
        "gamma_values": [0.5, 2.0],
        "n_steps": 10_000,
        "burn_in": 1_000,
        "nu": 5.0,
        "seeds": [0, 1, 2],
    },
    # ----- Extended geometric targets (anisotropic / banana / mixture / funnel /
    # double-well). d_values are EVEN because the banana is a product of 2D blocks.
    # target_params: constructor parameters per target, one run per entry.
    "extended_smoke": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["anisotropic_gaussian", "banana", "gaussian_mixture",
                          "funnel", "double_well"],
        "d_values": [2],
        "eta": {
            "ULA":   {"anisotropic_gaussian": [0.1], "banana": [0.01],
                       "gaussian_mixture": [0.1], "funnel": [0.01], "double_well": [0.1]},
            "MALA":  {"anisotropic_gaussian": [0.5], "banana": [0.1],
                       "gaussian_mixture": [0.5], "funnel": [0.1], "double_well": [0.5]},
            "BAOAB": {"anisotropic_gaussian": [0.2], "banana": [0.05],
                       "gaussian_mixture": [0.2], "funnel": [0.05], "double_well": [0.2]},
        },
        "gamma_values": [1.0],
        "target_params": {"anisotropic_gaussian": [{"kappa": 100.0}]},
        "n_steps": 5_000,
        "burn_in": 500,
        "nu": 5.0,
        "seeds": [0, 1],
    },
    "extended": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["anisotropic_gaussian", "banana", "gaussian_mixture",
                          "funnel", "double_well"],
        "d_values": [2, 4, 10],
        # eta grids tuned via a coarse d=10 sweep (divergence-free; MALA acceptance
        # kept within [~0.3, 0.95]; ULA/BAOAB chosen near best KS/ESS).
        "eta": {
            "ULA":   {"anisotropic_gaussian": [0.1, 0.3, 0.5, 1.0],
                       "banana": [0.005, 0.01, 0.02, 0.05],
                       "gaussian_mixture": [0.05, 0.1, 0.3, 0.5],
                       "funnel": [0.005, 0.01, 0.02, 0.05],
                       "double_well": [0.05, 0.1, 0.3, 0.5]},
            "MALA":  {"anisotropic_gaussian": [0.3, 0.7, 1.0, 2.0],
                       "banana": [0.05, 0.1, 0.2, 0.5],
                       "gaussian_mixture": [0.3, 0.5, 1.0],
                       "funnel": [0.02, 0.05, 0.1, 0.2],
                       "double_well": [0.2, 0.3, 0.5]},
            "BAOAB": {"anisotropic_gaussian": [0.3, 0.7, 1.0, 1.5],
                       "banana": [0.02, 0.05, 0.1, 0.2],
                       "gaussian_mixture": [0.1, 0.3, 0.5, 1.0],
                       "funnel": [0.01, 0.02, 0.05, 0.1],
                       "double_well": [0.1, 0.3, 0.5]},
        },
        "gamma_values": [0.5, 1.0, 2.0],
        "target_params": {"anisotropic_gaussian": [{"kappa": 10.0}, {"kappa": 100.0},
                                                   {"kappa": 1000.0}]},
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # ----- Calibration study (thesis/experiment_design_extension.md, §4): three
    # candidate values of the difficulty parameter per target, at d = 10. The eta
    # grids extend the `extended` grids upwards (best eta was at the top edge there)
    # and, where a candidate value is harder, downwards. They were extended once more
    # after the short check of step 7 (best eta at an edge; see §4.8 of the plan).
    "calibration": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["anisotropic_gaussian", "banana", "gaussian_mixture",
                          "funnel", "double_well"],
        "d_values": [10],
        "eta": {
            "ULA":   {"anisotropic_gaussian": [0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5],
                       "banana": [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5],
                       "gaussian_mixture": [0.05, 0.1, 0.3, 0.5, 1.0],
                       "funnel": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                       "double_well": [0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]},
            "MALA":  {"anisotropic_gaussian": [0.3, 0.7, 1.0, 2.0, 3.0],
                       "banana": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                       "gaussian_mixture": [0.1, 0.2, 0.3, 0.5, 1.0, 1.5],
                       "funnel": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                       "double_well": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]},
            "BAOAB": {"anisotropic_gaussian": [0.3, 0.7, 1.0, 1.5, 1.8, 1.9],
                       "banana": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                       "gaussian_mixture": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
                       "funnel": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                       "double_well": [0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]},
        },
        "gamma_values": [0.5, 1.0, 2.0],
        "target_params": {
            "anisotropic_gaussian": [{"kappa": 10.0}, {"kappa": 100.0}, {"kappa": 1000.0}],
            "banana":               [{"b": 0.01}, {"b": 0.03}, {"b": 0.1}],
            "gaussian_mixture":     [{"centers": (-4.0, 0.0, 4.0)},
                                     {"centers": (-6.0, 0.0, 6.0)},
                                     {"centers": (-8.0, 0.0, 8.0)}],
            "funnel":               [{"sigma_v": 1.0}, {"sigma_v": 2.0}, {"sigma_v": 3.0}],
            "double_well":          [{"beta": 1.0}, {"beta": 4.0}, {"beta": 8.0}],
        },
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # ----- Dimensional scaling, Part B, stage 1 (plan §10, E3): the five geometric
    # targets at their calibrated difficulty (plan §4.8, step 8, table T2), for
    # d <= 100. The eta grids are the calibration grids (checked for edges at d = 10);
    # for the double well (beta = 16) they are the `calibration_extra` grids.
    "scaling_B1": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["anisotropic_gaussian", "banana", "gaussian_mixture",
                          "funnel", "double_well"],
        "d_values": [2, 4, 10, 20, 50, 100],
        "eta": {
            "ULA":   {"anisotropic_gaussian": [0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5],
                       "banana": [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5],
                       "gaussian_mixture": [0.05, 0.1, 0.3, 0.5, 1.0],
                       "funnel": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                       "double_well": [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.05]},
            "MALA":  {"anisotropic_gaussian": [0.3, 0.7, 1.0, 2.0, 3.0],
                       "banana": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                       "gaussian_mixture": [0.1, 0.2, 0.3, 0.5, 1.0, 1.5],
                       "funnel": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                       "double_well": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]},
            "BAOAB": {"anisotropic_gaussian": [0.3, 0.7, 1.0, 1.5, 1.8, 1.9],
                       "banana": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                       "gaussian_mixture": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
                       "funnel": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                       "double_well": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3]},
        },
        "gamma_values": [0.5, 1.0, 2.0],
        "target_params": {
            "anisotropic_gaussian": [{"kappa": 1000.0}],
            "banana":               [{"b": 0.03}],
            "gaussian_mixture":     [{"centers": (-6.0, 0.0, 6.0)}],
            "funnel":               [{"sigma_v": 2.0}],
            "double_well":          [{"beta": 16.0}],
        },
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # ----- Additional eta values for `scaling_B1` (plan §4.8, step 9): the best eta
    # was at a grid edge for these (target, algorithm) pairs. Same settings otherwise.
    "scaling_B1_extra": {
        "algorithms": ["ULA", "MALA"],
        "distributions": ["anisotropic_gaussian", "banana", "funnel"],
        "d_values": [2, 4, 10, 20, 50, 100],
        "eta": {
            "ULA":  {"anisotropic_gaussian": [], "banana": [], "funnel": [0.001, 0.002]},
            "MALA": {"anisotropic_gaussian": [4.0, 5.0], "banana": [3.0, 5.0],
                      "funnel": [0.005, 0.01]},
        },
        "gamma_values": [0.5, 1.0, 2.0],
        "target_params": {
            "anisotropic_gaussian": [{"kappa": 1000.0}],
            "banana":               [{"b": 0.03}],
            "funnel":               [{"sigma_v": 2.0}],
        },
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # ----- Dimensional scaling, Part B, stage 2 (plan §4.8, step 9): d in {200, 500,
    # 1000}. Grids = best eta at d = 100 of stage 1 x {0.25, 0.5, 1, 1.5}, plus x0.125
    # where the best eta decreased from d = 50 to d = 100.
    "scaling_B2": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["anisotropic_gaussian", "banana", "gaussian_mixture",
                          "funnel", "double_well"],
        "d_values": [200, 500, 1000],
        "eta": {
            "ULA":   {"anisotropic_gaussian": [0.12, 0.25, 0.5, 0.75],
                       "banana": [0.12, 0.25, 0.5, 0.75],
                       "gaussian_mixture": [0.075, 0.15, 0.3, 0.45],
                       "funnel": [0.0006, 0.0013, 0.0025, 0.005, 0.0075],
                       "double_well": [0.0025, 0.005, 0.01, 0.015]},
            "MALA":  {"anisotropic_gaussian": [0.09, 0.17, 0.35, 0.7, 1.0],
                       "banana": [0.05, 0.1, 0.2, 0.3],
                       "gaussian_mixture": [0.04, 0.075, 0.15, 0.3, 0.45],
                       "funnel": [0.0013, 0.0025, 0.005, 0.01, 0.015],
                       "double_well": [0.0006, 0.0013, 0.0025, 0.005, 0.0075]},
            "BAOAB": {"anisotropic_gaussian": [1.0, 1.5, 1.8, 1.9],
                       "banana": [0.06, 0.12, 0.25, 0.5, 0.75],
                       "gaussian_mixture": [0.25, 0.5, 1.0, 1.5],
                       "funnel": [0.025, 0.05, 0.1, 0.15],
                       "double_well": [0.025, 0.05, 0.1, 0.15]},
        },
        "gamma_values": [0.5, 1.0, 2.0],
        "target_params": {
            "anisotropic_gaussian": [{"kappa": 1000.0}],
            "banana":               [{"b": 0.03}],
            "gaussian_mixture":     [{"centers": (-6.0, 0.0, 6.0)}],
            "funnel":               [{"sigma_v": 2.0}],
            "double_well":          [{"beta": 16.0}],
        },
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # ----- Dimensional scaling, Part A, d in {200, 500, 1000} (plan §6, decision 12).
    # The eta grids extend past the d = 100 optima of `high_d_only`, which were at a
    # grid edge in six of nine cases (Εργασία §4.6, item 3); BAOAB on the Gaussian
    # stops below its stability limit eta = 2. gamma as in the original thesis.
    "scaling_A_high": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian", "student_t", "cauchy"],
        "d_values": [200, 500, 1000],
        "eta": {
            "ULA":   {"gaussian":  [0.03, 0.05, 0.1, 0.2, 0.3],
                       "student_t": [0.002, 0.005, 0.01, 0.02, 0.03],
                       "cauchy":    [0.2, 0.5, 1.0, 2.0]},
            "MALA":  {"gaussian":  [0.1, 0.15, 0.2, 0.3, 0.5],
                       "student_t": [0.05, 0.1, 0.15, 0.3],
                       "cauchy":    [1.0, 2.0, 4.0, 8.0]},
            "BAOAB": {"gaussian":  [1.0, 1.5, 1.8, 1.9],
                       "student_t": [0.6, 1.0, 1.5, 2.0],
                       "cauchy":    [0.5, 1.0, 2.0, 4.0]},
        },
        "gamma_values": [0.5, 1.0, 2.0, 5.0],
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # ----- Additional eta values for `scaling_B2` and `scaling_A_high` (plan §4.8,
    # step 9b): the best eta was at a grid edge for these (target, algorithm) pairs,
    # after the divergence fix. Same settings as the base presets otherwise.
    "scaling_B2_extra": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["banana", "gaussian_mixture", "funnel", "double_well"],
        "d_values": [200, 500, 1000],
        "eta": {
            "ULA":   {"banana": [0.03, 0.06], "gaussian_mixture": [],
                       "funnel": [0.01, 0.015], "double_well": [0.02, 0.03]},
            "MALA":  {"banana": [0.0125, 0.025], "gaussian_mixture": [0.01, 0.02],
                       "funnel": [0.0003, 0.0006, 0.02, 0.03], "double_well": []},
            "BAOAB": {"banana": [], "gaussian_mixture": [2.0, 3.0],
                       "funnel": [0.006, 0.0125], "double_well": [0.2, 0.3]},
        },
        "gamma_values": [0.5, 1.0, 2.0],
        "target_params": {
            "banana":           [{"b": 0.03}],
            "gaussian_mixture": [{"centers": (-6.0, 0.0, 6.0)}],
            "funnel":           [{"sigma_v": 2.0}],
            "double_well":      [{"beta": 16.0}],
        },
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    "scaling_A_high_extra": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian", "student_t", "cauchy"],
        "d_values": [200, 500, 1000],
        "eta": {
            "ULA":   {"gaussian": [], "student_t": [0.05, 0.1], "cauchy": [4.0, 8.0]},
            "MALA":  {"gaussian": [0.05, 0.07], "student_t": [],
                       "cauchy": [0.1, 0.3, 0.5, 16.0, 32.0]},
            "BAOAB": {"gaussian": [], "student_t": [], "cauchy": [8.0, 16.0]},
        },
        "gamma_values": [0.5, 1.0, 2.0, 5.0],
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
    # ----- Additional calibration values (plan §4.8, step 8): the mixture changes
    # from "all succeed" (a=6) to "all fail" (a=8), so a=7 is added; the double well
    # is easy for every beta in {1, 4, 8}, so beta=16, 32 are added. Same settings as
    # `calibration`; the double-well eta grids extend downwards (best eta ~ 1/beta).
    "calibration_extra": {
        "algorithms": ["ULA", "MALA", "BAOAB"],
        "distributions": ["gaussian_mixture", "double_well"],
        "d_values": [10],
        "eta": {
            "ULA":   {"gaussian_mixture": [0.05, 0.1, 0.3, 0.5, 1.0],
                       "double_well": [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.05]},
            "MALA":  {"gaussian_mixture": [0.1, 0.2, 0.3, 0.5, 1.0, 1.5],
                       "double_well": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]},
            "BAOAB": {"gaussian_mixture": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
                       "double_well": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3]},
        },
        "gamma_values": [0.5, 1.0, 2.0],
        "target_params": {
            "gaussian_mixture": [{"centers": (-7.0, 0.0, 7.0)}],
            "double_well":      [{"beta": 16.0}, {"beta": 32.0}],
        },
        "n_steps": 50_000,
        "burn_in": 5_000,
        "nu": 5.0,
        "seeds": list(range(20)),
    },
}


def build_configs(preset: dict, n_steps_override: int | None = None,
                  burn_in_override: int | None = None) -> list[ExperimentConfig]:
    n_steps = n_steps_override if n_steps_override is not None else preset["n_steps"]
    burn_in = burn_in_override if burn_in_override is not None else preset["burn_in"]
    nu = preset["nu"]
    configs: list[ExperimentConfig] = []
    for algorithm, distribution, d, seed in itertools.product(
        preset["algorithms"], preset["distributions"], preset["d_values"], preset["seeds"]
    ):
        eta_list = preset["eta"][algorithm][distribution]
        # One run per entry of target_params; a missing target uses its defaults.
        params_list = preset.get("target_params", {}).get(distribution, [{}])
        nu_val = nu if distribution == "student_t" else None
        for eta, params in itertools.product(eta_list, params_list):
            gamma_list = preset["gamma_values"] if algorithm == "BAOAB" else [None]
            for gamma in gamma_list:
                configs.append(ExperimentConfig(
                    algorithm=algorithm, distribution=distribution, d=d,
                    eta=eta, gamma=gamma, nu=nu_val,
                    n_steps=n_steps, burn_in=burn_in, seed=seed, target_params=dict(params),
                ))
    return configs


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


def run_one(cfg: ExperimentConfig):
    sampler = build_sampler(cfg)
    t0 = time.perf_counter()
    samples_post = sampler.run()
    runtime = time.perf_counter() - t0
    diagnostics = build_diagnostics(cfg, sampler, samples_post)
    summary_row, per_dim_rows = extract_metrics(cfg, diagnostics, runtime)
    return summary_row, per_dim_rows


def write_excel(output_path: str, summary_rows: list[dict], per_dim_rows: list[dict],
                preset_name: str, preset: dict, n_steps: int, burn_in: int):
    summary_df = pd.DataFrame(summary_rows)
    per_dim_df = pd.DataFrame(per_dim_rows)

    config_records = [
        {"key": "preset", "value": preset_name},
        {"key": "algorithms", "value": ", ".join(preset["algorithms"])},
        {"key": "distributions", "value": ", ".join(preset["distributions"])},
        {"key": "d_values", "value": ", ".join(map(str, preset["d_values"]))},
        {"key": "gamma_values", "value": ", ".join(map(str, preset["gamma_values"]))},
        {"key": "nu", "value": preset["nu"]},
        {"key": "n_steps", "value": n_steps},
        {"key": "burn_in", "value": burn_in},
        {"key": "seeds", "value": ", ".join(map(str, preset["seeds"]))},
        {"key": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    config_df = pd.DataFrame(config_records)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        per_dim_df.to_excel(writer, sheet_name="per_dim", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)


def _run_safe(cfg: ExperimentConfig):
    """run_one for one chain; returns (summary_row, per_dim_rows, seconds, error)."""
    t0 = time.perf_counter()
    try:
        s_row, d_rows = run_one(cfg)
        return s_row, d_rows, time.perf_counter() - t0, None
    except Exception as exc:  # noqa: BLE001 - report the failure and keep going
        return None, None, time.perf_counter() - t0, str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="default",
                        help="Parameter-grid preset (default: 'default')")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Override n_steps for all runs")
    parser.add_argument("--burn-in", type=int, default=None,
                        help="Override burn_in for all runs")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel processes (default: 1). Results are identical; "
                             "only the time columns (runtime_seconds, ess_per_sec) can differ.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output xlsx path (default: experiments/results/extension/results_<preset>_<ts>.xlsx)")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    configs = build_configs(preset, n_steps_override=args.n_steps, burn_in_override=args.burn_in)
    n_steps_used = args.n_steps if args.n_steps is not None else preset["n_steps"]
    burn_in_used = args.burn_in if args.burn_in is not None else preset["burn_in"]

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(_HERE, "results", "extension", f"results_{args.preset}_{ts}.xlsx")
    else:
        output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Preset: {args.preset}  | runs: {len(configs)}  | n_steps={n_steps_used}, burn_in={burn_in_used}"
          f"  | workers: {args.workers}")
    print(f"Output: {output_path}\n")

    summary_rows: list[dict] = []
    per_dim_rows: list[dict] = []

    t_start = time.perf_counter()
    # Chains are independent (each has its own seed), so they can run in parallel
    # processes. executor.map returns the results in the order of `configs`, so the
    # output rows are in the same order as in a single-process run.
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(_run_safe, configs, chunksize=1)
    else:
        executor = None
        results = map(_run_safe, configs)
    for idx, (cfg, (s_row, d_rows, dt, error)) in enumerate(zip(configs, results), start=1):
        if error is None:
            summary_rows.append(s_row)
            if cfg.d <= PER_DIM_MAX_D:
                per_dim_rows.extend(d_rows)
            print(f"[{idx:>4}/{len(configs)}] OK ({dt:6.2f}s) {cfg.label()}", flush=True)
        else:
            print(f"[{idx:>4}/{len(configs)}] FAIL {cfg.label()}: {error}", flush=True)
            summary_rows.append({
                "algorithm": cfg.algorithm,
                "distribution": cfg.distribution,
                "d": cfg.d,
                "eta": cfg.eta,
                "gamma": cfg.gamma if cfg.gamma is not None else float("nan"),
                "target_params": cfg.params_label(),
                "nu": cfg.nu if cfg.nu is not None else float("nan"),
                "n_steps": cfg.n_steps,
                "burn_in": cfg.burn_in,
                "seed": cfg.seed,
                "error": error,
            })
    if executor is not None:
        executor.shutdown()

    total_dt = time.perf_counter() - t_start
    write_excel(output_path, summary_rows, per_dim_rows, args.preset, preset,
                n_steps_used, burn_in_used)
    print(f"\nFinished {len(configs)} runs in {total_dt:.1f}s")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
