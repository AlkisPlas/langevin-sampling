"""
Generate trace plots, autocorrelation plots, and goodness-of-fit plots
(histogram + Q-Q) from comprehensive_diagnostics for five representative
configurations from the 288-run grid study. Saves PDF + PNG to
experiments/figures/extension/case_studies/.

Each case re-runs the sampler with fixed parameters and seed; no results file
is read. (The cases were first picked from a single-seed pilot grid run,
results_default_20260523_200256.xlsx, which is no longer kept.)
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.run_experiments import (  # noqa: E402
    ExperimentConfig,
    build_diagnostics,
    build_sampler,
)

FIG_DIR = os.path.join(_HERE, "figures", "extension", "case_studies")
os.makedirs(FIG_DIR, exist_ok=True)

# Four representative cases at the multi-seed best configurations.
CASES: list[dict] = [
    {
        "label": "01_mala_gaussian_gold_standard",
        "title": "MALA Gaussian, $d{=}1$, $\\eta{=}1.0$ (reference: near-IID sampling)",
        "algorithm": "MALA", "distribution": "gaussian",
        "d": 1, "eta": 1.0, "gamma": None, "nu": None,
    },
    {
        "label": "02_ula_gaussian_mild_bias",
        "title": "ULA Gaussian, $d{=}1$, $\\eta{=}0.30$ (mild discretisation bias)",
        "algorithm": "ULA", "distribution": "gaussian",
        "d": 1, "eta": 0.30, "gamma": None, "nu": None,
    },
    {
        "label": "03_ula_cauchy_tail_collapse",
        "title": "ULA Cauchy, $d{=}1$, $\\eta{=}0.10$ (heavy-tail collapse)",
        "algorithm": "ULA", "distribution": "cauchy",
        "d": 1, "eta": 0.10, "gamma": None, "nu": None,
    },
    {
        "label": "04_baoab_cauchy_best",
        "title": "BAOAB Cauchy, $d{=}1$, $\\eta{=}0.5$, $\\gamma{=}0.5$ (best heavy-tail fit)",
        "algorithm": "BAOAB", "distribution": "cauchy",
        "d": 1, "eta": 0.5, "gamma": 0.5, "nu": None,
    },
    {
        "label": "05_mala_cauchy_d50",
        "title": "MALA Cauchy, $d{=}50$, $\\eta{=}1.0$ (heavy tails at moderate dimension)",
        "algorithm": "MALA", "distribution": "cauchy",
        "d": 50, "eta": 1.0, "gamma": None, "nu": None,
    },
]

N_STEPS = 50_000
BURN_IN = 5_000
SEED = 0


def _save_current(label: str, suffix: str):
    fig = plt.gcf()
    fig.suptitle("")  # avoid suptitle stomping subplot titles
    out_pdf = os.path.join(FIG_DIR, f"{label}_{suffix}.pdf")
    out_png = os.path.join(FIG_DIR, f"{label}_{suffix}.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"    wrote {out_pdf}")


def run_case(case: dict):
    print(f"[{case['label']}] running...")
    cfg = ExperimentConfig(
        algorithm=case["algorithm"],
        distribution=case["distribution"],
        d=case["d"],
        eta=case["eta"],
        gamma=case["gamma"],
        nu=case["nu"] if case["nu"] is not None else (5.0 if case["distribution"] == "student_t" else None),
        n_steps=N_STEPS,
        burn_in=BURN_IN,
        seed=SEED,
    )
    sampler = build_sampler(cfg)
    samples_post = sampler.run()
    diagnostics = build_diagnostics(cfg, sampler, samples_post)

    # Compact one-row summary so the reader can cross-reference with the xlsx.
    acc = getattr(sampler, "acceptance_rate", None)
    ess = diagnostics.compute_ess(dim=0)
    print(f"    n_samples_post={diagnostics.n_samples}  ess[0]={ess:.1f}  "
          f"acceptance={'-' if acc is None else f'{acc:.3f}'}")

    # 1) Trace plot (dim 0 only — all dims share the same marginal here)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnostics.plot_trace(dims=[0])
    plt.gcf().suptitle(case["title"], y=1.02, fontsize=11)
    _save_current(case["label"], "trace")

    # 2) Autocorrelation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnostics.plot_autocorrelation(max_lag=200, dims=[0])
    plt.gcf().suptitle(case["title"], y=1.02, fontsize=11)
    _save_current(case["label"], "acf")

    # 3) Goodness of fit (histogram + Q-Q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnostics.plot_tail_exploration(dim=0)
    plt.gcf().suptitle(case["title"], y=1.02, fontsize=11)
    _save_current(case["label"], "gof")


def main():
    print(f"Writing case-study figures to {FIG_DIR}\n")
    for case in CASES:
        run_case(case)
    print("\nDone.")


if __name__ == "__main__":
    main()
