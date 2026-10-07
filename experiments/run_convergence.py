"""
Convergence study: take the best (algorithm, distribution) configurations
from the prior grid search and study how diagnostics evolve with sample size.

Design choice: instead of running 4 independent chains of 50k/100k/500k/1M
(which wastes work), we run a single deterministic chain of 1M iterations per
(algorithm, distribution) and evaluate diagnostics on the prefixes
[0:50k], [0:100k], [0:500k], [0:1M]. Under a fixed seed each prefix is
identical to a standalone shorter run, so the prefix view gives strictly more
information at strictly less cost.

Output: experiments/results/extension/convergence_<study>_<ts>.xlsx (--study thesis | extension)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.run_experiments import (  # noqa: E402
    ExperimentConfig,
    build_diagnostics,
    build_sampler,
    extract_metrics,
)


# Best per-(algorithm, distribution) configs picked from analysis.md
# (lowest ks_stat_mean among runs with div_rate=0 and acceptance in [0.2, 0.95]).
# All best configs landed at d=1.
BEST_CONFIGS: list[dict] = [
    # Selected by lowest median KS over 20 seeds, subject to divergence_rate == 0
    # and (MALA only) acceptance in [0.2, 0.95].
    {"algorithm": "ULA",   "distribution": "gaussian",  "eta": 0.05, "gamma": None, "nu": None},
    {"algorithm": "ULA",   "distribution": "student_t", "eta": 0.05, "gamma": None, "nu": 5.0},
    {"algorithm": "ULA",   "distribution": "cauchy",    "eta": 0.20, "gamma": None, "nu": None},
    # MALA Gaussian: eta=1.0 chosen over 0.5 — KS statistically indistinguishable
    # (medians 0.0056 vs 0.0053, IQRs overlap heavily) but ESS is 2.1x higher.
    {"algorithm": "MALA",  "distribution": "gaussian",  "eta": 1.0,  "gamma": None, "nu": None},
    {"algorithm": "MALA",  "distribution": "student_t", "eta": 1.0,  "gamma": None, "nu": 5.0},
    {"algorithm": "MALA",  "distribution": "cauchy",    "eta": 4.0,  "gamma": None, "nu": None},
    # BAOAB Gaussian: gamma=0.5 chosen over 2.0 — KS statistically indistinguishable
    # (medians 0.0041 vs 0.0038, IQRs overlap) but ESS is 1.39x higher.
    {"algorithm": "BAOAB", "distribution": "gaussian",  "eta": 1.0,  "gamma": 0.5,  "nu": None},
    {"algorithm": "BAOAB", "distribution": "student_t", "eta": 0.6,  "gamma": 1.0,  "nu": 5.0},
    {"algorithm": "BAOAB", "distribution": "cauchy",    "eta": 0.5,  "gamma": 0.5,  "nu": None},
]

# Extension (plan E6): best configuration of each algorithm at d = 20 on the four
# targets where some algorithm fails, at the calibrated difficulty (plan §4.8,
# table T2). Selected by the lowest median b2_grouped over 20 seeds in the
# dimension-scaling runs (experiments/results/extension/scaling_B_best.csv),
# subject to divergence_rate == 0 and (MALA only) acceptance in [0.2, 0.95].
_FUNNEL, _BANANA = {"sigma_v": 2.0}, {"b": 0.03}
_MIXTURE, _DWELL = {"centers": (-6.0, 0.0, 6.0)}, {"beta": 16.0}
EXTENSION_CONFIGS: list[dict] = [
    {"algorithm": "ULA",   "distribution": "funnel",           "eta": 0.01,  "gamma": None, "target_params": _FUNNEL},
    {"algorithm": "MALA",  "distribution": "funnel",           "eta": 0.1,   "gamma": None, "target_params": _FUNNEL},
    {"algorithm": "BAOAB", "distribution": "funnel",           "eta": 0.2,   "gamma": 1.0,  "target_params": _FUNNEL},
    {"algorithm": "ULA",   "distribution": "banana",           "eta": 0.2,   "gamma": None, "target_params": _BANANA},
    {"algorithm": "MALA",  "distribution": "banana",           "eta": 0.5,   "gamma": None, "target_params": _BANANA},
    {"algorithm": "BAOAB", "distribution": "banana",           "eta": 0.5,   "gamma": 0.5,  "target_params": _BANANA},
    {"algorithm": "ULA",   "distribution": "gaussian_mixture", "eta": 0.3,   "gamma": None, "target_params": _MIXTURE},
    {"algorithm": "MALA",  "distribution": "gaussian_mixture", "eta": 0.3,   "gamma": None, "target_params": _MIXTURE},
    {"algorithm": "BAOAB", "distribution": "gaussian_mixture", "eta": 1.0,   "gamma": 0.5,  "target_params": _MIXTURE},
    {"algorithm": "ULA",   "distribution": "double_well",      "eta": 0.01,  "gamma": None, "target_params": _DWELL},
    {"algorithm": "MALA",  "distribution": "double_well",      "eta": 0.005, "gamma": None, "target_params": _DWELL},
    {"algorithm": "BAOAB", "distribution": "double_well",      "eta": 0.1,   "gamma": 2.0,  "target_params": _DWELL},
    # The three tail-weight targets at the same d = 20, so that all eight targets
    # are compared at one dimension. Best configuration at d = 20 by the lowest
    # median KS (results_high_d_only_regenerated), same filters.
    {"algorithm": "ULA",   "distribution": "gaussian",         "eta": 0.1,   "gamma": None, "nu": None},
    {"algorithm": "MALA",  "distribution": "gaussian",         "eta": 0.3,   "gamma": None, "nu": None},
    {"algorithm": "BAOAB", "distribution": "gaussian",         "eta": 1.5,   "gamma": 0.5,  "nu": None},
    {"algorithm": "ULA",   "distribution": "student_t",        "eta": 0.03,  "gamma": None, "nu": 5.0},
    {"algorithm": "MALA",  "distribution": "student_t",        "eta": 0.3,   "gamma": None, "nu": 5.0},
    {"algorithm": "BAOAB", "distribution": "student_t",        "eta": 0.6,   "gamma": 0.5,  "nu": 5.0},
    {"algorithm": "ULA",   "distribution": "cauchy",           "eta": 0.2,   "gamma": None, "nu": None},
    {"algorithm": "MALA",  "distribution": "cauchy",           "eta": 0.3,   "gamma": None, "nu": None},
    {"algorithm": "BAOAB", "distribution": "cauchy",           "eta": 0.5,   "gamma": 0.5,  "nu": None},
]
D_EXTENSION = 20

CHECKPOINTS = [50_000, 100_000, 500_000, 1_000_000]
N_TOTAL = max(CHECKPOINTS)
BURN_IN = 5_000
# Seeds 100-109: the best configurations were selected on seeds 0-19 of the grid runs,
# so the convergence study uses chains that were not used for the selection.
SEEDS = list(range(100, 110))
D = 1


def _run_chain(params: dict, d: int, seed: int):
    """Run one chain of N_TOTAL steps; return its summary and per_dim rows for
    every checkpoint (diagnostics on the prefix of the first `ckpt` steps)."""
    common = dict(algorithm=params["algorithm"], distribution=params["distribution"], d=d,
                  eta=params["eta"], gamma=params["gamma"], nu=params.get("nu"),
                  burn_in=BURN_IN, seed=seed, target_params=dict(params.get("target_params", {})))
    cfg_full = ExperimentConfig(n_steps=N_TOTAL, **common)
    sampler = build_sampler(cfg_full)
    t0 = time.perf_counter()
    samples_post_full = sampler.run()  # shape (N_TOTAL - BURN_IN, d)
    full_runtime = time.perf_counter() - t0
    full_acc = getattr(sampler, "acceptance_rate", None)

    summary_rows, per_dim_rows = [], []
    for ckpt in CHECKPOINTS:
        if ckpt <= BURN_IN:
            continue
        prefix = samples_post_full[:ckpt - BURN_IN]
        # Pretend the chain ran for exactly `ckpt` steps; reuse the same sampler
        # for acceptance_rate (≈ stationary value, stable across prefix lengths).
        diagnostics = build_diagnostics(cfg_full, SimpleNamespace(acceptance_rate=full_acc), prefix)
        cfg_ckpt = ExperimentConfig(n_steps=ckpt, **common)
        s_row, d_rows = extract_metrics(cfg_ckpt, diagnostics, full_runtime * ckpt / N_TOTAL)
        s_row["checkpoint"] = ckpt
        for d_row in d_rows:
            d_row["checkpoint"] = ckpt
        summary_rows.append(s_row)
        per_dim_rows.extend(d_rows)
    return summary_rows, per_dim_rows, full_runtime, cfg_full.label()


def _run_chain_args(args):
    return _run_chain(*args)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["thesis", "extension"], default="thesis",
                        help="thesis: the 9 configurations of the original thesis (d=1, default); "
                             "extension: the 12 configurations of plan E6 (d=20)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel processes (default: 1). Results are identical; "
                             "only the time columns can differ.")
    args = parser.parse_args()
    configs = BEST_CONFIGS if args.study == "thesis" else EXTENSION_CONFIGS
    d = D if args.study == "thesis" else D_EXTENSION

    summary_rows: list[dict] = []
    per_dim_rows: list[dict] = []

    jobs = [(params, d, seed) for params in configs for seed in SEEDS]
    total_chains = len(jobs)
    print(f"Convergence study ({args.study}): {len(configs)} configs x {len(SEEDS)} seeds "
          f"= {total_chains} chains, "
          f"N_total={N_TOTAL}, burn_in={BURN_IN}, checkpoints={CHECKPOINTS}, d={d}, "
          f"workers={args.workers}\n")

    t_start = time.perf_counter()
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(_run_chain_args, jobs, chunksize=1)
    else:
        executor = None
        results = map(_run_chain_args, jobs)
    for chain_idx, (s_rows, d_rows, runtime, label) in enumerate(results, start=1):
        summary_rows.extend(s_rows)
        per_dim_rows.extend(d_rows)
        last = s_rows[-1]
        print(f"[{chain_idx}/{total_chains}] {label}  chain runtime: {runtime:.2f}s  "
              f"ks={last['ks_stat_mean']:.5f}  b2_grouped={last.get('b2_grouped')}", flush=True)
    if executor is not None:
        executor.shutdown()

    total_dt = time.perf_counter() - t_start
    print(f"Finished {total_chains} chains in {total_dt:.1f}s")

    summary_df = pd.DataFrame(summary_rows)
    per_dim_df = pd.DataFrame(per_dim_rows)

    # Put `checkpoint` near the front for readability
    cols_summary = ["algorithm", "distribution", "checkpoint", "d", "eta", "gamma", "nu",
                    "n_steps", "burn_in", "n_samples_post", "seed"]
    if "target_params" in summary_df.columns:
        cols_summary.insert(7, "target_params")
    other_summary = [c for c in summary_df.columns if c not in cols_summary]
    summary_df = summary_df[cols_summary + other_summary]

    cols_per_dim = ["algorithm", "distribution", "checkpoint", "d", "eta", "gamma", "nu",
                    "seed", "dim"]
    other_per_dim = [c for c in per_dim_df.columns if c not in cols_per_dim]
    per_dim_df = per_dim_df[cols_per_dim + other_per_dim]

    config_records = [
        {"key": "study", "value": f"convergence ({args.study})"},
        {"key": "configs", "value": ", ".join(f"{c['algorithm']}/{c['distribution']}" for c in configs)},
        {"key": "checkpoints", "value": ", ".join(map(str, CHECKPOINTS))},
        {"key": "N_total", "value": N_TOTAL},
        {"key": "burn_in", "value": BURN_IN},
        {"key": "seeds", "value": ", ".join(map(str, SEEDS))},
        {"key": "d", "value": d},
        {"key": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    config_df = pd.DataFrame(config_records)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(_HERE, "results", "extension", f"convergence_{args.study}_{ts}.xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="convergence_summary", index=False)
        per_dim_df.to_excel(writer, sheet_name="convergence_per_dim", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
