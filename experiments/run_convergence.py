"""
Convergence study: take the best (algorithm, distribution) configurations
from the prior grid search and study how diagnostics evolve with sample size.

Design choice: instead of running 4 independent chains of 50k/100k/500k/1M
(which wastes work), we run a single deterministic chain of 1M iterations per
(algorithm, distribution) and evaluate diagnostics on the prefixes
[0:50k], [0:100k], [0:500k], [0:1M]. Under a fixed seed each prefix is
identical to a standalone shorter run, so the prefix view gives strictly more
information at strictly less cost.

Output: experiments/results/extension/convergence_<ts>.xlsx
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

CHECKPOINTS = [50_000, 100_000, 500_000, 1_000_000]
N_TOTAL = max(CHECKPOINTS)
BURN_IN = 5_000
SEEDS = list(range(10))
D = 1


def main():
    summary_rows: list[dict] = []
    per_dim_rows: list[dict] = []

    total_chains = len(BEST_CONFIGS) * len(SEEDS)
    print(f"Convergence study: {len(BEST_CONFIGS)} configs x {len(SEEDS)} seeds "
          f"= {total_chains} chains, "
          f"N_total={N_TOTAL}, burn_in={BURN_IN}, checkpoints={CHECKPOINTS}, d={D}\n")

    t_start = time.perf_counter()
    chain_idx = 0
    for params in BEST_CONFIGS:
        for seed in SEEDS:
            chain_idx += 1
            cfg_full = ExperimentConfig(
                algorithm=params["algorithm"],
                distribution=params["distribution"],
                d=D,
                eta=params["eta"],
                gamma=params["gamma"],
                nu=params["nu"],
                n_steps=N_TOTAL,
                burn_in=BURN_IN,
                seed=seed,
            )
            sampler = build_sampler(cfg_full)
            t0 = time.perf_counter()
            samples_post_full = sampler.run()  # shape (N_TOTAL - BURN_IN, D)
            full_runtime = time.perf_counter() - t0
            full_acc = getattr(sampler, "acceptance_rate", None)

            print(f"[{chain_idx}/{total_chains}] {cfg_full.label()}  "
                  f"chain runtime: {full_runtime:.2f}s")

            for ckpt in CHECKPOINTS:
                if ckpt <= BURN_IN:
                    continue
                n_post = ckpt - BURN_IN
                prefix = samples_post_full[:n_post]

                # Pretend the chain ran for exactly `ckpt` steps; reuse the same sampler
                # for acceptance_rate (≈ stationary value, stable across prefix lengths).
                faux_sampler = SimpleNamespace(acceptance_rate=full_acc)
                diagnostics = build_diagnostics(cfg_full, faux_sampler, prefix)

                ckpt_runtime = full_runtime * ckpt / N_TOTAL

                cfg_ckpt = ExperimentConfig(
                    algorithm=params["algorithm"],
                    distribution=params["distribution"],
                    d=D,
                    eta=params["eta"],
                    gamma=params["gamma"],
                    nu=params["nu"],
                    n_steps=ckpt,
                    burn_in=BURN_IN,
                    seed=seed,
                )
                s_row, d_rows = extract_metrics(cfg_ckpt, diagnostics, ckpt_runtime)
                s_row["checkpoint"] = ckpt
                for d_row in d_rows:
                    d_row["checkpoint"] = ckpt
                summary_rows.append(s_row)
                per_dim_rows.extend(d_rows)
            print(f"    ckpt={ckpt:>8}  ks={s_row['ks_stat_mean']:.5f}  "
                  f"q_mae_avg={s_row['quantile_mae_avg']:.4f}  "
                  f"ess_mean={s_row['ess_mean']:.0f}  "
                  f"mean_bias={s_row.get('mean_abs_bias')}")
        print()

    total_dt = time.perf_counter() - t_start
    print(f"Finished {total_chains} chains in {total_dt:.1f}s")

    summary_df = pd.DataFrame(summary_rows)
    per_dim_df = pd.DataFrame(per_dim_rows)

    # Put `checkpoint` near the front for readability
    cols_summary = ["algorithm", "distribution", "checkpoint", "d", "eta", "gamma", "nu",
                    "n_steps", "burn_in", "n_samples_post", "seed"]
    other_summary = [c for c in summary_df.columns if c not in cols_summary]
    summary_df = summary_df[cols_summary + other_summary]

    cols_per_dim = ["algorithm", "distribution", "checkpoint", "d", "eta", "gamma", "nu",
                    "seed", "dim"]
    other_per_dim = [c for c in per_dim_df.columns if c not in cols_per_dim]
    per_dim_df = per_dim_df[cols_per_dim + other_per_dim]

    config_records = [
        {"key": "study", "value": "convergence"},
        {"key": "configs", "value": ", ".join(f"{c['algorithm']}/{c['distribution']}" for c in BEST_CONFIGS)},
        {"key": "checkpoints", "value": ", ".join(map(str, CHECKPOINTS))},
        {"key": "N_total", "value": N_TOTAL},
        {"key": "burn_in", "value": BURN_IN},
        {"key": "seeds", "value": ", ".join(map(str, SEEDS))},
        {"key": "d", "value": D},
        {"key": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    config_df = pd.DataFrame(config_records)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(_HERE, "results", "extension", f"convergence_{ts}.xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="convergence_summary", index=False)
        per_dim_df.to_excel(writer, sheet_name="convergence_per_dim", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
