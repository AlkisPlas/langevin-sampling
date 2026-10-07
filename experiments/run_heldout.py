"""
Held-out evaluation: run the best configuration of each (target, algorithm, d) on
20 new seeds (100-119), which were not used to select it.

The best configurations are the ones in experiments/results/
scaling_A_best.csv (median KS) and scaling_B_best.csv (median b2_grouped), which
were selected on seeds 0-19. The difficulty parameter of each target is the one
of the `scaling_B1` preset. All other settings (n_steps, burn_in, initial state)
are the same as in the grid runs.

Output: experiments/results/results_heldout.xlsx (and _agg.xlsx with
experiments/aggregate_seeds.py).

Run from the repo root:
    python3 experiments/run_heldout.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.run_experiments import (  # noqa: E402
    PER_DIM_MAX_D,
    PRESETS,
    ExperimentConfig,
    _run_safe,
    write_excel,
)

RESULTS_DIR = os.path.join(_HERE, "results")
SEEDS = list(range(100, 120))
GRID = PRESETS["scaling_B1"]
NU = {"student_t": 5.0}


def build_heldout_configs() -> list[ExperimentConfig]:
    best = pd.concat([pd.read_csv(os.path.join(RESULTS_DIR, "scaling_A_best.csv")),
                      pd.read_csv(os.path.join(RESULTS_DIR, "scaling_B_best.csv"))], ignore_index=True)
    configs = []
    for _, r in best.iterrows():
        params = GRID["target_params"].get(r["target"], [{}])[0]
        for seed in SEEDS:
            configs.append(ExperimentConfig(
                algorithm=r["alg"], distribution=r["target"], d=int(r["d"]), eta=float(r["eta"]),
                gamma=None if pd.isna(r["gamma"]) else float(r["gamma"]), nu=NU.get(r["target"]),
                n_steps=GRID["n_steps"], burn_in=GRID["burn_in"], seed=seed, target_params=dict(params),
            ))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "results_heldout.xlsx"))
    args = parser.parse_args()

    configs = build_heldout_configs()
    print(f"Held-out runs: {len(configs)}  | workers: {args.workers}\nOutput: {args.output}\n")

    summary_rows, per_dim_rows = [], []
    t_start = time.perf_counter()
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
    if executor is not None:
        executor.shutdown()

    meta = {"algorithms": ["ULA", "MALA", "BAOAB"],
            "distributions": sorted({c.distribution for c in configs}),
            "d_values": sorted({c.d for c in configs}), "gamma_values": GRID["gamma_values"],
            "nu": 5.0, "seeds": SEEDS}
    write_excel(args.output, summary_rows, per_dim_rows, "heldout", meta,
                GRID["n_steps"], GRID["burn_in"])
    print(f"\nWrote: {args.output}  ({time.perf_counter() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
