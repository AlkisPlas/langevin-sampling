"""
Extract the key numbers used in analysis.md / convergence_analysis.md from
the multi-seed aggregated xlsx files. Prints out a structured report that
can be pasted into the analysis files.

Run from repo root:
    python3 experiments/extract_analysis_numbers.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results", "original_thesis")


def _latest(pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern} in {RESULTS_DIR}")
    return matches[-1]


def _fmt(val, fmt="{:.5f}"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NaN"
    return fmt.format(val)


def _spread(row, key):
    c = row.get(f"{key}_central", np.nan)
    lo = row.get(f"{key}_low", np.nan)
    hi = row.get(f"{key}_high", np.nan)
    return f"{_fmt(c)} ({_fmt(lo)}, {_fmt(hi)})"


def best_configs(grid_agg):
    valid = grid_agg[grid_agg["divergence_rate_central"] == 0].copy()
    mala_mask = ((valid["algorithm"] != "MALA") |
                 ((valid["acceptance_rate_central"] >= 0.2) &
                  (valid["acceptance_rate_central"] <= 0.95)))
    valid = valid[mala_mask]

    print("\n=== BEST CONFIGURATIONS (min central KS subject to filters) ===\n")
    print(f"{'algo':6} {'dist':12} {'d':>3} {'eta':>6} {'gamma':>6} "
          f"{'KS (med, IQR)':>30} {'ess_mean (mean ± SE)':>32} "
          f"{'q_mae_avg (med, IQR)':>30} {'accept':>15}")
    for algo in ["ULA", "MALA", "BAOAB"]:
        for dist in ["gaussian", "student_t", "cauchy"]:
            sub = valid[(valid["algorithm"] == algo) &
                        (valid["distribution"] == dist)]
            if sub.empty:
                continue
            row = sub.loc[sub["ks_stat_mean_central"].idxmin()]
            gamma = row["gamma"]
            gamma_s = f"{gamma:.1f}" if pd.notna(gamma) else "-"
            print(f"{algo:6} {dist:12} {int(row['d']):>3} "
                  f"{row['eta']:>6.3f} {gamma_s:>6} "
                  f"{_spread(row, 'ks_stat_mean'):>30} "
                  f"{_spread(row, 'ess_mean'):>32} "
                  f"{_spread(row, 'quantile_mae_avg'):>30} "
                  f"{_spread(row, 'acceptance_rate'):>15}")


def convergence_table(conv_agg):
    print("\n=== CONVERGENCE TABLE: ks_stat_mean per (algo, dist, checkpoint) ===\n")
    print(f"{'algo':6} {'dist':12} {'ckpt':>9} {'KS (median, IQR)':>30}")
    for algo in ["ULA", "MALA", "BAOAB"]:
        for dist in ["gaussian", "student_t", "cauchy"]:
            sub = conv_agg[(conv_agg["algorithm"] == algo) &
                           (conv_agg["distribution"] == dist)]
            for _, row in sub.sort_values("checkpoint").iterrows():
                print(f"{algo:6} {dist:12} {int(row['checkpoint']):>9} "
                      f"{_spread(row, 'ks_stat_mean'):>30}")
        print()


def dimension_scaling(grid_agg):
    print("\n=== DIMENSION SCALING: mean(KS_central) per (algo, dist, d) ===\n")
    print(f"{'algo':6} {'dist':12} {'d':>3} {'mean KS':>10} {'IQR':>22}")
    for algo in ["ULA", "MALA", "BAOAB"]:
        for dist in ["gaussian", "student_t", "cauchy"]:
            sub = grid_agg[(grid_agg["algorithm"] == algo) &
                           (grid_agg["distribution"] == dist) &
                           (grid_agg["divergence_rate_central"] == 0)]
            for d in sorted(sub["d"].unique()):
                ds = sub[sub["d"] == d]["ks_stat_mean_central"]
                m = ds.mean()
                q25, q75 = np.percentile(ds, [25, 75])
                print(f"{algo:6} {dist:12} {int(d):>3} {m:>10.5f} "
                      f"({q25:.5f}, {q75:.5f})")
            print()


def gamma_sensitivity(grid_agg):
    print("\n=== BAOAB GAMMA SENSITIVITY: mean(KS,ess) per gamma ===\n")
    sub = grid_agg[grid_agg["algorithm"] == "BAOAB"]
    print(f"{'dist':12} {'gamma':>6} {'mean KS':>10} {'mean ESS':>10}")
    for dist in ["gaussian", "student_t", "cauchy"]:
        for gamma in sorted(sub[sub["distribution"] == dist]["gamma"].unique()):
            ss = sub[(sub["distribution"] == dist) & (sub["gamma"] == gamma)]
            print(f"{dist:12} {gamma:>6.1f} "
                  f"{ss['ks_stat_mean_central'].mean():>10.5f} "
                  f"{ss['ess_mean_central'].mean():>10.0f}")
        print()


def mala_acceptance(grid_agg):
    print("\n=== MALA ACCEPTANCE BY ETA AND D ===\n")
    sub = grid_agg[grid_agg["algorithm"] == "MALA"]
    for dist in ["gaussian", "student_t", "cauchy"]:
        print(f"--- {dist} ---")
        ss = sub[sub["distribution"] == dist]
        for d in sorted(ss["d"].unique()):
            row = ss[ss["d"] == d].sort_values("eta")
            etas_accs = [(r["eta"], r["acceptance_rate_central"])
                         for _, r in row.iterrows()]
            print(f"d={int(d)}: " + ", ".join(
                f"eta={e}->{_fmt(a, '{:.3f}')}" for e, a in etas_accs))
        print()


def main():
    grid_path = _latest("results_default_*_agg.xlsx")
    conv_path = _latest("convergence_*_agg.xlsx")
    print(f"Grid : {grid_path}")
    print(f"Conv : {conv_path}")
    grid_agg = pd.read_excel(grid_path, sheet_name="aggregated")
    conv_agg = pd.read_excel(conv_path, sheet_name="aggregated")
    print(f"\nGrid: {len(grid_agg)} configurations from "
          f"{grid_agg['n_seeds'].iloc[0]} seeds each.")
    print(f"Conv: {len(conv_agg)} (config, checkpoint) entries from "
          f"{conv_agg['n_seeds'].iloc[0]} seeds each.")

    best_configs(grid_agg)
    convergence_table(conv_agg)
    dimension_scaling(grid_agg)
    gamma_sensitivity(grid_agg)
    mala_acceptance(grid_agg)


if __name__ == "__main__":
    main()
