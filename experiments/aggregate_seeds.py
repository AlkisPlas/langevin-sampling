"""
Aggregate per-seed run rows into per-configuration summaries.

Reads the `summary` (or `convergence_summary`) sheet of an experiments xlsx
and produces a new sheet `aggregated` with one row per configuration. For
each numeric metric column, the aggregated sheet stores four columns:

    <metric>_central   the central value (mean or median per AGG_RULES)
    <metric>_low       the lower spread bound (depends on the rule)
    <metric>_high      the upper spread bound
    <metric>_n_valid   number of seeds with a finite value

Run from the repo root:
    python3 experiments/aggregate_seeds.py path/to/results.xlsx
    python3 experiments/aggregate_seeds.py path/to/convergence.xlsx --convergence
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

# Columns that identify a configuration (everything except seed). For the
# convergence xlsx the checkpoint column is also a config-level key.
GRID_GROUP = ["algorithm", "distribution", "d", "eta", "gamma", "nu",
              "n_steps", "burn_in"]
CONV_GROUP = ["algorithm", "distribution", "d", "eta", "gamma", "nu",
              "checkpoint"]

# Aggregation rules: each entry maps a metric column to
#   (central, low, high)
# where central is "mean" or "median" and low/high are one of:
#   "q05", "q25", "q75", "q95", "se_lo", "se_hi", "max"
# When a metric is not in AGG_RULES it defaults to ("mean", "se_lo", "se_hi").
AGG_RULES: dict[str, tuple[str, str, str]] = {
    # KS and quantile errors: median + IQR (skew-resistant for heavy-tail metrics)
    "ks_stat_mean":         ("median", "q25", "q75"),
    "quantile_mae_avg":     ("median", "q25", "q75"),
    "quantile_mae_q0.025":  ("median", "q25", "q75"),
    "quantile_mae_q0.5":    ("median", "q25", "q75"),
    "quantile_mae_q0.975":  ("median", "q25", "q75"),
    # Tail coverage: median + 5-95 (bounded ratio, count-based)
    "tail_cov_ratio_q0.01": ("median", "q05", "q95"),
    "tail_cov_ratio_q0.05": ("median", "q05", "q95"),
    "tail_cov_ratio_q0.95": ("median", "q05", "q95"),
    "tail_cov_ratio_q0.99": ("median", "q05", "q95"),
    # ESS family: mean + SE
    "ess_mean":             ("mean",   "se_lo", "se_hi"),
    "ess_min":              ("mean",   "se_lo", "se_hi"),
    "ess_max":              ("mean",   "se_lo", "se_hi"),
    "ess_ratio_mean":       ("mean",   "se_lo", "se_hi"),
    "ess_per_sec":          ("mean",   "se_lo", "se_hi"),
    # IAT: median + IQR
    "iat_mean":             ("median", "q25", "q75"),
    # Acceptance: mean + SE (bounded in [0,1])
    "acceptance_rate":      ("mean",   "se_lo", "se_hi"),
    # Divergence: max (worst-case)
    "divergence_rate":      ("max",    "max",  "max"),
    # Moment biases: mean + SE
    "mean_abs_bias":        ("mean",   "se_lo", "se_hi"),
    "var_abs_bias":         ("mean",   "se_lo", "se_hi"),
    "median_abs_bias":      ("mean",   "se_lo", "se_hi"),
    # Runtime: median + IQR
    "runtime_seconds":      ("median", "q25", "q75"),
    # ACF lags: mean + SE
    "acf_lag_1":            ("mean",   "se_lo", "se_hi"),
    "acf_lag_10":           ("mean",   "se_lo", "se_hi"),
    "acf_lag_50":           ("mean",   "se_lo", "se_hi"),
    "acf_lag_100":          ("mean",   "se_lo", "se_hi"),
    # Empirical moments: mean + SE
    "empirical_mean_avg":   ("mean",   "se_lo", "se_hi"),
    "empirical_var_avg":    ("mean",   "se_lo", "se_hi"),
    "empirical_std_avg":    ("mean",   "se_lo", "se_hi"),
    "empirical_median_avg": ("mean",   "se_lo", "se_hi"),
    # Theoretical values are constants - propagate the first finite value
    "theoretical_mean_avg":   ("mean", "mean", "mean"),
    "theoretical_var_avg":    ("mean", "mean", "mean"),
    "theoretical_median_avg": ("mean", "mean", "mean"),
    # Sample size: mean (should be deterministic but loop in case of failures)
    "n_samples_post":       ("mean", "mean", "mean"),
}

DEFAULT_RULE = ("mean", "se_lo", "se_hi")


def _agg_value(vals: np.ndarray, rule: str) -> float:
    """Return one summary statistic of `vals` according to `rule`."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan")
    if rule == "mean":
        return float(np.mean(vals))
    if rule == "median":
        return float(np.median(vals))
    if rule == "max":
        return float(np.max(vals))
    if rule == "min":
        return float(np.min(vals))
    if rule.startswith("q"):
        p = int(rule[1:])
        return float(np.percentile(vals, p))
    if rule == "se_lo":
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return mean - se
    if rule == "se_hi":
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return mean + se
    raise ValueError(f"Unknown aggregation rule: {rule}")


def _is_metric_column(col: str, group_keys: list[str]) -> bool:
    if col in group_keys:
        return False
    if col in ("seed", "error"):
        return False
    return True


def aggregate_summary(summary_df: pd.DataFrame,
                      group_keys: list[str]) -> pd.DataFrame:
    """Aggregate `summary_df` over seeds using AGG_RULES.

    Returns a wide DataFrame with one row per configuration and four columns
    per metric: <m>_central, <m>_low, <m>_high, <m>_n_valid.
    """
    metric_cols = [c for c in summary_df.columns if _is_metric_column(c, group_keys)]
    # Keep only numeric metrics
    metric_cols = [c for c in metric_cols
                   if pd.api.types.is_numeric_dtype(summary_df[c])]

    rows = []
    grouped = summary_df.groupby(group_keys, dropna=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = dict(zip(group_keys, keys))
        row["n_seeds"] = len(sub)
        for col in metric_cols:
            central, low, high = AGG_RULES.get(col, DEFAULT_RULE)
            vals = sub[col].to_numpy()
            row[f"{col}_central"] = _agg_value(vals, central)
            row[f"{col}_low"]     = _agg_value(vals, low)
            row[f"{col}_high"]    = _agg_value(vals, high)
            row[f"{col}_n_valid"] = int(np.sum(~np.isnan(vals.astype(float))))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_dim(per_dim_df: pd.DataFrame,
                      group_keys: list[str]) -> pd.DataFrame:
    """Aggregate the per_dim sheet similarly, grouping by configuration + dim.

    Some columns in the summary group key (n_steps, burn_in, n_samples_post)
    are not present on the per_dim sheet, so we restrict to those that exist.
    """
    group_keys_dim = [k for k in group_keys if k in per_dim_df.columns] + ["dim"]
    return aggregate_summary(per_dim_df, group_keys_dim)


def process(xlsx_path: str, kind: str, output: str | None) -> str:
    if kind == "grid":
        runs_sheet = "summary"
        per_dim_sheet = "per_dim"
        group_keys = GRID_GROUP
    else:
        runs_sheet = "convergence_summary"
        per_dim_sheet = "convergence_per_dim"
        group_keys = CONV_GROUP

    xls = pd.ExcelFile(xlsx_path)
    if runs_sheet not in xls.sheet_names:
        raise SystemExit(
            f"Expected sheet '{runs_sheet}' not found in {xlsx_path}; "
            f"available sheets: {xls.sheet_names}"
        )

    summary_df = pd.read_excel(xlsx_path, sheet_name=runs_sheet)
    per_dim_df = pd.read_excel(xlsx_path, sheet_name=per_dim_sheet) \
                 if per_dim_sheet in xls.sheet_names else None

    # Sanity check group keys
    missing = [k for k in group_keys if k not in summary_df.columns]
    if missing:
        raise SystemExit(f"Group keys missing from summary: {missing}")

    print(f"Aggregating {len(summary_df)} runs over {summary_df['seed'].nunique()} seeds...")
    agg_summary = aggregate_summary(summary_df, group_keys)
    print(f"  -> {len(agg_summary)} configurations")

    agg_per_dim = None
    if per_dim_df is not None:
        agg_per_dim = aggregate_per_dim(per_dim_df, group_keys)
        print(f"  per_dim: {len(per_dim_df)} rows -> {len(agg_per_dim)} aggregated rows")

    # Copy the source file and append the aggregated sheets.
    if output is None:
        base, ext = os.path.splitext(xlsx_path)
        output = f"{base}_agg{ext}"
    shutil.copy(xlsx_path, output)

    with pd.ExcelWriter(output, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        agg_summary.to_excel(writer, sheet_name="aggregated", index=False)
        if agg_per_dim is not None:
            agg_per_dim.to_excel(writer, sheet_name="aggregated_per_dim", index=False)
        # Add a meta row
        meta = pd.DataFrame([{
            "key": "aggregated_at",
            "value": datetime.now().isoformat(timespec="seconds"),
        }, {
            "key": "n_seeds_per_config",
            "value": int(summary_df["seed"].nunique()),
        }, {
            "key": "n_configurations",
            "value": int(len(agg_summary)),
        }, {
            "key": "kind",
            "value": kind,
        }])
        meta.to_excel(writer, sheet_name="aggregation_meta", index=False)

    print(f"\nWrote: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx", help="Path to the input xlsx")
    parser.add_argument("--convergence", action="store_true",
                        help="Treat input as a convergence-study xlsx")
    parser.add_argument("--output", default=None,
                        help="Output xlsx path (default: <input>_agg.xlsx)")
    args = parser.parse_args()
    kind = "convergence" if args.convergence else "grid"
    process(args.xlsx, kind, args.output)


if __name__ == "__main__":
    main()
