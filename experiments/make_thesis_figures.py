"""
Generate figures for the thesis chapter from the multi-seed aggregated xlsx
files produced by experiments/run_experiments.py + aggregate_seeds.py and
experiments/run_convergence.py + aggregate_seeds.py --convergence.

The figures use:
 - the central value (median or mean per AGG_RULES) for the main lines/bars,
 - the low/high columns to draw shaded bands (line plots) or yerr bars (bars).

Saves PDF + PNG figures into experiments/figures/extension/.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results", "original_thesis")
FIG_DIR = os.path.join(_HERE, "figures", "extension")
os.makedirs(FIG_DIR, exist_ok=True)


def _latest(pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern} in {RESULTS_DIR}")
    return matches[-1]


DIST_ORDER = ["gaussian", "student_t", "cauchy"]
DIST_LABEL = {
    "gaussian": r"Gaussian",
    "student_t": r"Student-$t\;(\nu=5)$",
    "cauchy": r"Cauchy",
}
ALGO_ORDER = ["ULA", "MALA", "BAOAB"]
ALGO_COLOR = {"ULA": "#d62728", "MALA": "#1f77b4", "BAOAB": "#2ca02c"}
ALGO_MARKER = {"ULA": "o", "MALA": "s", "BAOAB": "^"}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})


def save(fig, name: str):
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def _algo_band(ax, df, x_col, y_metric, log_x=True, log_y=True, alpha=0.18):
    """Plot per-algorithm central line with shaded low/high band."""
    yc = f"{y_metric}_central"
    yl = f"{y_metric}_low"
    yh = f"{y_metric}_high"
    for algo in ALGO_ORDER:
        r = df[df["algorithm"] == algo].sort_values(x_col)
        if r.empty:
            continue
        x = r[x_col].to_numpy()
        c = r[yc].to_numpy()
        lo = r[yl].to_numpy()
        hi = r[yh].to_numpy()
        ax.fill_between(x, lo, hi, color=ALGO_COLOR[algo], alpha=alpha,
                        linewidth=0)
        ax.plot(x, c, marker=ALGO_MARKER[algo], color=ALGO_COLOR[algo],
                label=algo)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


# ---------------------------------------------------------------------------
# 1) KS convergence (3 panels, one per distribution)
# ---------------------------------------------------------------------------
def fig_ks_convergence(conv_agg: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, dist in zip(axes, DIST_ORDER):
        sub = conv_agg[conv_agg["distribution"] == dist]
        _algo_band(ax, sub, "checkpoint", "ks_stat_mean",
                   log_x=True, log_y=True)
        ns = sorted(sub["checkpoint"].unique())
        ks_anchor = sub[(sub["algorithm"] == "MALA") &
                        (sub["checkpoint"] == ns[0])]["ks_stat_mean_central"].values
        if ks_anchor.size:
            ref = ks_anchor[0] * np.sqrt(ns[0] / np.array(ns))
            ax.plot(ns, ref, "--", color="gray", alpha=0.7, label=r"$\propto n^{-1/2}$")
        ax.set_xlabel(r"Iterations $n$")
        ax.set_title(DIST_LABEL[dist])
    axes[0].set_ylabel("Kolmogorov--Smirnov statistic")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    save(fig, "ks_convergence")


# ---------------------------------------------------------------------------
# 2) ESS scaling
# ---------------------------------------------------------------------------
def fig_ess_scaling(conv_agg: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, dist in zip(axes, DIST_ORDER):
        sub = conv_agg[conv_agg["distribution"] == dist]
        _algo_band(ax, sub, "checkpoint", "ess_mean",
                   log_x=True, log_y=True)
        ns = np.array(sorted(sub["checkpoint"].unique()), dtype=float)
        anchor = sub[(sub["algorithm"] == "MALA") &
                     (sub["checkpoint"] == ns[0])]["ess_mean_central"].values
        if anchor.size:
            ax.plot(ns, anchor[0] * ns / ns[0], "--", color="gray", alpha=0.7,
                    label=r"$\propto n$")
        ax.set_xlabel(r"Iterations $n$")
        ax.set_title(DIST_LABEL[dist])
    axes[0].set_ylabel(r"Mean ESS")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    save(fig, "ess_scaling")


# ---------------------------------------------------------------------------
# 3) Quantile-MAE convergence
# ---------------------------------------------------------------------------
def fig_quantile_mae_convergence(conv_agg: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, dist in zip(axes, DIST_ORDER):
        sub = conv_agg[conv_agg["distribution"] == dist]
        _algo_band(ax, sub, "checkpoint", "quantile_mae_avg",
                   log_x=True, log_y=True)
        ax.set_xlabel(r"Iterations $n$")
        ax.set_title(DIST_LABEL[dist])
    axes[0].set_ylabel(r"Quantile MAE (averaged)")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    save(fig, "quantile_mae_convergence")


# ---------------------------------------------------------------------------
# 4) Cauchy tail coverage convergence (2 panels with shaded bands)
# ---------------------------------------------------------------------------
def fig_cauchy_tail_coverage(conv_agg: pd.DataFrame):
    sub = conv_agg[conv_agg["distribution"] == "cauchy"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    for ax, q, title in zip(
            axes,
            ["tail_cov_ratio_q0.01", "tail_cov_ratio_q0.99"],
            [r"Lower tail ($q=0.01$)", r"Upper tail ($q=0.99$)"]):
        for algo in ALGO_ORDER:
            r = sub[sub["algorithm"] == algo].sort_values("checkpoint")
            if r.empty:
                continue
            x = r["checkpoint"].to_numpy()
            c = r[f"{q}_central"].to_numpy()
            lo = r[f"{q}_low"].to_numpy()
            hi = r[f"{q}_high"].to_numpy()
            ax.fill_between(x, lo, hi, color=ALGO_COLOR[algo], alpha=0.18,
                            linewidth=0)
            ax.plot(x, c, marker=ALGO_MARKER[algo],
                    color=ALGO_COLOR[algo], label=algo)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6,
                   label="ideal")
        ax.set_xscale("log")
        ax.set_xlabel(r"Iterations $n$")
        ax.set_title(title)
    axes[0].set_ylabel("Tail coverage ratio (emp./expected)")
    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "cauchy_tail_coverage")


# ---------------------------------------------------------------------------
# 5) Best-configuration KS bar chart with error bars
# ---------------------------------------------------------------------------
def _select_best_per_algo_dist(grid_agg: pd.DataFrame) -> pd.DataFrame:
    """Pick the configuration with lowest central KS, subject to filters."""
    valid = grid_agg[(grid_agg["divergence_rate_central"] == 0)].copy()
    # MALA acceptance must be in [0.2, 0.95]
    mala_mask = ((valid["algorithm"] != "MALA") |
                 ((valid["acceptance_rate_central"] >= 0.2) &
                  (valid["acceptance_rate_central"] <= 0.95)))
    valid = valid[mala_mask]

    rows = []
    for algo in ALGO_ORDER:
        for dist in DIST_ORDER:
            sub = valid[(valid["algorithm"] == algo) & (valid["distribution"] == dist)]
            if sub.empty:
                continue
            rows.append(sub.loc[sub["ks_stat_mean_central"].idxmin()])
    return pd.DataFrame(rows)


def fig_best_config_bars(grid_agg: pd.DataFrame):
    best = _select_best_per_algo_dist(grid_agg)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(DIST_ORDER))
    width = 0.25
    for i, algo in enumerate(ALGO_ORDER):
        vals = []
        errs_lo = []
        errs_hi = []
        for d in DIST_ORDER:
            r = best[(best["algorithm"] == algo) & (best["distribution"] == d)]
            if r.empty:
                vals.append(np.nan); errs_lo.append(0); errs_hi.append(0); continue
            c = float(r["ks_stat_mean_central"].values[0])
            lo = float(r["ks_stat_mean_low"].values[0])
            hi = float(r["ks_stat_mean_high"].values[0])
            vals.append(c)
            errs_lo.append(max(c - lo, 0))
            errs_hi.append(max(hi - c, 0))
        bars = ax.bar(x + (i - 1) * width, vals, width, color=ALGO_COLOR[algo],
                      label=algo, edgecolor="black", linewidth=0.5,
                      yerr=[errs_lo, errs_hi], capsize=3,
                      error_kw=dict(elinewidth=0.8, ecolor="black"))
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.4f}",
                        ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([DIST_LABEL[d] for d in DIST_ORDER])
    ax.set_ylabel("Best KS statistic (median over seeds)")
    ax.set_yscale("log")
    ax.set_title("Best configuration per algorithm / distribution (IQR error bars)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "best_config_bars")


# ---------------------------------------------------------------------------
# 6) Dimension scaling of KS (mean of centrals over eta, gamma)
# ---------------------------------------------------------------------------
def fig_dimension_scaling(grid_agg: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    valid = grid_agg[grid_agg["divergence_rate_central"] == 0]
    for ax, dist in zip(axes, DIST_ORDER):
        sub = valid[valid["distribution"] == dist]
        for algo in ALGO_ORDER:
            ss = sub[sub["algorithm"] == algo]
            if ss.empty:
                continue
            # For each d, take mean+IQR over hyperparameter configurations
            stats = ss.groupby("d")["ks_stat_mean_central"].agg(
                central="mean",
                lo=lambda x: np.percentile(x, 25),
                hi=lambda x: np.percentile(x, 75),
            ).reset_index()
            ax.fill_between(stats["d"], stats["lo"], stats["hi"],
                            color=ALGO_COLOR[algo], alpha=0.18, linewidth=0)
            ax.plot(stats["d"], stats["central"],
                    marker=ALGO_MARKER[algo], color=ALGO_COLOR[algo], label=algo)
        ax.set_xlabel(r"Dimension $d$")
        ax.set_xscale("log")
        ax.set_title(DIST_LABEL[dist])
        ax.set_yscale("log")
    axes[0].set_ylabel(r"KS (mean over $\eta, \gamma$; IQR band)")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "dimension_scaling_ks")


# ---------------------------------------------------------------------------
# 7) MALA acceptance vs eta and dimension (per-d lines with seed band)
# ---------------------------------------------------------------------------
def fig_mala_acceptance(grid_agg: pd.DataFrame):
    sub = grid_agg[grid_agg["algorithm"] == "MALA"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, dist in zip(axes, DIST_ORDER):
        ss = sub[sub["distribution"] == dist]
        ds = sorted(ss["d"].unique())
        for d in ds:
            row = ss[ss["d"] == d].sort_values("eta")
            x = row["eta"].to_numpy()
            c = row["acceptance_rate_central"].to_numpy()
            lo = row["acceptance_rate_low"].to_numpy()
            hi = row["acceptance_rate_high"].to_numpy()
            line, = ax.plot(x, c, marker="o", label=fr"$d={d}$")
            ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.18,
                            linewidth=0)
        ax.set_xlabel(r"Step size $\eta$")
        ax.set_xscale("log")
        ax.set_title(DIST_LABEL[dist])
        ax.set_ylim(-0.02, 1.02)
        # Roberts-Rosenthal (1998) MALA asymptotic optimum is 0.574.
        # (The 0.234 value belongs to random-walk Metropolis, not MALA.)
        ax.axhline(0.574, color="orange", linestyle="--", linewidth=0.9,
                   alpha=0.7)
    axes[0].set_ylabel("MALA acceptance rate (mean over seeds, $\\pm$SE band)")
    axes[0].legend(framealpha=0.9, ncol=2)
    fig.tight_layout()
    save(fig, "mala_acceptance")


# ---------------------------------------------------------------------------
# 8) BAOAB friction sensitivity (KS and ESS vs gamma)
# ---------------------------------------------------------------------------
def fig_baoab_gamma(grid_agg: pd.DataFrame):
    sub = grid_agg[grid_agg["algorithm"] == "BAOAB"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax_ks, ax_ess = axes
    for dist in DIST_ORDER:
        ss = sub[sub["distribution"] == dist]
        agg_ks = ss.groupby("gamma")["ks_stat_mean_central"].agg(
            central="mean",
            lo=lambda x: np.percentile(x, 25),
            hi=lambda x: np.percentile(x, 75),
        ).reset_index()
        agg_ess = ss.groupby("gamma")["ess_mean_central"].agg(
            central="mean",
            lo=lambda x: np.percentile(x, 25),
            hi=lambda x: np.percentile(x, 75),
        ).reset_index()
        line, = ax_ks.plot(agg_ks["gamma"], agg_ks["central"], marker="o",
                           label=DIST_LABEL[dist])
        ax_ks.fill_between(agg_ks["gamma"], agg_ks["lo"], agg_ks["hi"],
                           color=line.get_color(), alpha=0.18, linewidth=0)
        line2, = ax_ess.plot(agg_ess["gamma"], agg_ess["central"], marker="o",
                             label=DIST_LABEL[dist])
        ax_ess.fill_between(agg_ess["gamma"], agg_ess["lo"], agg_ess["hi"],
                            color=line2.get_color(), alpha=0.18, linewidth=0)
    for a, ylab in [(ax_ks, "KS (mean over $\\eta, d$; IQR band)"),
                    (ax_ess, "Mean ESS (mean over $\\eta, d$; IQR band)")]:
        a.set_xlabel(r"Friction $\gamma$")
        a.set_ylabel(ylab)
        a.set_yscale("log")
    ax_ks.legend()
    fig.tight_layout()
    save(fig, "baoab_gamma_sensitivity")


# ---------------------------------------------------------------------------
# 9) ULA bias inversion across tails (fixed eta=0.1, d=1) with error bars
# ---------------------------------------------------------------------------
def fig_ula_bias_inversion(grid_agg: pd.DataFrame):
    sub = grid_agg[(grid_agg["algorithm"] == "ULA") &
                   (grid_agg["d"] == 1) & (grid_agg["eta"] == 0.1)]
    if sub.empty:
        print("  skipping ULA bias inversion: no rows at eta=0.1, d=1")
        return
    sub = sub.set_index("distribution").reindex(DIST_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    x = np.arange(len(DIST_ORDER))
    width = 0.35

    def _bars(ax, lo_q, hi_q):
        for off, q, label, color in [(-width/2, lo_q, fr"${lo_q.split('q')[-1]}$", "#1f77b4"),
                                     (+width/2, hi_q, fr"${hi_q.split('q')[-1]}$", "#d62728")]:
            c = sub[f"{q}_central"].to_numpy().astype(float)
            lo = sub[f"{q}_low"].to_numpy().astype(float)
            hi = sub[f"{q}_high"].to_numpy().astype(float)
            err_lo = np.clip(c - lo, 0, None)
            err_hi = np.clip(hi - c, 0, None)
            ax.bar(x + off, c, width, color=color, edgecolor="black", linewidth=0.5,
                   yerr=[err_lo, err_hi], capsize=3,
                   error_kw=dict(elinewidth=0.8, ecolor="black"),
                   label=label)

    _bars(axes[0], "quantile_mae_q0.025", "quantile_mae_q0.975")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DIST_LABEL[d] for d in DIST_ORDER])
    axes[0].set_ylabel("Quantile MAE")
    axes[0].set_title(r"ULA quantile error at $\eta=0.1,\, d=1$")
    axes[0].legend(title="q")

    _bars(axes[1], "tail_cov_ratio_q0.01", "tail_cov_ratio_q0.99")
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6,
                    label="ideal")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DIST_LABEL[d] for d in DIST_ORDER])
    axes[1].set_ylabel("Tail coverage ratio")
    axes[1].set_title(r"ULA tail coverage at $\eta=0.1,\, d=1$")
    axes[1].legend(title="q")
    fig.tight_layout()
    save(fig, "ula_bias_inversion")


# ---------------------------------------------------------------------------
# 10) Seed-spread overlay (new): show all individual seeds plus mean for one
#     headline metric on one target, demonstrating seed variability.
# ---------------------------------------------------------------------------
def fig_seed_spread(conv_runs: pd.DataFrame):
    """Show per-seed KS curves overlaid with the mean for each algorithm,
    on the Cauchy target. Demonstrates that conclusions are not artifacts
    of a single seed."""
    sub = conv_runs[conv_runs["distribution"] == "cauchy"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for algo in ALGO_ORDER:
        ss = sub[sub["algorithm"] == algo]
        if ss.empty:
            continue
        for seed in sorted(ss["seed"].unique()):
            r = ss[ss["seed"] == seed].sort_values("checkpoint")
            ax.plot(r["checkpoint"], r["ks_stat_mean"], "-",
                    color=ALGO_COLOR[algo], alpha=0.18, linewidth=0.8)
        mean = ss.groupby("checkpoint")["ks_stat_mean"].mean().reset_index()
        ax.plot(mean["checkpoint"], mean["ks_stat_mean"], marker=ALGO_MARKER[algo],
                color=ALGO_COLOR[algo], label=f"{algo} (mean of {ss['seed'].nunique()} seeds)",
                linewidth=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Iterations $n$")
    ax.set_ylabel("KS statistic")
    ax.set_title("Per-seed KS trajectories on Cauchy")
    ax.legend()
    fig.tight_layout()
    save(fig, "seed_spread_cauchy")


def _load_grid_agg(extra_paths):
    """Load the main grid aggregated xlsx, optionally concatenated with
    additional aggregated xlsx files (e.g., the high-d extension)."""
    main_path = _latest("results_default_*_agg.xlsx")
    dfs = [pd.read_excel(main_path, sheet_name="aggregated")]
    paths_loaded = [main_path]
    for p in extra_paths:
        try:
            dfs.append(pd.read_excel(p, sheet_name="aggregated"))
            paths_loaded.append(p)
        except FileNotFoundError:
            pass
    return pd.concat(dfs, ignore_index=True), paths_loaded


# ---------------------------------------------------------------------------
# 11) Tail-focused dimension-scaling figures (uses combined grid)
# ---------------------------------------------------------------------------
def fig_ess_per_sec_gaussian(grid_agg: pd.DataFrame):
    """Plot ESS-per-second as a function of dimension at the best
    configuration per algorithm on the Gaussian target."""
    sub = grid_agg[(grid_agg["distribution"] == "gaussian")
                   & (grid_agg["divergence_rate_central"] == 0)].copy()
    sub = sub[(sub["algorithm"] != "MALA") |
              ((sub["acceptance_rate_central"] >= 0.2) &
               (sub["acceptance_rate_central"] <= 0.95))]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for algo in ALGO_ORDER:
        ss = sub[sub["algorithm"] == algo]
        rows = []
        for d in sorted(ss["d"].unique()):
            ssd = ss[ss["d"] == d]
            if ssd.empty:
                continue
            row = ssd.loc[ssd["ks_stat_mean_central"].idxmin()]
            rows.append({
                "d": d,
                "central": row["ess_per_sec_central"],
                "lo": row["ess_per_sec_low"],
                "hi": row["ess_per_sec_high"],
            })
        if not rows:
            continue
        x = [r["d"] for r in rows]
        c = np.array([r["central"] for r in rows], dtype=float)
        lo = np.array([r["lo"] for r in rows], dtype=float)
        hi = np.array([r["hi"] for r in rows], dtype=float)
        ax.fill_between(x, lo, hi, color=ALGO_COLOR[algo], alpha=0.18,
                        linewidth=0)
        ax.plot(x, c, marker=ALGO_MARKER[algo], color=ALGO_COLOR[algo],
                label=algo)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel("ESS per second (mean across seeds)")
    ax.set_title("Execution-time efficiency on the Gaussian target")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "ess_per_sec_gaussian")


def fig_cauchy_dimension_scaling(grid_agg: pd.DataFrame):
    """Three-panel plot of Cauchy diagnostics versus dimension:
    (a) KS, (b) quantile-MAE at q=0.975, (c) tail-coverage ratio at q=0.99.
    For each algorithm we plot the best configuration's median value across
    seeds, with IQR error bars."""
    sub = grid_agg[(grid_agg["distribution"] == "cauchy")
                   & (grid_agg["divergence_rate_central"] == 0)]
    # MALA acceptance filter for fair best-config selection
    sub = sub[(sub["algorithm"] != "MALA") |
              ((sub["acceptance_rate_central"] >= 0.2) &
               (sub["acceptance_rate_central"] <= 0.95))]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [
        ("ks_stat_mean", "KS statistic"),
        ("quantile_mae_q0.975", "Quantile MAE at $q=0.975$"),
        ("tail_cov_ratio_q0.99", "Tail coverage ratio at $q=0.99$"),
    ]
    for ax, (metric, ylab) in zip(axes, metrics):
        for algo in ALGO_ORDER:
            ss = sub[sub["algorithm"] == algo]
            # For each d, the best config = lowest ks_stat_mean_central
            rows_per_d = []
            for d in sorted(ss["d"].unique()):
                ssd = ss[ss["d"] == d]
                if ssd.empty:
                    continue
                row = ssd.loc[ssd["ks_stat_mean_central"].idxmin()]
                rows_per_d.append({
                    "d": d,
                    "central": row[f"{metric}_central"],
                    "low": row[f"{metric}_low"],
                    "high": row[f"{metric}_high"],
                })
            if not rows_per_d:
                continue
            x = [r["d"] for r in rows_per_d]
            c = np.array([r["central"] for r in rows_per_d], dtype=float)
            lo = np.array([r["low"] for r in rows_per_d], dtype=float)
            hi = np.array([r["high"] for r in rows_per_d], dtype=float)
            ax.fill_between(x, lo, hi, color=ALGO_COLOR[algo], alpha=0.18, linewidth=0)
            ax.plot(x, c, marker=ALGO_MARKER[algo], color=ALGO_COLOR[algo], label=algo)
        ax.set_xlabel(r"Dimension $d$")
        ax.set_xscale("log")
        ax.set_ylabel(ylab)
        if metric == "tail_cov_ratio_q0.99":
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        else:
            ax.set_yscale("log")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(r"Cauchy diagnostics versus dimension (best configuration per $d$)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "cauchy_dimension_scaling")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default=None,
                        help="Path to aggregated grid xlsx (default: latest results_default_*_agg.xlsx)")
    parser.add_argument("--conv", default=None,
                        help="Path to aggregated convergence xlsx (default: latest convergence_*_agg.xlsx)")
    parser.add_argument("--extra-grids", nargs="*", default=None,
                        help="Additional aggregated grid xlsx files to concatenate "
                             "(e.g. results_high_d_only_<ts>_agg.xlsx)")
    args = parser.parse_args()

    if args.grid:
        grid_agg = pd.read_excel(args.grid, sheet_name="aggregated")
        grid_paths = [args.grid]
    else:
        extras = args.extra_grids
        if extras is None:
            # Default: auto-detect a high-d xlsx if present
            import glob
            hd = sorted(glob.glob(os.path.join(RESULTS_DIR, "results_high_d_only_*_agg.xlsx")))
            extras = hd[-1:] if hd else []
        grid_agg, grid_paths = _load_grid_agg(extras)
    conv_path = args.conv or _latest("convergence_*_agg.xlsx")
    print(f"Grid : {grid_paths}")
    print(f"Conv : {conv_path}\n")

    conv_agg = pd.read_excel(conv_path, sheet_name="aggregated")
    conv_runs = pd.read_excel(conv_path, sheet_name="convergence_summary")

    print("Generating figures...")
    fig_ks_convergence(conv_agg)
    fig_ess_scaling(conv_agg)
    fig_quantile_mae_convergence(conv_agg)
    fig_cauchy_tail_coverage(conv_agg)
    fig_best_config_bars(grid_agg)
    fig_dimension_scaling(grid_agg)
    fig_mala_acceptance(grid_agg)
    fig_baoab_gamma(grid_agg)
    fig_ula_bias_inversion(grid_agg)
    fig_seed_spread(conv_runs)
    fig_cauchy_dimension_scaling(grid_agg)
    fig_ess_per_sec_gaussian(grid_agg)
    print(f"\nDone. Output: {FIG_DIR}")


if __name__ == "__main__":
    main()
