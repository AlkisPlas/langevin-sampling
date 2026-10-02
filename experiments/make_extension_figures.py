"""
Generate the figures and tables of the thesis extension (plan, step 11) from the
aggregated xlsx files in experiments/results/extension/.

Part B (the five geometric targets, at their calibrated difficulty):
  b_best_config_bars       best configuration at d = 10: b^2 and KS
  b_dimension_scaling_b2   b^2 against d (2 .. 1000)
  b_dimension_scaling_ks   KS against d
  b_ess_vs_d               ESS and ESS/sec against d
  b_tail_qmae_vs_d         tail coverage (q = 0.99) and q-MAE (q = 0.975) against d
  b_mala_acceptance        MALA acceptance against eta, one line per d
  b_baoab_gamma            BAOAB: b^2 and ESS against gamma
  b_convergence_*          b^2, KS, ESS and tail coverage against N (E6)
  calibration_b2           supporting: b^2 against the candidate difficulty value
Part A (the three tail-weight targets), extended to d = 1000:
  a_dimension_scaling_ks, a_ess_per_sec_gaussian, a_cauchy_dimension_scaling
Tables (LaTeX, in figures/extension/tables/):
  calibration_selected.tex, b_best_configs_d10.tex

"Best configuration" = lowest median of the primary metric (Part B: b2_grouped;
Part A: KS) among the configurations with divergence_rate == 0 and, for MALA,
acceptance in [0.2, 0.95] - the same filters as the original thesis. Scaling
figures use the best configuration at each d, because the eta grids change with
d (an average over a grid is not comparable across d).

Style (colours, markers, fonts) is shared with make_thesis_figures.py.
Saves PDF + PNG into experiments/figures/extension/.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.make_thesis_figures import (  # noqa: E402  (also sets rcParams)
    ALGO_COLOR, ALGO_MARKER, ALGO_ORDER, DIST_LABEL, save,
)

EXT = os.path.join(_HERE, "results", "extension")
ORIG = os.path.join(_HERE, "results", "original_thesis")
TABLE_DIR = os.path.join(_HERE, "figures", "extension", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

B_ORDER = ["anisotropic_gaussian", "banana", "gaussian_mixture", "funnel", "double_well"]
B_LABEL = {
    "anisotropic_gaussian": r"Anisotropic ($\kappa=1000$)",
    "banana": r"Banana ($b=0.03$)",
    "gaussian_mixture": r"Mixture ($a=6$)",
    "funnel": r"Funnel ($\sigma_v=2$)",
    "double_well": r"Double well ($\beta=16$)",
}
# The calibrated difficulty of each target (plan §4.8, step 8, table T2).
SELECTED = {
    "anisotropic_gaussian": "kappa=1000.0", "banana": "b=0.03",
    "gaussian_mixture": "centers=(-6.0, 0.0, 6.0)", "funnel": "sigma_v=2.0",
    "double_well": "beta=16.0",
}
# Coordinates without a closed-form marginal are left out of KS, q-MAE and tail
# coverage (plan §5): the bent banana coordinates and the funnel x_i.
PARTIAL_NOTE = "KS, q-MAE, tail coverage: banana even coordinates only, funnel $v$ only"
B2_THRESHOLD = 0.01
TAIL_YMAX = 2.5  # y-limit of the tail-coverage panels; wider IQR bands are clipped

SCALING_B = ["results_scaling_B1_20261001_204724", "results_scaling_B1_extra_20261001_220733",
             "results_scaling_B2_20261001_223034", "results_scaling_B2_extra_20261002_104132"]
SCALING_A = ["results_scaling_A_high_20261002_024016", "results_scaling_A_high_extra_20261002_094835"]
CALIBRATION = ["results_calibration_20260929_143922", "results_calibration_extra_20260930_104420"]
CONVERGENCE_B = "convergence_extension_20261002_122239"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _agg(names, folder=EXT):
    dfs = [pd.read_excel(os.path.join(folder, n + "_agg.xlsx"), sheet_name="aggregated") for n in names]
    df = pd.concat(dfs, ignore_index=True)
    if "target_params" in df.columns:
        df["target_params"] = df["target_params"].fillna("")
    return df


def _valid(df):
    df = df[df["divergence_rate_central"] == 0]
    return df[(df["algorithm"] != "MALA") |
              ((df["acceptance_rate_central"] >= 0.2) & (df["acceptance_rate_central"] <= 0.95))]


def _best(df, metric, by=("distribution", "algorithm", "d")):
    """One row per group: the valid configuration with the lowest median metric."""
    v = _valid(df).dropna(subset=[f"{metric}_central"])
    idx = v.groupby(list(by))[f"{metric}_central"].idxmin()
    return v.loc[idx].reset_index(drop=True)


def _line(ax, x, c, lo, hi, algo, alpha=0.18):
    ax.fill_between(x, lo, hi, color=ALGO_COLOR[algo], alpha=alpha, linewidth=0)
    ax.plot(x, c, marker=ALGO_MARKER[algo], color=ALGO_COLOR[algo], label=algo)


def _per_target_panels(n=5, h=3.0, sharex=True):
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, h), sharex=sharex)
    return fig, np.atleast_1d(axes)


def _metric_vs(ax, best, target, x, metric, log_y=True):
    sub = best[best["distribution"] == target]
    for algo in ALGO_ORDER:
        r = sub[sub["algorithm"] == algo].sort_values(x)
        if r.empty:
            continue
        _line(ax, r[x].to_numpy(float), r[f"{metric}_central"].to_numpy(float),
              r[f"{metric}_low"].to_numpy(float), r[f"{metric}_high"].to_numpy(float), algo)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


# ---------------------------------------------------------------------------
# Part B figures
# ---------------------------------------------------------------------------
def fig_b_best_config_bars(best_b):
    d10 = best_b[best_b["d"] == 10]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    width = 0.26
    x = np.arange(len(B_ORDER))
    for ax, metric, ylab in [(axes[0], "b2_grouped", r"$b^2_{\mathrm{grouped}}$ (median, IQR)"),
                             (axes[1], "ks_stat_mean", "KS (median, IQR)")]:
        for i, algo in enumerate(ALGO_ORDER):
            r = d10[d10["algorithm"] == algo].set_index("distribution").reindex(B_ORDER)
            c = r[f"{metric}_central"].to_numpy(float)
            err = np.vstack([c - r[f"{metric}_low"].to_numpy(float), r[f"{metric}_high"].to_numpy(float) - c])
            ax.bar(x + (i - 1) * width, c, width * 0.92, yerr=err, color=ALGO_COLOR[algo],
                   label=algo, capsize=2, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([B_LABEL[t].split(" (")[0] for t in B_ORDER], rotation=15)
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
    axes[0].axhline(B2_THRESHOLD, color="black", linestyle="--", linewidth=0.9)
    axes[0].legend(loc="upper left")
    axes[0].set_title(r"$b^2_{\mathrm{grouped}}$, $d=10$ (dashed: threshold 0.01)")
    axes[1].set_title(r"KS, $d=10$")
    fig.tight_layout()
    save(fig, "b_best_config_bars")


def fig_b_dimension_scaling(best_b, metric, name, ylab, threshold=None):
    fig, axes = _per_target_panels()
    for ax, target in zip(axes, B_ORDER):
        _metric_vs(ax, best_b, target, "d", metric)
        if threshold is not None:
            ax.axhline(threshold, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(B_LABEL[target])
        ax.set_xlabel(r"Dimension $d$")
    axes[0].set_ylabel(ylab)
    axes[0].legend(loc="best")
    fig.tight_layout()
    save(fig, name)


def fig_b_ess(best_b):
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharex=True)
    for j, target in enumerate(B_ORDER):
        _metric_vs(axes[0, j], best_b, target, "d", "ess_mean")
        _metric_vs(axes[1, j], best_b, target, "d", "ess_per_sec")
        axes[0, j].set_title(B_LABEL[target])
        axes[1, j].set_xlabel(r"Dimension $d$")
    axes[0, 0].set_ylabel("ESS (mean, SE)")
    axes[1, 0].set_ylabel("ESS per second (mean, SE)")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    save(fig, "b_ess_vs_d")


def fig_b_tail_qmae(best_b):
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharex=True)
    for j, target in enumerate(B_ORDER):
        _metric_vs(axes[0, j], best_b, target, "d", "tail_cov_ratio_q0.99", log_y=False)
        axes[0, j].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        axes[0, j].set_ylim(0.0, TAIL_YMAX)  # wide IQR bands (banana, funnel) are clipped
        _metric_vs(axes[1, j], best_b, target, "d", "quantile_mae_q0.975")
        axes[0, j].set_title(B_LABEL[target])
        axes[1, j].set_xlabel(r"Dimension $d$")
    axes[0, 0].set_ylabel(r"Tail coverage ratio, $q=0.99$")
    axes[1, 0].set_ylabel(r"q-MAE, $q=0.975$")
    axes[0, 0].legend(loc="best")
    fig.suptitle(PARTIAL_NOTE + f"; tail-coverage axis limited to [0, {TAIL_YMAX:g}]", fontsize=9, y=1.0)
    fig.tight_layout()
    save(fig, "b_tail_qmae_vs_d")


def fig_b_mala_acceptance(grid_b):
    sub = grid_b[grid_b["algorithm"] == "MALA"]
    ds = sorted(sub["d"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {d: cmap(i / max(len(ds) - 1, 1)) for i, d in enumerate(ds)}
    fig, axes = _per_target_panels(h=3.2)
    for ax, target in zip(axes, B_ORDER):
        t = sub[sub["distribution"] == target]
        for d in ds:
            r = t[t["d"] == d].sort_values("eta")
            if r.empty:
                continue
            ax.plot(r["eta"], r["acceptance_rate_central"], marker="o", markersize=3.5,
                    color=colors[d], label=f"$d={d}$")
        ax.axhspan(0.2, 0.95, color="grey", alpha=0.10, linewidth=0)
        ax.set_xscale("log")
        ax.set_title(B_LABEL[target])
        ax.set_xlabel(r"Step size $\eta$")
    axes[0].set_ylabel("MALA acceptance rate (mean)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    save(fig, "b_mala_acceptance")


def fig_b_baoab_gamma(grid_b):
    """Median over the (eta, d) configurations of each gamma, as in the original
    thesis (which used the mean of KS); the median is used for b^2 because b^2 is
    strongly skewed across configurations."""
    sub = _valid(grid_b[grid_b["algorithm"] == "BAOAB"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    gammas = sorted(sub["gamma"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    for k, target in enumerate(B_ORDER):
        t = sub[sub["distribution"] == target]
        b2 = [t[t["gamma"] == g]["b2_grouped_central"].median() for g in gammas]
        ess = [t[t["gamma"] == g]["ess_mean_central"].median() for g in gammas]
        style = dict(marker="os^Dv"[k], color=cmap(k + 3), label=B_LABEL[target])
        axes[0].plot(gammas, b2, **style)
        axes[1].plot(gammas, ess, **style)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$b^2_{\mathrm{grouped}}$ (median over $\eta$, $d$)")
    axes[1].set_ylabel(r"ESS (median over $\eta$, $d$)")
    for ax in axes:
        ax.set_xlabel(r"Friction $\gamma$")
        ax.set_xticks(gammas)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save(fig, "b_baoab_gamma")


def fig_b_convergence(conv_b):
    specs = [
        ("b2_grouped", r"$b^2_{\mathrm{grouped}}$", -1.0, True, "b_convergence_b2"),
        ("ks_stat_mean", "KS", -0.5, True, "b_convergence_ks"),
        ("ess_mean", "ESS", 1.0, True, "b_convergence_ess"),
        ("tail_cov_ratio_q0.99", r"Tail coverage ratio, $q=0.99$", None, False, "b_convergence_tail"),
    ]
    targets = [t for t in B_ORDER if t in set(conv_b["distribution"])]
    for metric, ylab, slope, log_y, name in specs:
        fig, axes = _per_target_panels(n=len(targets), h=3.1)
        for ax, target in zip(axes, targets):
            _metric_vs(ax, conv_b, target, "checkpoint", metric, log_y=log_y)
            if slope is not None:
                n = np.array([5e4, 1e6])
                ref = conv_b[(conv_b["distribution"] == target) & (conv_b["checkpoint"] == 50_000)][f"{metric}_central"]
                y0 = float(np.nanmedian(ref))
                ax.plot(n, y0 * (n / n[0]) ** slope, color="grey", linestyle="--", linewidth=0.9,
                        label=f"slope {slope:g}")
            if metric == "b2_grouped":
                ax.axhline(B2_THRESHOLD, color="black", linestyle=":", linewidth=0.9)
            if metric.startswith("tail_cov"):
                ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
            ax.set_title(B_LABEL[target])
            ax.set_xlabel(r"Chain length $N$")
        axes[0].set_ylabel(ylab)
        axes[0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        save(fig, name)


def fig_calibration(sel):
    """Supporting figure: b^2 at the best configuration against the candidate
    difficulty value; the selected value is marked."""
    value = {
        "anisotropic_gaussian": ("kappa", r"$\kappa$", True),
        "banana": ("b", r"$b$", True),
        "gaussian_mixture": ("centers", r"$a$", False),
        "funnel": ("sigma_v", r"$\sigma_v$", False),
        "double_well": ("beta", r"$\beta$", True),
    }

    def param(target, p):
        if target == "gaussian_mixture":
            return float(p.split("(")[1].split(",")[0].replace("-", ""))
        return float(p.split("=")[1])

    fig, axes = _per_target_panels(h=3.0, sharex=False)
    for ax, target in zip(axes, B_ORDER):
        t = sel[sel["target"] == target].copy()
        t["x"] = [param(target, p) for p in t["params"]]
        for algo in ALGO_ORDER:
            r = t[t["alg"] == algo].sort_values("x")
            _line(ax, r["x"].to_numpy(), r["b2g"].to_numpy(), r["lo"].to_numpy(), r["hi"].to_numpy(), algo)
        ax.axhline(B2_THRESHOLD, color="black", linestyle="--", linewidth=0.9)
        ax.axvline(param(target, SELECTED[target]), color="grey", linewidth=6, alpha=0.25)
        key, lab, logx = value[target]
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.set_title(B_LABEL[target].split(" (")[0])
    axes[0].set_ylabel(r"$b^2_{\mathrm{grouped}}$ (median, IQR), $d=10$")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save(fig, "calibration_b2")


# ---------------------------------------------------------------------------
# Part A figures (extended to d = 1000)
# ---------------------------------------------------------------------------
def fig_a_scaling(best_a):
    targets = ["gaussian", "student_t", "cauchy"]
    fig, axes = _per_target_panels(n=3, h=3.3)
    for ax, target in zip(axes, targets):
        _metric_vs(ax, best_a, target, "d", "ks_stat_mean")
        ax.set_title(DIST_LABEL[target])
        ax.set_xlabel(r"Dimension $d$")
    axes[0].set_ylabel("KS at the best configuration (median, IQR)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    save(fig, "a_dimension_scaling_ks")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    _metric_vs(ax, best_a, "gaussian", "d", "ess_per_sec")
    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel("ESS per second (mean across seeds)")
    ax.set_title("Execution-time efficiency on the Gaussian target")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "a_ess_per_sec_gaussian")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (metric, ylab, logy) in zip(axes, [("ks_stat_mean", "KS statistic", True),
                                               ("quantile_mae_q0.975", "Quantile MAE at $q=0.975$", True),
                                               ("tail_cov_ratio_q0.99", "Tail coverage ratio at $q=0.99$", False)]):
        _metric_vs(ax, best_a, "cauchy", "d", metric, log_y=logy)
        if not logy:
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel(r"Dimension $d$")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="best")
    fig.suptitle(r"Cauchy diagnostics versus dimension (best configuration per $d$)", fontsize=11)
    fig.tight_layout()
    save(fig, "a_cauchy_dimension_scaling")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _fmt(v):
    return "---" if pd.isna(v) else (f"{v:.4f}" if v >= 1e-3 else f"{v:.1e}")


def table_calibration(sel):
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Στόχος & Τιμή & ULA & MALA & BAOAB \\", r"\midrule"]
    for target in B_ORDER:
        r = sel[(sel["target"] == target) & (sel["params"] == SELECTED[target])].set_index("alg")
        cells = [(_fmt(r.loc[a, "b2g"]) + (r"$^{\checkmark}$" if r.loc[a, "b2g"] < B2_THRESHOLD else ""))
                 for a in ALGO_ORDER]
        lines.append(f"{B_LABEL[target].split(' (')[0]} & {B_LABEL[target].split('(')[1].rstrip(')')} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLE_DIR, "calibration_selected.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tables/calibration_selected.tex")


def table_best_configs(best_b):
    d10 = best_b[best_b["d"] == 10]
    lines = [r"\begin{tabular}{llrrcccc}", r"\toprule",
             r"Αλγ. & Στόχος & $\eta$ & $\gamma$ & $b^2_{\mathrm{grouped}}$ & KS & ESS & Αποδοχή \\",
             r"\midrule"]
    for algo in ALGO_ORDER:
        for target in B_ORDER:
            r = d10[(d10["algorithm"] == algo) & (d10["distribution"] == target)].iloc[0]
            gamma = "---" if pd.isna(r["gamma"]) else f"{r['gamma']:g}"
            acc = f"{r['acceptance_rate_central']:.2f}" if algo == "MALA" else "---"
            lines.append(f"{algo} & {B_LABEL[target].split(' (')[0]} & {r['eta']:g} & {gamma} & "
                         f"{_fmt(r['b2_grouped_central'])} & {_fmt(r['ks_stat_mean_central'])} & "
                         f"{r['ess_mean_central']:,.0f} & {acc} \\\\")
        lines.append(r"\midrule" if algo != ALGO_ORDER[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(os.path.join(TABLE_DIR, "b_best_configs_d10.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tables/b_best_configs_d10.tex")


# ---------------------------------------------------------------------------
def main():
    print("Loading aggregated results...")
    grid_b = _agg(SCALING_B)
    best_b = _best(grid_b, "b2_grouped")
    conv_b = pd.read_excel(os.path.join(EXT, CONVERGENCE_B + "_agg.xlsx"), sheet_name="aggregated")
    sel = pd.read_csv(os.path.join(EXT, "calibration_selection_b2_grouped.csv"))
    sel["params"] = sel["params"].fillna("")
    grid_a = pd.concat([_agg(["results_default_20260525_104845", "results_high_d_only_regenerated"], ORIG),
                        _agg(SCALING_A)], ignore_index=True)
    best_a = _best(grid_a, "ks_stat_mean")

    print("Part B figures...")
    fig_b_best_config_bars(best_b)
    fig_b_dimension_scaling(best_b, "b2_grouped", "b_dimension_scaling_b2",
                            r"$b^2_{\mathrm{grouped}}$ at the best configuration", threshold=B2_THRESHOLD)
    fig_b_dimension_scaling(best_b, "ks_stat_mean", "b_dimension_scaling_ks", "KS at the best configuration")
    fig_b_ess(best_b)
    fig_b_tail_qmae(best_b)
    fig_b_mala_acceptance(grid_b)
    fig_b_baoab_gamma(grid_b)
    fig_b_convergence(conv_b)
    fig_calibration(sel)
    print("Part A figures...")
    fig_a_scaling(best_a)
    print("Tables...")
    table_calibration(sel)
    table_best_configs(best_b)
    print("Done.")


if __name__ == "__main__":
    main()