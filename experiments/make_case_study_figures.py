"""
Case studies of the thesis and the barrier-mass table of the double well
(plan §4.8, step 8).

Case studies: for each geometric target at its calibrated difficulty and d = 10,
the best and the worst algorithm by b2_grouped (best configuration of each
algorithm, from experiments/results/scaling_B_best.csv). One chain
(seed 0, N = 50,000, burn-in 5,000) per algorithm. Panels per algorithm:
  - 2D scatter of the samples on the contour of the exact 2D marginal
    (the marginal of the plotted pair equals the d = 2 target),
  - trace of the key coordinate,
  - autocorrelation of the key coordinate (up to lag 2,000).

Barrier-mass table: P(|x| < 0.1) per algorithm on the double well (beta = 16,
d = 10), against the exact value, from 5 chains of 200,000 steps each.

Saves PDF + PNG into experiments/figures/case_studies/ and the table
into experiments/figures/tables/.
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

from experiments.make_figures import (  # noqa: E402  (also sets rcParams)
    ALGO_COLOR, B_LABEL, B_ORDER, RESULTS_DIR, TABLE_DIR,
)
from experiments.run_experiments import ExperimentConfig, build_diagnostics, build_sampler  # noqa: E402
from targets import targets  # noqa: E402

FIG_DIR = os.path.join(_HERE, "figures", "case_studies")
os.makedirs(FIG_DIR, exist_ok=True)

D = 10
N_STEPS, BURN_IN, SEED = 50_000, 5_000, 0
N_EXACT = 1_000_000          # exact draws for the reference quantiles of the key coordinate
CASE_ORDER = ["student_t", "cauchy"] + B_ORDER   # the Gaussian case adds nothing to the Student-t one
PARAMS = {
    "gaussian": {}, "student_t": {}, "cauchy": {},
    "anisotropic_gaussian": {"kappa": 1000.0},
    "banana": {"b": 0.03},
    "gaussian_mixture": {"centers": (-6.0, 0.0, 6.0)},
    "funnel": {"sigma_v": 2.0},
    "double_well": {"beta": 16.0},
}
NU = {"student_t": 5.0}
# Plotted pair (x-axis, y-axis) and the key coordinate for trace / ACF / histogram / Q-Q.
PAIR = {"gaussian": (0, 1), "student_t": (0, 1), "cauchy": (0, 1),
        "anisotropic_gaussian": (0, D - 1), "banana": (0, 1), "gaussian_mixture": (0, 1),
        "funnel": (1, 0), "double_well": (0, 1)}
KEY = {"gaussian": 0, "student_t": 0, "cauchy": 0,
       "anisotropic_gaussian": D - 1, "banana": 1, "gaussian_mixture": 0, "funnel": 0, "double_well": 0}
KEY_LABEL = {"gaussian": r"$x_0$", "student_t": r"$x_0$", "cauchy": r"$x_0$",
             "anisotropic_gaussian": r"$x_{9}$ (widest)", "banana": r"$x_1$ (bent)",
             "gaussian_mixture": r"$x_0$ (modes)", "funnel": r"$v$", "double_well": r"$x_0$"}
AXIS_LABEL = {"gaussian": (r"$x_0$", r"$x_1$"), "student_t": (r"$x_0$", r"$x_1$"), "cauchy": (r"$x_0$", r"$x_1$"),
              "anisotropic_gaussian": (r"$x_0$ ($\lambda=1$)", r"$x_9$ ($\lambda=\kappa$)"),
              "banana": (r"$x_0$", r"$x_1$"), "gaussian_mixture": (r"$x_0$", r"$x_1$"),
              "funnel": (r"$x_1$", r"$v$"), "double_well": (r"$x_0$", r"$x_1$")}
TITLE = {"gaussian": "Gaussian", "student_t": r"Student-$t$ ($\nu=5$)", "cauchy": "Cauchy", **B_LABEL}
BLUE, RED = "C0", "red"      # the style of the original case-study figures


def _cfg(target, alg, eta, gamma, d=D, n_steps=N_STEPS, seed=SEED):
    return ExperimentConfig(algorithm=alg, distribution=target, d=d, eta=float(eta),
                            gamma=None if pd.isna(gamma) else float(gamma), nu=NU.get(target),
                            n_steps=n_steps, burn_in=BURN_IN, seed=seed,
                            target_params=dict(PARAMS[target]))


def _density_2d(target, xlim, ylim, n=300):
    """Exact 2D marginal of the plotted pair: the d = 2 target, evaluated with the
    potential of its MALA class. For the anisotropic target the d = 2 variances
    are (1, kappa), the extremes of the d = 10 target."""
    pot = build_sampler(_cfg(target, "MALA", 0.1, None, d=2, n_steps=1))
    xs, ys = np.linspace(*xlim, n), np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    i, j = (0, 1) if target != "funnel" else (1, 0)   # funnel: v is coordinate 0
    U = np.empty_like(X)
    for a in range(n):
        for b in range(n):
            z = np.empty(2)
            z[i], z[j] = X[a, b], Y[a, b]
            U[a, b] = pot.f(z)
    return X, Y, np.exp(-(U - U.min()))


def _acf(x, max_lag=2000):
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return ac / ac[0]


def _exact_key(target):
    """Exact i.i.d. draws of the key coordinate (reference for the histogram and Q-Q plot)."""
    kw = dict(PARAMS[target])
    if target == "student_t":
        kw["nu"] = NU[target]
    x = targets.exact_samples(target, D, N_EXACT, np.random.default_rng(12345), **kw)
    return x[:, KEY[target]]


def _best_rows(target):
    if target in ("gaussian", "student_t", "cauchy"):
        best = pd.read_csv(os.path.join(RESULTS_DIR, "scaling_A_best.csv"))
    else:
        best = pd.read_csv(os.path.join(RESULTS_DIR, "scaling_B_best.csv"))
    rows = best[(best["target"] == target) & (best["d"] == D)].sort_values("value")
    return [rows.iloc[0], rows.iloc[-1]]                # best and worst algorithm


def fig_case_study(target):
    metric = "KS" if target in ("gaussian", "student_t", "cauchy") else r"$b^2$"
    chains = [(r, build_sampler(_cfg(target, r["alg"], r["eta"], r["gamma"])).run())
              for r in _best_rows(target)]
    ref = _exact_key(target)
    probs = np.linspace(0.5, 99.5, 100)
    ref_q = np.percentile(ref, probs)
    marks = np.percentile(ref, [1, 5, 25, 50, 75, 95, 99])
    i, j = PAIR[target]
    k = KEY[target]
    allx = np.concatenate([c[:, i] for _, c in chains]); ally = np.concatenate([c[:, j] for _, c in chains])
    xlim = tuple(np.percentile(allx, [0.5, 99.5]) * 1.15); ylim = tuple(np.percentile(ally, [0.5, 99.5]) * 1.15)
    X, Y, P = _density_2d(target, xlim, ylim)

    fig = plt.figure(figsize=(7.2, 9.6), layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.04)
    for blk, (r, x) in enumerate(chains):
        sf = subfigs[blk]
        gs = sf.add_gridspec(2, 3)
        gamma = "" if pd.isna(r["gamma"]) else rf", $\gamma={r['gamma']:g}$"
        role = "best" if blk == 0 else "worst"
        head = rf"{role}: {r['alg']} ($\eta={r['eta']:g}${gamma}), {metric} $={r['value']:.4f}$"
        row = 0
        # 2D samples on the log-level contours of the exact density
        ax = sf.add_subplot(gs[row, 0])
        ax.contour(X, Y, P, levels=np.exp(-np.array([8, 6, 4.5, 3, 2, 1.2, 0.5])),
                   colors="grey", linewidths=0.6)
        ax.scatter(x[::20, i], x[::20, j], s=1.5, color=BLUE, alpha=0.35, linewidths=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(AXIS_LABEL[target][0]); ax.set_ylabel(AXIS_LABEL[target][1])
        ax.set_title("Samples on density contours", fontsize=9)
        # trace
        ax = sf.add_subplot(gs[row, 1:])
        ax.plot(x[:, k], color=BLUE, linewidth=0.4, alpha=0.8)
        ax.axhline(np.median(ref), color=RED, linestyle="--", linewidth=1.0, label="Theoretical median")
        ax.set_xlabel("Iteration (after burn-in)"); ax.set_ylabel(KEY_LABEL[target])
        ax.set_title(f"Trace of {KEY_LABEL[target]}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        # autocorrelation
        ax = sf.add_subplot(gs[row + 1, 0])
        ax.plot(_acf(x[:, k]), color=BLUE, linewidth=1.0)
        ax.axhline(0.05, color=RED, linestyle="--", linewidth=0.9)
        ax.set_ylim(-0.1, 1.02)
        ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Autocorrelation of {KEY_LABEL[target]}", fontsize=9)
        # histogram with the theoretical quantiles
        ax = sf.add_subplot(gs[row + 1, 1])
        lo, hi = np.percentile(x[:, k], [0.5, 99.5])
        lo, hi = min(lo, marks[0]), max(hi, marks[-1])
        ax.hist(x[:, k], bins=50, range=(lo, hi), density=True, alpha=0.7,
                color=BLUE, edgecolor="black", linewidth=0.4)
        for m in marks:
            ax.axvline(m, color=RED, linestyle="--", alpha=0.5, linewidth=0.9)
        ax.set_xlabel(KEY_LABEL[target]); ax.set_ylabel("Density")
        ax.set_title("Histogram, theoretical quantiles", fontsize=9)
        # Q-Q plot
        ax = sf.add_subplot(gs[row + 1, 2])
        sq = np.percentile(x[:, k], probs)
        ax.scatter(ref_q, sq, s=8, alpha=0.5, color=BLUE)
        lim = [min(ref_q.min(), sq.min()), max(ref_q.max(), sq.max())]
        ax.plot(lim, lim, "--", color=RED, linewidth=1.2)
        ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
        ax.set_title("Q-Q plot", fontsize=9)
        sf.suptitle(f"{TITLE[target]}, $d={D}$. {head[0].upper() + head[1:]}", fontsize=10)
    out = os.path.join(FIG_DIR, f"case_{target}")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")



def fig_ula_vs_mala(eta=0.3, n_steps=1_000_000, target="gaussian"):
    """ULA against MALA with the same step on the Gaussian target (d = 10): the only
    difference is the Metropolis test. Five panels per algorithm, as in the other case
    studies, and one shared panel with the running estimate of the variance against N.
    For ULA on N(0, 1) the stationary variance is 1 / (1 - eta / 2)."""
    chains = []
    for alg in ["ULA", "MALA"]:
        smp = build_sampler(_cfg(target, alg, eta, np.nan, n_steps=n_steps))
        x = smp.run()
        chains.append((alg, x, getattr(smp, "acceptance_rate", None)))
    ref = _exact_key(target)
    probs = np.linspace(0.5, 99.5, 100)
    ref_q = np.percentile(ref, probs)
    marks = np.percentile(ref, [1, 5, 25, 50, 75, 95, 99])
    var_ula = 1.0 / (1.0 - eta / 2.0)
    i, j, k = 0, 1, 0
    lim = 3.2 * np.sqrt(var_ula)
    X, Y, P = _density_2d(target, (-lim, lim), (-lim, lim))

    fig = plt.figure(figsize=(7.2, 10.2), layout="constrained")
    subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 0.55], hspace=0.04)
    for blk, (alg, x, acc) in enumerate(chains):
        sf = subfigs[blk]
        gs = sf.add_gridspec(2, 3)
        v = float(np.mean(x ** 2))
        extra = "" if acc is None else f", acceptance $={acc:.2f}$"
        sf.suptitle(rf"Gaussian, $d={D}$. {alg} ($\eta={eta:g}$){extra}, "
                    rf"variance estimate $={v:.3f}$ (true $1$)", fontsize=10)
        ax = sf.add_subplot(gs[0, 0])
        ax.contour(X, Y, P, levels=np.exp(-np.array([8, 6, 4.5, 3, 2, 1.2, 0.5])), colors="grey", linewidths=0.6)
        ax.scatter(x[::400, i], x[::400, j], s=1.5, color=BLUE, alpha=0.35, linewidths=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$x_0$"); ax.set_ylabel(r"$x_1$")
        ax.set_title("Samples on density contours", fontsize=9)
        ax = sf.add_subplot(gs[0, 1:])
        ax.plot(x[:50_000, k], color=BLUE, linewidth=0.4, alpha=0.8)
        ax.axhline(0.0, color=RED, linestyle="--", linewidth=1.0, label="Theoretical median")
        ax.set_xlabel("Iteration (after burn-in, first 50,000)"); ax.set_ylabel(r"$x_0$")
        ax.set_title(r"Trace of $x_0$", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax = sf.add_subplot(gs[1, 0])
        ax.plot(_acf(x[:, k], max_lag=200), color=BLUE, linewidth=1.0)
        ax.axhline(0.05, color=RED, linestyle="--", linewidth=0.9)
        ax.set_ylim(-0.1, 1.02)
        ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
        ax.set_title(r"Autocorrelation of $x_0$", fontsize=9)
        ax = sf.add_subplot(gs[1, 1])
        lo, hi = -lim, lim
        ax.hist(x[:, k], bins=60, range=(lo, hi), density=True, alpha=0.7, color=BLUE,
                edgecolor="black", linewidth=0.3)
        for m in marks:
            ax.axvline(m, color=RED, linestyle="--", alpha=0.5, linewidth=0.9)
        ax.set_xlabel(r"$x_0$"); ax.set_ylabel("Density")
        ax.set_title("Histogram, theoretical quantiles", fontsize=9)
        ax = sf.add_subplot(gs[1, 2])
        sq = np.percentile(x[:, k], probs)
        ax.scatter(ref_q, sq, s=8, alpha=0.5, color=BLUE)
        l2 = [min(ref_q.min(), sq.min()), max(ref_q.max(), sq.max())]
        ax.plot(l2, l2, "--", color=RED, linewidth=1.2)
        ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
        ax.set_title("Q-Q plot", fontsize=9)
    # running estimate of the variance (mean over the coordinates of the running mean of x^2)
    ax = subfigs[2].add_subplot(1, 1, 1)
    ns = np.unique(np.logspace(2, np.log10(n_steps - BURN_IN), 300).astype(int))
    for (alg, x, _), ls in zip(chains, ["-", "-"]):
        run = np.cumsum(np.mean(x ** 2, axis=1)) / np.arange(1, len(x) + 1)
        ax.plot(ns, run[ns - 1], color=ALGO_COLOR[alg], label=alg, linewidth=1.4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="true variance $1$")
    ax.axhline(var_ula, color=ALGO_COLOR["ULA"], linestyle=":", linewidth=1.2,
               label=rf"ULA stationary variance $1/(1-\eta/2)={var_ula:.3f}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Chain length $N$ (after burn-in)"); ax.set_ylabel(r"Running estimate of $\mathrm{Var}[x_i]$")
    ax.set_title("Convergence of the variance estimate", fontsize=10)
    ax.set_ylim(0.85, 1.35)
    ax.legend(loc="center right", ncol=2, fontsize=8, frameon=False)
    out = os.path.join(FIG_DIR, "case_ula_vs_mala_gaussian")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)
    for alg, x, acc in chains:
        print(f"  {alg}: variance {np.mean(x**2):.4f}, acceptance {acc}, "
              f"tail coverage q=0.99 {np.mean(x[:, 0] > np.percentile(ref, 99)) / 0.01:.2f}, "
              f"lag-1 acf {_acf(x[:, 0], 1)[1]:.3f}")
    print(f"  wrote {out}.pdf")


def table_barrier_mass(n_chains=5, n_steps=200_000):
    """P(|x| < 0.1) per algorithm on the double well, against the exact value."""
    m = targets.double_well_marginal(PARAMS["double_well"]["beta"])
    exact = float(m.cdf(0.1) - m.cdf(-0.1))
    best = pd.read_csv(os.path.join(RESULTS_DIR, "scaling_B_best.csv"))
    rows = best[(best["target"] == "double_well") & (best["d"] == D)].set_index("alg")
    lines = [r"\begin{tabular}{lrrcc}", r"\toprule",
             r"Αλγ. & $\eta$ & $\gamma$ & $P(\lvert x\rvert<0.1)$ & Λόγος προς τη σωστή τιμή \\", r"\midrule"]
    for alg in ["ULA", "MALA", "BAOAB"]:
        r = rows.loc[alg]
        p = np.mean([np.mean(np.abs(build_sampler(_cfg("double_well", alg, r["eta"], r["gamma"],
                                                       n_steps=n_steps, seed=s)).run()) < 0.1)
                     for s in range(n_chains)])
        gamma = "---" if pd.isna(r["gamma"]) else f"{r['gamma']:g}"
        lines.append(f"{alg} & {r['eta']:g} & {gamma} & {p:.5f} & {p / exact:.2f} \\\\")
        print(f"  {alg}: P(|x|<0.1) = {p:.5f} ({p / exact:.2f} x exact)")
    lines += [r"\midrule", rf"Σωστή τιμή & & & {exact:.5f} & 1.00 \\", r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLE_DIR, "double_well_barrier_mass.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tables/double_well_barrier_mass.tex")


def main():
    for target in CASE_ORDER:
        fig_case_study(target)
    fig_ula_vs_mala()
    table_barrier_mass()


if __name__ == "__main__":
    main()
