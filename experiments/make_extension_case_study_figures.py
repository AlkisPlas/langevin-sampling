"""
Case studies of the thesis extension (plan E5) and the barrier-mass table of the
double well (plan §4.8, step 8).

Case studies: for each geometric target at its calibrated difficulty and d = 10,
the best and the worst algorithm by b2_grouped (best configuration of each
algorithm, from experiments/results/extension/scaling_B_best.csv). One chain
(seed 0, N = 50,000, burn-in 5,000) per algorithm. Panels per algorithm:
  - 2D scatter of the samples on the contour of the exact 2D marginal
    (the marginal of the plotted pair equals the d = 2 target),
  - trace of the key coordinate,
  - autocorrelation of the key coordinate (up to lag 2,000).

Barrier-mass table: P(|x| < 0.1) per algorithm on the double well (beta = 16,
d = 10), against the exact value, from 5 chains of 200,000 steps each.

Saves PDF + PNG into experiments/figures/extension/case_studies/ and the table
into experiments/figures/extension/tables/.
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

from experiments.make_thesis_figures import ALGO_COLOR  # noqa: E402  (also sets rcParams)
from experiments.make_extension_figures import B_LABEL, B_ORDER, TABLE_DIR  # noqa: E402
from experiments.run_experiments import ExperimentConfig, build_diagnostics, build_sampler  # noqa: E402
from targets import targets  # noqa: E402

FIG_DIR = os.path.join(_HERE, "figures", "extension", "case_studies")
os.makedirs(FIG_DIR, exist_ok=True)

D = 10
N_STEPS, BURN_IN, SEED = 50_000, 5_000, 0
PARAMS = {
    "anisotropic_gaussian": {"kappa": 1000.0},
    "banana": {"b": 0.03},
    "gaussian_mixture": {"centers": (-6.0, 0.0, 6.0)},
    "funnel": {"sigma_v": 2.0},
    "double_well": {"beta": 16.0},
}
# Plotted pair (x-axis, y-axis) and the key coordinate for the trace / ACF.
PAIR = {"anisotropic_gaussian": (0, D - 1), "banana": (0, 1), "gaussian_mixture": (0, 1),
        "funnel": (1, 0), "double_well": (0, 1)}
KEY = {"anisotropic_gaussian": D - 1, "banana": 1, "gaussian_mixture": 0, "funnel": 0, "double_well": 0}
KEY_LABEL = {"anisotropic_gaussian": r"$x_{9}$ (widest)", "banana": r"$x_1$ (bent)",
             "gaussian_mixture": r"$x_0$ (modes)", "funnel": r"$v$", "double_well": r"$x_0$"}
AXIS_LABEL = {"anisotropic_gaussian": (r"$x_0$ ($\lambda=1$)", r"$x_9$ ($\lambda=\kappa$)"),
              "banana": (r"$x_0$", r"$x_1$"), "gaussian_mixture": (r"$x_0$", r"$x_1$"),
              "funnel": (r"$x_1$", r"$v$"), "double_well": (r"$x_0$", r"$x_1$")}


def _cfg(target, alg, eta, gamma, d=D, n_steps=N_STEPS, seed=SEED):
    return ExperimentConfig(algorithm=alg, distribution=target, d=d, eta=float(eta),
                            gamma=None if pd.isna(gamma) else float(gamma), nu=None,
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


def fig_case_study(target, best):
    rows = best[(best["target"] == target) & (best["d"] == D)].sort_values("b2")
    chosen = [rows.iloc[0], rows.iloc[-1]]              # best and worst algorithm
    chains = []
    for r in chosen:
        s = build_sampler(_cfg(target, r["alg"], r["eta"], r["gamma"]))
        chains.append((r, s.run()))
    i, j = PAIR[target]
    allx = np.concatenate([c[:, i] for _, c in chains]); ally = np.concatenate([c[:, j] for _, c in chains])
    xlim = tuple(np.percentile(allx, [0.5, 99.5]) * 1.15); ylim = tuple(np.percentile(ally, [0.5, 99.5]) * 1.15)
    X, Y, P = _density_2d(target, xlim, ylim)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), gridspec_kw={"width_ratios": [1.0, 1.6, 1.0]})
    for row, (r, x) in enumerate(chains):
        color = ALGO_COLOR[r["alg"]]
        gamma = "" if pd.isna(r["gamma"]) else rf", $\gamma={r['gamma']:g}$"
        head = rf"{r['alg']} ($\eta={r['eta']:g}${gamma}), $b^2_{{\mathrm{{grouped}}}}={r['b2']:.4f}$"
        ax = axes[row, 0]
        # Levels on the log density: with linear levels a sharply peaked density
        # (the funnel neck) puts every line near the peak.
        ax.contour(X, Y, P, levels=np.exp(-np.array([8, 6, 4.5, 3, 2, 1.2, 0.5])),
                   colors="grey", linewidths=0.7)
        ax.scatter(x[::20, i], x[::20, j], s=2, color=color, alpha=0.35, linewidths=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(AXIS_LABEL[target][0]); ax.set_ylabel(AXIS_LABEL[target][1])
        ax.set_title(head, fontsize=9)
        ax = axes[row, 1]
        ax.plot(x[:, KEY[target]], color=color, linewidth=0.4)
        ax.set_xlabel("Iteration (after burn-in)"); ax.set_ylabel(KEY_LABEL[target])
        ax.set_title(f"Trace of {KEY_LABEL[target]}", fontsize=9)
        ax = axes[row, 2]
        ac = _acf(x[:, KEY[target]])
        ax.plot(ac, color=color)
        ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.1, 1.02)
        ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Autocorrelation of {KEY_LABEL[target]}", fontsize=9)
    fig.suptitle(rf"{B_LABEL[target]}, $d={D}$: best (top) and worst (bottom) algorithm", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"case_{target}")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)
    print(f"  wrote {out}.pdf")


def table_barrier_mass(best, n_chains=5, n_steps=200_000):
    """P(|x| < 0.1) per algorithm on the double well, against the exact value."""
    m = targets.double_well_marginal(PARAMS["double_well"]["beta"])
    exact = float(m.cdf(0.1) - m.cdf(-0.1))
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
    best = pd.read_csv(os.path.join(_HERE, "results", "extension", "scaling_B_best.csv"))
    best = best.rename(columns={"value": "b2"})
    for target in B_ORDER:
        fig_case_study(target, best)
    table_barrier_mass(best)


if __name__ == "__main__":
    main()
