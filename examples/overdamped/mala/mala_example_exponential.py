"""
Exponential (rate 1) -- MALA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.mala.mala_example_exponential
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import ExponentialDiagnostics, ExponentialMALA  # noqa: E402


def main():
    print("Running Exponential MALA sampler (Exp(1): mean 1, var 1)...")
    mala = ExponentialMALA(d=3, eta=0.1, n_steps=100000, burn_in=10000,
                           rate=1.0, x0=np.ones(3), seed=42)
    samples_post = mala.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = ExponentialDiagnostics(samples_post, rate=1.0,
                                         acceptance_rate=mala.acceptance_rate)
    diagnostics.print_comprehensive_stats()

    # MALA returns f = +inf outside x > 0, so the Metropolis test rejects
    # boundary violations exactly -- no bias, unlike the ULA demo.
    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0, 1])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
    diagnostics.plot_tail_exploration(dim=0)


if __name__ == "__main__":
    main()
