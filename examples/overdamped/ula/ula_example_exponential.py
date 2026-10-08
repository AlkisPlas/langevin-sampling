"""
Exponential (rate 1) -- ULA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.ula.ula_example_exponential
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import ExponentialDiagnostics, ExponentialULA  # noqa: E402


def main():
    print("Running Exponential ULA sampler (Exp(1): mean 1, var 1)...")
    ula = ExponentialULA(d=1, eta=0.1, n_steps=100000, burn_in=10000,
                         rate=1.0, x0=np.ones(1), seed=42)
    samples_post = ula.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = ExponentialDiagnostics(samples_post, rate=1.0)
    diagnostics.print_comprehensive_stats()

    # ULA has no Metropolis test, so the boundary at x=0 is handled by a
    # large restoring gradient -- which biases the chain. Compare MALA.
    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
    diagnostics.plot_tail_exploration(dim=0)


if __name__ == "__main__":
    main()
