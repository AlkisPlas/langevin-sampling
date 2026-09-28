"""
Standard Cauchy -- BAOAB demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.underdamped.kinetic.splitting.baoab.kinetic_baoab_example_cauchy
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import CauchyBAOAB, CauchyDiagnostics  # noqa: E402


def main():
    print("Running Cauchy BAOAB sampler (1e6 steps)...")
    kinetic = CauchyBAOAB(d=3, eta=0.2, gamma=2, n_steps=1000000, burn_in=10000, seed=42)
    samples_post = kinetic.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = CauchyDiagnostics(samples_post, location=np.zeros(3), scale=1.0)
    diagnostics.print_comprehensive_stats()

    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0, 1, 2], max_samples=1000000)
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1, 2])
    diagnostics.plot_tail_exploration(dim=0)
    diagnostics.plot_quantile_mae_over_time()


if __name__ == "__main__":
    main()
