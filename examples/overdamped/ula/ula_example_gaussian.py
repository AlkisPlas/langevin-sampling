"""
Standard Gaussian -- ULA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.ula.ula_example_gaussian
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import GaussianDiagnostics, GaussianULA  # noqa: E402


def main():
    print("Running Gaussian ULA sampler...")
    ula = GaussianULA(d=1, eta=0.3, n_steps=100000, burn_in=10000, seed=42)
    samples_post = ula.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = GaussianDiagnostics(samples_post, mu=np.zeros(1), Sigma=np.eye(1))
    diagnostics.print_comprehensive_stats()

    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
    diagnostics.plot_tail_exploration(dim=0)
    diagnostics.plot_quantile_mae_over_time()


if __name__ == "__main__":
    main()
