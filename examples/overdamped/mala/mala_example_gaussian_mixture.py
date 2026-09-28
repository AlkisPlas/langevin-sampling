"""
Three-component Gaussian mixture -- MALA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.mala.mala_example_gaussian_mixture
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import MIX_CENTERS, GaussianMixtureDiagnostics, GaussianMixtureMALA  # noqa: E402


def main():
    # Start inside the left mode, to expose mode-trapping.
    x0 = np.zeros(2)
    x0[0] = MIX_CENTERS[0]
    
    print("Running Gaussian Mixture (3 components) MALA sampler...")
    mala = GaussianMixtureMALA(d=2, eta=0.5, n_steps=100000, burn_in=10000,
                               x0=x0, seed=42)
    samples_post = mala.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = GaussianMixtureDiagnostics(samples_post,
                                             acceptance_rate=mala.acceptance_rate)
    diagnostics.print_comprehensive_stats()

    # dim 0 is trimodal, the other dims are unimodal N(0,1)
    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0, 1])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
    diagnostics.plot_tail_exploration(dim=0)
    diagnostics.plot_quantile_mae_over_time()


if __name__ == "__main__":
    main()
