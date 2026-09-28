"""
Neal's funnel -- ULA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.ula.ula_example_funnel
"""

import os
import sys


_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import FunnelDiagnostics, FunnelULA  # noqa: E402


def main():
    print("Running Neal's Funnel ULA sampler...")
    ula = FunnelULA(d=3, eta=0.01, n_steps=100000, burn_in=10000, seed=42)
    samples_post = ula.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = FunnelDiagnostics(samples_post)
    diagnostics.print_comprehensive_stats()

    # dim 0 is the neck (log-scale); dims >= 1 live inside the funnel
    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0, 1])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
    diagnostics.plot_tail_exploration(dim=0)
    diagnostics.plot_quantile_mae_over_time()


if __name__ == "__main__":
    main()
