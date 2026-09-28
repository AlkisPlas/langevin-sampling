"""
Banana (Rosenbrock) -- ULA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.ula.ula_example_banana
"""

import os
import sys


_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import BananaDiagnostics, BananaULA  # noqa: E402


def main():
    print("Running Banana (Rosenbrock) ULA sampler...")
    ula = BananaULA(d=2, eta=0.01, n_steps=100000, burn_in=10000, seed=42)
    samples_post = ula.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = BananaDiagnostics(samples_post)
    diagnostics.print_comprehensive_stats()

    # even dims are the wide axes, odd dims the bent ones
    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0, 1])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
    diagnostics.plot_tail_exploration(dim=0)


if __name__ == "__main__":
    main()
