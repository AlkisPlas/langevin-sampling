"""
Double well (non-smooth) -- MALA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.mala.mala_example_double_well
"""

import os
import sys


_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import DoubleWellDiagnostics, DoubleWellMALA  # noqa: E402


def main():
    print("Running Double-Well MALA sampler...")
    mala = DoubleWellMALA(d=1, eta=1, n_steps=100000, burn_in=10000, seed=42)
    samples_post = mala.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = DoubleWellDiagnostics(samples_post,
                                        acceptance_rate=mala.acceptance_rate)
    diagnostics.print_comprehensive_stats()

    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
    diagnostics.plot_tail_exploration(dim=0)


if __name__ == "__main__":
    main()
