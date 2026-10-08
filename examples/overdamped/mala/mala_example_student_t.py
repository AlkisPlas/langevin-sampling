"""
Student-t (nu=5) -- MALA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.mala.mala_example_student_t
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import StudentTDiagnostics, StudentTMALA  # noqa: E402


def main():
    print("Running Student-t MALA sampler (nu=5, 1e6 steps)...")
    mala = StudentTMALA(d=1, eta=0.7, n_steps=1000000, burn_in=10000, nu=5, seed=42)
    samples_post = mala.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = StudentTDiagnostics(samples_post, location=np.zeros(1),
                                      scale=np.ones(1), nu=5,
                                      acceptance_rate=mala.acceptance_rate)
    diagnostics.print_comprehensive_stats()

    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
    diagnostics.plot_tail_exploration(dim=0)


if __name__ == "__main__":
    main()
