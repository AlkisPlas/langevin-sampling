"""
Anisotropic (ill-conditioned) Gaussian -- ULA demo.

The target potential and its diagnostics are imported from targets/targets.py,
which is the single source of truth for every target distribution in this
project. This script only drives the sampler and reports diagnostics; it defines
no potential of its own, so importing it has no side effects.

Run from the repo root:
    python -m examples.overdamped.ula.ula_example_anisotropic_gaussian
"""

import os
import sys


_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from targets.targets import AnisotropicGaussianDiagnostics, AnisotropicGaussianULA  # noqa: E402


def main():
    print("Running Anisotropic (ill-conditioned) Gaussian ULA sampler...")
    ula = AnisotropicGaussianULA(d=5, eta=0.1, n_steps=100000, burn_in=10000,
                                 kappa=100.0, seed=42)
    samples_post = ula.run()

    print("\nGenerating comprehensive diagnostics...\n")
    diagnostics = AnisotropicGaussianDiagnostics(samples_post, kappa=100.0)
    diagnostics.print_comprehensive_stats()

    # dim 0 is the stiffest (var=1), the last dim the widest (var=kappa)
    print("\nGenerating visualizations...")
    diagnostics.plot_trace(dims=[0, 4])
    diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 4])
    diagnostics.plot_tail_exploration(dim=4)
    diagnostics.plot_quantile_mae_over_time()


if __name__ == "__main__":
    main()
