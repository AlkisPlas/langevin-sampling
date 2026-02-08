import numpy as np
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from overdamped.ula.ula_runner import ULA

class DoubleWellULA(ULA):
    def grad_f(self, x):
        # f(x) = sum_i (1/4 x_i^4 - 1/2 x_i^2)
        return x**3 - x

class DoubleWellDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975],
                        acceptance_rate=None)  # ULA has no acceptance step

    def quantile(self, p, dim=None):
        # No closed-form theoretical quantiles for double-well
        # Return NaN - some diagnostic methods will skip quantile-based checks
        return np.nan

# Run the sampler
print("Running Double-Well ULA sampler...")
ula = DoubleWellULA(d=1, eta=0.15, n_steps=1_000_000, burn_in=10_000, seed=42)
samples_post = ula.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = DoubleWellDiagnostics(samples_post)
diagnostics.print_comprehensive_stats()

# Visualizations
print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
diagnostics.plot_tail_exploration(dim=0)