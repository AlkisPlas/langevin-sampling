import numpy as np
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from overdamped.mala.mala_runner import MALA

class DoubleWellMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)

    def f(self, x):
        # Double-well potential: f(x) = sum_i (1/4 x_i^4 - 1/2 x_i^2)
        return np.sum(0.25 * x**4 - 0.5 * x**2)

    def grad_f(self, x):
        # Gradient: grad_f(x) = x^3 - x
        return x**3 - x

class DoubleWellDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975],
                        acceptance_rate=acceptance_rate)

    def quantile(self, p, dim=None):
        # No closed-form theoretical quantiles for double-well
        # Return NaN - some diagnostic methods will skip quantile-based checks
        return np.nan

# Run the sampler
print("Running Double-Well MALA sampler...")
mala = DoubleWellMALA(d=1, eta=1, n_steps=1_000_000, burn_in=10_000, seed=42)
samples_post = mala.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = DoubleWellDiagnostics(samples_post, acceptance_rate=mala.acceptance_rate)
diagnostics.print_comprehensive_stats()

# Visualizations
print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
diagnostics.plot_tail_exploration(dim=0)
