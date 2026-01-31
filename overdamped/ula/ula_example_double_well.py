import numpy as np
from metrics.median_and_quantile_diagnostics import DistributionDiagnostics
from overdamped.ula.ula_runner import ULA

class DoubleWellULA(ULA):
    def grad_f(self, x):
        # f(x) = sum_i (1/4 x_i^4 - 1/2 x_i^2)
        return x**3 - x

class DoubleWellDiagnostics(DistributionDiagnostics):
    def quantile(self, p, dim=None):
        # No closed-form theoretical quantiles
        return np.nan
    
ula = DoubleWellULA(d=1, eta=0.01, n_steps=1_000_000, burn_in=10_000, seed=42)
samples_post = ula.run()

diagnostics = DoubleWellDiagnostics(samples_post)
diagnostics.print_stats()
diagnostics.plot_median_convergence(window_size=10000, y_min=-3, y_max=3)