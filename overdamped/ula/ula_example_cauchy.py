import numpy as np
from metrics.median_and_quantile_diagnostics import DistributionDiagnostics
from overdamped.ula.ula_runner import ULA

class CauchyULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)

    def grad_f(self, x):
        return (self.d + 1) * x / (1 + np.sum(x**2))

class CauchyDiagnostics(DistributionDiagnostics):
    def quantile(self, p, dim=None):
        return np.tan(np.pi * (p - 0.5))
    
ula = CauchyULA(d=1, eta=0.01, n_steps=100000, burn_in=10000, seed=42)
samples_post = ula.run()

diagnostics = CauchyDiagnostics(samples_post)
diagnostics.print_stats()
diagnostics.plot_median_convergence(window_size=1000, y_min=-10, y_max=10)