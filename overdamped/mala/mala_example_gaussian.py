from scipy.stats import norm
import numpy as np
from metrics.median_and_quantile_diagnostics import DistributionDiagnostics
from overdamped.mala.mala_runner import MALA

class GaussianMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.mu = np.zeros(self.d)
        self.Sigma = 0.5 * np.eye(self.d) + 0.5 * np.ones((self.d, self.d))
        self.Sigma_inv = np.linalg.inv(self.Sigma)

    def f(self, x):
        diff = x - self.mu
        return 0.5 * diff @ self.Sigma_inv @ diff

    def grad_f(self, x):
        return self.Sigma_inv @ (x - self.mu)

class GaussianDiagnostics(DistributionDiagnostics):
    def __init__(self, samples_post, d):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975])
        self.mu = np.zeros(d)
        self.Sigma = 0.5 * np.eye(d) + 0.5
        self.std = np.sqrt(np.diag(self.Sigma))

    def quantile(self, p, dim=None):
        if dim is None:
            # Return all dimensions
            return self.mu + self.std * norm.ppf(p)
        else:
            return self.mu[dim] + self.std[dim] * norm.ppf(p)

    
mala = GaussianMALA(d=3, eta=0.01, n_steps=100000, burn_in=10000, seed=42)
samples_post = mala.run()

diagnostics = GaussianDiagnostics(samples_post, d=3)
diagnostics.print_stats()
diagnostics.plot_median_convergence(window_size=1000, y_min=-10, y_max=10)