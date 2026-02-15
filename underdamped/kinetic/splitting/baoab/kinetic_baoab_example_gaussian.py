import numpy as np
from scipy.stats import norm
from underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics

class GaussianBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, mu=None, Sigma=None,
                  x0=None, v0 = None, seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
        self.mu = np.zeros(self.d) if mu is None else np.array(mu)
        self.Sigma = (0.5 * np.eye(self.d) + 0.5 * np.ones((self.d, self.d))) if Sigma is None else np.array(Sigma)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
    
    def f(self, x):
        diff = x - self.mu
        return 0.5 * diff @ self.Sigma_inv @ diff

    def grad_f(self, x):
        return self.Sigma_inv @ (x - self.mu)

class GaussianDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, d, mu=None, Sigma=None):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975])
        self.mu = np.zeros(d) if mu is None else np.array(mu)
        self.Sigma = (0.5 * np.eye(d) + 0.5) if Sigma is None else np.array(Sigma)
        self.std = np.sqrt(np.diag(self.Sigma))

    def quantile(self, p, dim=None):
        if dim is None:
            # Return all dimensions
            return self.mu + self.std * norm.ppf(p)
        else:
            return self.mu[dim] + self.std[dim] * norm.ppf(p)

kinetic = GaussianBAOAB(d=1, eta=1, gamma=1, n_steps=100000, burn_in=10000, seed=42)
samples_post = kinetic.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = GaussianDiagnostics(samples_post, d=1)
diagnostics.print_comprehensive_stats()

print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
diagnostics.plot_tail_exploration(dim=0)