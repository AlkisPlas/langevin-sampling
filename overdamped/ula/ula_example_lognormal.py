from scipy.stats import lognorm
import numpy as np
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from overdamped.ula.ula_runner import ULA

class LognormalULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, mu=None, sigma=None, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        # Parameters in log-space: log(X) ~ N(mu, diag(sigma^2))
        self.mu = np.zeros(self.d) if mu is None else np.array(mu)  # mean in log-space
        self.sigma = np.ones(self.d) if sigma is None else np.array(sigma)  # std dev in log-space

    def grad_f(self, x):
        """
        Gradient of negative log-density with respect to x.
        grad_f(x)_i = (1/x_i) * (1 + (log(x_i) - mu_i) / sigma_i^2)
        """
        # Avoid division by zero or log of non-positive values
        if np.any(x <= 0):
            return np.ones(self.d) * 1e10  # large gradient to push away from boundary

        log_x = np.log(x)
        return (1 / x) * (1 + (log_x - self.mu) / self.sigma**2)

class LognormalDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, d, mu=None, sigma=None):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975],
                        acceptance_rate=None)  # ULA has no acceptance step
        self.d = d
        self.mu = np.zeros(d) if mu is None else np.array(mu)  # mean in log-space
        self.sigma = np.ones(d) if sigma is None else np.array(sigma)  # std dev in log-space

    def quantile(self, p, dim=None):
        """
        Theoretical quantile for lognormal distribution.
        For LogNormal(mu, sigma), the quantile is exp(mu + sigma * Phi^{-1}(p))
        where Phi^{-1} is the inverse CDF of standard normal.
        """
        from scipy.stats import norm
        if dim is None:
            # Return quantiles for all dimensions
            return np.exp(self.mu + self.sigma * norm.ppf(p))
        else:
            return np.exp(self.mu[dim] + self.sigma[dim] * norm.ppf(p))

# Run the sampler
print("Running Lognormal ULA sampler...")
ula = LognormalULA(d=3, eta=0.01, n_steps=100000, burn_in=10000, x0=np.ones(3), seed=42)
samples_post = ula.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = LognormalDiagnostics(samples_post, d=3)
diagnostics.print_comprehensive_stats()

# Visualizations
print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0, 1])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
diagnostics.plot_tail_exploration(dim=0)
