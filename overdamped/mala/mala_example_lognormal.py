from scipy.stats import lognorm
import numpy as np
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from overdamped.mala.mala_runner import MALA

class LognormalMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, mu=None, sigma=None, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        # Parameters in log-space: log(X) ~ N(mu, diag(sigma^2))
        self.mu = np.zeros(self.d) if mu is None else np.array(mu)  # mean in log-space
        self.sigma = np.ones(self.d) if sigma is None else np.array(sigma)  # std dev in log-space

    def f(self, x):
        """
        Negative log-density (up to constant) for lognormal.
        For x > 0: f(x) = sum(log(x_i)) + 0.5 * sum((log(x_i) - mu_i)^2 / sigma_i^2)
        """
        # Avoid log of non-positive values
        if np.any(x <= 0):
            return np.inf

        log_x = np.log(x)
        return np.sum(log_x) + 0.5 * np.sum(((log_x - self.mu) / self.sigma)**2)

    def grad_f(self, x):
        """
        Gradient of f with respect to x.
        grad_f(x)_i = (1/x_i) * (1 + (log(x_i) - mu_i) / sigma_i^2)
        """
        # Avoid division by zero or log of non-positive values
        if np.any(x <= 0):
            return np.ones(self.d) * 1e10  # large gradient to push away from boundary

        log_x = np.log(x)
        return (1 / x) * (1 + (log_x - self.mu) / self.sigma**2)

class LognormalDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, d, mu=None, sigma=None, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975],
                        acceptance_rate=acceptance_rate)
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
print("Running Lognormal MALA sampler...")
mala = LognormalMALA(d=1, eta=0.1, n_steps=100000, burn_in=10000, x0=np.ones(1), seed=42)
samples_post = mala.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = LognormalDiagnostics(samples_post, d=1, acceptance_rate=mala.acceptance_rate)
diagnostics.print_comprehensive_stats()

# Visualizations
print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
diagnostics.plot_tail_exploration(dim=0)
diagnostics.plot_quantile_mae_over_time()
