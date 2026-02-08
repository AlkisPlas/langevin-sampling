import numpy as np
from scipy.stats import expon
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from overdamped.ula.ula_runner import ULA

class ExponentialULA(ULA):
    def __init__(self, d, eta, n_steps, burn_in, rate=None, x0=None, seed=None):
        """
        Exponential distribution sampler using ULA.

        Parameters:
        -----------
        rate : array-like or float or None
            Rate parameter λ (default: ones). Can be:
            - None: uses λ=1 for all dimensions
            - float: same rate for all dimensions
            - array: different rate per dimension

        Properties:
        -----------
        - Support: x > 0 (all components positive)
        - Mean: 1/λ
        - Variance: 1/λ²
        - Median: log(2)/λ ≈ 0.693/λ
        - Memoryless property
        """
        super().__init__(d, eta, n_steps, burn_in, x0, seed)

        # Handle rate parameter
        if rate is None:
            self.rate = np.ones(self.d)
        elif np.isscalar(rate):
            self.rate = np.ones(self.d) * rate
        else:
            self.rate = np.array(rate)

        # Default initial point in valid region if not provided
        if x0 is None:
            self.x0 = 1.0 / self.rate  # Start at the mean

    def grad_f(self, x):
        """
        Gradient of negative log-density with respect to x.

        ∇f(x) = λ (constant gradient!)

        For boundary handling, returns large gradient if x ≤ 0.
        """
        if np.any(x <= 0):
            return np.ones(self.d) * 1e10  # Large gradient to push away from boundary

        return self.rate

class ExponentialDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, d, rate=None):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975],
                        acceptance_rate=None)  # ULA has no acceptance step

        # Handle rate parameter
        if rate is None:
            self.rate = np.ones(d)
        elif np.isscalar(rate):
            self.rate = np.ones(d) * rate
        else:
            self.rate = np.array(rate)

    def quantile(self, p, dim=None):
        """
        Theoretical quantile for exponential distribution.

        For Exp(λ), quantile(p) = -log(1-p)/λ
        """
        if dim is None:
            # Return quantiles for all dimensions
            return -np.log(1 - p) / self.rate
        else:
            return -np.log(1 - p) / self.rate[dim]

# Run the sampler
print("Running Exponential ULA sampler...")
print("Distribution: Exp(λ=1) for each dimension")
print("Properties: Mean=1, Variance=1, Median≈0.693\n")

# Use rate=1 (standard exponential) and start at mean
ula = ExponentialULA(
    d=3,
    eta=0.1,
    n_steps=100000,
    burn_in=10000,
    rate=1.0,
    x0=np.ones(3),  # Start at mean
    seed=42
)
samples_post = ula.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = ExponentialDiagnostics(samples_post, d=3, rate=1.0)
diagnostics.print_comprehensive_stats()

# Visualizations
print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0, 1])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
diagnostics.plot_tail_exploration(dim=0)
