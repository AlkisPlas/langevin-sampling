import numpy as np
from scipy.stats import t as student_t
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from overdamped.mala.mala_runner import MALA

class StudentTMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, nu=3.0, location=None, scale=None, x0=None, seed=None):
        """
        Student's t-distribution sampler using MALA.

        Parameters:
        -----------
        nu : float
            Degrees of freedom (nu > 0). Smaller values = heavier tails.
            nu=1: Cauchy, nu=∞: Gaussian
        location : array-like or None
            Location parameter (default: zeros)
        scale : array-like or None
            Scale matrix (default: identity). For simplicity, using diagonal scale.
        """
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.nu = nu
        self.location = np.zeros(self.d) if location is None else np.array(location)

        # For simplicity, using diagonal scale matrix
        if scale is None:
            self.scale = np.ones(self.d)
        else:
            self.scale = np.array(scale) if np.isscalar(scale) or len(np.array(scale).shape) == 0 else np.array(scale)
            if self.scale.shape == ():
                self.scale = np.ones(self.d) * self.scale

        self.scale_sq = self.scale ** 2

    def f(self, x):
        """
        Negative log-density (up to constant) for Student's t-distribution.

        For multivariate t with diagonal scale:
        f(x) = ((nu + d)/2) * log(1 + (1/nu) * sum((x_i - mu_i)^2 / scale_i^2))
        """
        diff = (x - self.location) / self.scale
        mahalanobis_sq = np.sum(diff ** 2)
        return ((self.nu + self.d) / 2) * np.log(1 + mahalanobis_sq / self.nu)

    def grad_f(self, x):
        """
        Gradient of f with respect to x.

        grad_f(x) = ((nu + d) / nu) * (x - mu) / scale^2 / (1 + ||standardized(x - mu)||^2 / nu)
        """
        diff = x - self.location
        standardized = diff / self.scale
        mahalanobis_sq = np.sum(standardized ** 2)

        denominator = 1 + mahalanobis_sq / self.nu
        return ((self.nu + self.d) / self.nu) * (diff / self.scale_sq) / denominator

class StudentTDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, d, nu=3.0, location=None, scale=None, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975],
                        acceptance_rate=acceptance_rate)
        self.nu = nu
        self.location = np.zeros(d) if location is None else np.array(location)

        if scale is None:
            self.scale = np.ones(d)
        else:
            self.scale = np.array(scale) if np.isscalar(scale) or len(np.array(scale).shape) == 0 else np.array(scale)
            if self.scale.shape == ():
                self.scale = np.ones(d) * self.scale

    def quantile(self, p, dim=None):
        """
        Theoretical quantile for Student's t-distribution.

        For t(nu, location, scale):
        quantile(p) = location + scale * t_nu^{-1}(p)
        where t_nu^{-1} is the inverse CDF of standard Student's t with nu df.
        """
        # Standard t quantile
        t_quantile = student_t.ppf(p, df=self.nu)

        if dim is None:
            # Return quantiles for all dimensions
            return self.location + self.scale * t_quantile
        else:
            return self.location[dim] + self.scale[dim] * t_quantile

# Run the sampler
print("Running Student's t MALA sampler...")
print("Distribution: t(nu=3, location=0, scale=1)")
print("Note: nu=3 gives moderate heavy tails (variance exists but 4th moment doesn't)\n")

mala = StudentTMALA(d=3, eta=0.7, n_steps=100000, burn_in=10000, nu=3.0, seed=42)
samples_post = mala.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = StudentTDiagnostics(samples_post, d=3, nu=3.0, acceptance_rate=mala.acceptance_rate)
diagnostics.print_comprehensive_stats()

# Visualizations
print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0, 1])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
diagnostics.plot_tail_exploration(dim=0)
