import numpy as np
from scipy.stats import t as student_t
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from underdamped.kinetic.splitting.baoab.kinetic_baoab_runner import BAOAB

class StudentTBAOAB(BAOAB):
    def __init__(self, d, eta, gamma, n_steps, burn_in, nu, location=None, scale=None, x0=None, v0=None,seed=None):
        super().__init__(d, eta, gamma, n_steps, burn_in, x0, v0, seed)
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
        diff = (x - self.location) / self.scale
        mahalanobis_sq = np.sum(diff ** 2)
        return ((self.nu + self.d) / 2) * np.log(1 + mahalanobis_sq / self.nu)


    def grad_f(self, x):
        diff = x - self.location
        standardized = diff / self.scale
        mahalanobis_sq = np.sum(standardized ** 2)

        denominator = 1 + mahalanobis_sq / self.nu
        return ((self.nu + self.d) / self.nu) * (diff / self.scale_sq) / denominator

class StudentTDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, d, nu=3.0, location=None, scale=None):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975],
                        acceptance_rate=None)
        self.nu = nu
        self.location = np.zeros(d) if location is None else np.array(location)

        if scale is None:
            self.scale = np.ones(d)
        else:
            self.scale = np.array(scale) if np.isscalar(scale) or len(np.array(scale).shape) == 0 else np.array(scale)
            if self.scale.shape == ():
                self.scale = np.ones(d) * self.scale

    def quantile(self, p, dim=None):
        # Standard t quantile
        t_quantile = student_t.ppf(p, df=self.nu)

        if dim is None:
            # Return quantiles for all dimensions
            return self.location + self.scale * t_quantile
        else:
            return self.location[dim] + self.scale[dim] * t_quantile

# Run the sampler
kinetic = StudentTBAOAB(d=1, eta=0.6, gamma=1.5, n_steps=100000, burn_in=10000, nu=5, seed=42)
samples_post = kinetic.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = StudentTDiagnostics(samples_post, d=1, nu=5)
diagnostics.print_comprehensive_stats()

# Visualizations
print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
diagnostics.plot_tail_exploration(dim=0)
diagnostics.plot_quantile_mae_over_time()