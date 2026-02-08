import numpy as np
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics
from overdamped.mala.mala_runner import MALA

class CauchyMALA(MALA):
    def __init__(self, d, eta, n_steps, burn_in, location=None, scale=1.0, x0=None, seed=None):
        super().__init__(d, eta, n_steps, burn_in, x0, seed)
        self.location = np.zeros(self.d) if location is None else np.array(location)
        self.scale = scale

    def f(self, x):
        diff = x - self.location
        return ((self.d + 1) / 2) * np.log(self.scale**2 + np.sum(diff**2))

    def grad_f(self, x):
        diff = x - self.location
        return (self.d + 1) * diff / (self.scale**2 + np.sum(diff**2))

class CauchyDiagnostics(ComprehensiveDiagnostics):
    def __init__(self, samples_post, location=None, scale=1.0, acceptance_rate=None):
        super().__init__(samples_post, quantile_levels=[0.025, 0.5, 0.975],
                        acceptance_rate=acceptance_rate)
        n_samples, d = samples_post.shape
        self.location = np.zeros(d) if location is None else np.array(location)
        self.scale = scale

    def quantile(self, p, dim=None):
        # Standard Cauchy quantile: tan(π(p - 0.5))
        # With location and scale: location + scale * tan(π(p - 0.5))
        q = np.tan(np.pi * (p - 0.5))
        if dim is None:
            return self.location + self.scale * q
        else:
            return self.location[dim] + self.scale * q
    
print("Running Cauchy MALA sampler...")
mala = CauchyMALA(d=1, eta=0.01, n_steps=100000, burn_in=10000, seed=42)
samples_post = mala.run()

print("\nGenerating comprehensive diagnostics...\n")
diagnostics = CauchyDiagnostics(samples_post, acceptance_rate=mala.acceptance_rate)
diagnostics.print_comprehensive_stats()

# Visualizations
print("\nGenerating visualizations...")
diagnostics.plot_trace(dims=[0])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0])
diagnostics.plot_tail_exploration(dim=0)