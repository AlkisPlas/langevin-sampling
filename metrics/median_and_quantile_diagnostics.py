import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class DistributionDiagnostics(ABC):
    def __init__(self, samples_post, quantile_levels=None):
        """
        samples_post: np.array of shape (n_samples, d)
        quantile_levels: list of quantiles (0 < q < 1) to track
        """
        self.samples = samples_post
        self.n_samples, self.d = samples_post.shape
        self.quantile_levels = quantile_levels if quantile_levels else [0.1, 0.25, 0.5, 0.75, 0.9]
    
    @abstractmethod
    def quantile(self, p):
        """
        Return the theoretical quantile at probability p (0 < p < 1)
        Must be implemented in subclasses
        """
        pass
    
    # ------------------------------
    # Empirical diagnostics
    # ------------------------------
    def compute_empirical_stats(self):
        stats = []
        for dim in range(self.d):
            data = self.samples[:, dim]
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            emp_quantiles = np.percentile(data, [q*100 for q in self.quantile_levels])
            theor_quantiles = np.array([self.quantile(q) for q in self.quantile_levels])
            stats.append({
                "dim": dim,
                "median": median,
                "mad": mad,
                "empirical_quantiles": emp_quantiles,
                "theoretical_quantiles": theor_quantiles
            })
        return stats
    
    # ------------------------------
    # Plot median convergence over time
    # ------------------------------
    def plot_median_convergence(self, window_size=10000, y_min=None, y_max=None):
        n_windows = self.n_samples // window_size
        plt.figure(figsize=(12, 6))
        for dim in range(self.d):
            medians = []
            for i in range(n_windows):
                w = self.samples[i*window_size:(i+1)*window_size, dim]
                medians.append(np.median(w))
            plt.plot(medians, label=f"Dim {dim} median")
            # optional: theoretical median line
            plt.axhline(self.quantile(0.5, dim=dim), color='k', linestyle='--', label='Theoretical median' if dim==0 else "")
        plt.xlabel(f"Window index (size {window_size})")
        plt.ylabel("Median")
        plt.title("Convergence of median over time")
        if y_min is not None or y_max is not None:
            plt.ylim(y_min, y_max)
        plt.legend()
        plt.show()

    # ------------------------------
    # Print empirical stats
    # ------------------------------
    def print_stats(self):
        """
        Nicely print median, MAD, and quantiles for all dimensions
        """
        stats = self.compute_empirical_stats()
        for s in stats:
            print(f"\nDim {s['dim']}:")
            print(f"  Median = {s['median']:.3f}, MAD = {s['mad']:.3f}")
            print(f"  Empirical quantiles = {np.round(s['empirical_quantiles'], 3)}")
            print(f"  Theoretical quantiles = {np.round(s['theoretical_quantiles'], 3)}")
