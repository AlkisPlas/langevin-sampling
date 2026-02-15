import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class ComprehensiveDiagnostics(ABC):
    def __init__(self, samples_post, quantile_levels=None, acceptance_rate=None):
        """
        Comprehensive MCMC diagnostics for Langevin samplers

        Parameters:
        -----------
        samples_post : np.array of shape (n_samples, d)
            Post-burn-in samples
        quantile_levels : list
            Quantile levels to track (default: [0.025, 0.5, 0.975])
        acceptance_rate : float or None
            Acceptance rate for MALA (None for ULA)
        """
        self.samples = samples_post
        self.n_samples, self.d = samples_post.shape
        self.quantile_levels = quantile_levels if quantile_levels else [0.025, 0.5, 0.975]
        self.acceptance_rate = acceptance_rate

    @abstractmethod
    def quantile(self, p, dim=None):
        """
        Return the theoretical quantile at probability p (0 < p < 1)
        Must be implemented in subclasses
        """
        pass

    # ========================================
    # ESS (Effective Sample Size)
    # ========================================
    def compute_autocorrelation(self, x, max_lag=None):
        """
        Compute autocorrelation for a 1D time series up to max_lag.
        Uses FFT for efficiency.
        """
        if max_lag is None:
            max_lag = min(len(x) // 2, 1000)

        x = x - np.mean(x)
        #autocorr = np.correlate(x, x, mode='full')
        #autocorr = autocorr[len(autocorr)//2:]
        n = len(x)
        fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fft_x = np.fft.fft(x, n=fft_size)
        autocorr = np.fft.ifft(fft_x * np.conj(fft_x)).real[:max_lag]
        autocorr = autocorr / autocorr[0]

        return autocorr[:max_lag]

    def compute_ess(self, dim=None):
        """
        Compute Effective Sample Size using autocorrelation.
        ESS = n / (1 + 2 * sum(rho_k)) where rho_k is autocorrelation at lag k
        """
        if dim is None:
            # Compute for all dimensions
            return [self.compute_ess(d) for d in range(self.d)]

        x = self.samples[:, dim]
        n = len(x)

        # Compute autocorrelation
        max_lag = min(n // 2, 500)
        acf = self.compute_autocorrelation(x, max_lag=max_lag)

        # Sum autocorrelations until they become negative or very small
        # (initial positive sequence estimator)
        tau = 1.0  # Start with 1 for rho_0 = 1
        for k in range(1, len(acf)):
            if acf[k] < 0.05:  # Stop when autocorrelation is small
                break
            tau += 2 * acf[k]

        ess = n / tau
        return ess

    # ========================================
    # Autocorrelation Analysis
    # ========================================
    def plot_autocorrelation(self, max_lag=100, dims=None):
        """
        Plot autocorrelation functions for specified dimensions.
        """
        if dims is None:
            dims = range(min(self.d, 3))  # Plot first 3 dimensions by default

        fig, axes = plt.subplots(1, len(dims), figsize=(6*len(dims), 4))
        if len(dims) == 1:
            axes = [axes]

        for idx, dim in enumerate(dims):
            x = self.samples[:, dim]
            acf = self.compute_autocorrelation(x, max_lag=max_lag)

            axes[idx].plot(acf, linewidth=2)
            axes[idx].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[idx].axhline(0.05, color='r', linestyle='--', alpha=0.3, label='0.05 threshold')
            axes[idx].set_xlabel('Lag')
            axes[idx].set_ylabel('Autocorrelation')
            axes[idx].set_title(f'ACF - Dimension {dim}')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ========================================
    # Divergence Metrics
    # ========================================
    def compute_divergence_rate(self, threshold=1e6):
        """
        Compute the rate of divergent samples (samples exceeding threshold).
        """
        n_divergent = np.sum(np.any(np.abs(self.samples) > threshold, axis=1))
        return n_divergent / self.n_samples

    def compute_quantile_divergence(self):
        """
        Compute divergence between empirical and theoretical quantiles.
        Returns mean absolute error across all dimensions and quantiles.
        Returns None if theoretical quantiles are not available (NaN).
        """
        divergences = []
        for dim in range(self.d):
            data = self.samples[:, dim]
            emp_quantiles = np.percentile(data, [q*100 for q in self.quantile_levels])
            theor_quantiles = np.array([self.quantile(q, dim=dim) for q in self.quantile_levels])

            # Check if theoretical quantiles are available
            if np.any(np.isnan(theor_quantiles)):
                return None

            mae = np.mean(np.abs(emp_quantiles - theor_quantiles))
            divergences.append(mae)
        return np.array(divergences)

    # ========================================
    # Tail Exploration
    # ========================================
    def compute_tail_coverage(self, tail_quantiles=[0.01, 0.05, 0.95, 0.99]):
        """
        Measure how well the sampler explores the tails.
        Returns the percentage of samples that reach various tail quantiles.
        Returns None if theoretical quantiles are not available (NaN).
        """
        # Check if theoretical quantiles are available
        test_quantile = self.quantile(0.5, dim=0)
        if np.isnan(test_quantile):
            return None

        tail_stats = {}
        for dim in range(self.d):
            data = self.samples[:, dim]
            stats = {}
            for q in tail_quantiles:
                theor_q = self.quantile(q, dim=dim)
                if q < 0.5:
                    # Lower tail: fraction of samples below theoretical quantile
                    coverage = np.mean(data <= theor_q)
                else:
                    # Upper tail: fraction of samples above theoretical quantile
                    coverage = np.mean(data >= theor_q)
                stats[f'q_{q}'] = {
                    'theoretical': theor_q,
                    'coverage': coverage,
                    'expected': q if q < 0.5 else (1 - q)
                }
            tail_stats[f'dim_{dim}'] = stats
        return tail_stats

    def plot_tail_exploration(self, dim=0):
        """
        Visualize tail exploration by plotting histogram vs theoretical density.
        If theoretical quantiles are not available, plots histogram only.
        """
        data = self.samples[:, dim]

        # Check if theoretical quantiles are available
        test_quantile = self.quantile(0.5, dim=dim)
        has_theoretical = not np.isnan(test_quantile)

        if has_theoretical:
            plt.figure(figsize=(12, 5))

            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')

            # Mark theoretical quantiles
            for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
                theor_q = self.quantile(q, dim=dim)
                plt.axvline(theor_q, color='r', linestyle='--', alpha=0.5, linewidth=1)

            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Histogram with Theoretical Quantiles - Dim {dim}')
            plt.grid(alpha=0.3)

            # Q-Q plot
            plt.subplot(1, 2, 2)
            sample_quantiles = np.percentile(data, np.linspace(0.5, 99.5, 100))
            theoretical_quantiles = [self.quantile(q/100, dim=dim) for q in np.linspace(0.5, 99.5, 100)]

            plt.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=20)

            # Add diagonal line
            min_val = min(np.min(theoretical_quantiles), np.min(sample_quantiles))
            max_val = max(np.max(theoretical_quantiles), np.max(sample_quantiles))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')

            plt.xlabel('Theoretical Quantiles')
            plt.ylabel('Sample Quantiles')
            plt.title(f'Q-Q Plot - Dim {dim}')
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
        else:
            # No theoretical quantiles - plot histogram only
            plt.figure(figsize=(10, 5))
            plt.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Empirical Distribution - Dim {dim}')
            plt.grid(alpha=0.3)
            plt.show()

    # ========================================
    # Comprehensive Summary
    # ========================================
    def print_comprehensive_stats(self):
        """
        Print all diagnostic statistics in a comprehensive report.
        """
        print("=" * 70)
        print("COMPREHENSIVE MCMC DIAGNOSTICS")
        print("=" * 70)

        # Basic info
        print(f"\nSample size: {self.n_samples}")
        print(f"Dimensions: {self.d}")

        # Acceptance rate (for MALA)
        if self.acceptance_rate is not None:
            print(f"\n--- ACCEPTANCE RATE ---")
            print(f"Acceptance rate: {self.acceptance_rate:.3f}")
            if self.acceptance_rate < 0.2:
                print("  ⚠ Warning: Low acceptance rate (< 0.2). Consider reducing step size.")
            elif self.acceptance_rate > 0.8:
                print("  ⚠ Warning: High acceptance rate (> 0.8). Consider increasing step size.")

        # ESS
        print(f"\n--- EFFECTIVE SAMPLE SIZE (ESS) ---")
        ess_values = self.compute_ess()
        for dim in range(self.d):
            ess = ess_values[dim]
            ess_ratio = ess / self.n_samples
            print(f"Dim {dim}: ESS = {ess:.1f} ({ess_ratio:.2%} of total samples)")
            if ess_ratio < 0.1:
                print(f"  ⚠ Warning: Low ESS ratio (< 10%). High autocorrelation detected.")

        # Divergence
        print(f"\n--- DIVERGENCE METRICS ---")
        div_rate = self.compute_divergence_rate()
        print(f"Divergent samples rate: {div_rate:.4f}")
        if div_rate > 0.01:
            print(f"  ⚠ Warning: High divergence rate (> 1%). Check step size and initialization.")

        quantile_div = self.compute_quantile_divergence()
        if quantile_div is not None:
            print(f"\nQuantile MAE (Mean Absolute Error):")
            for dim in range(self.d):
                print(f"  Dim {dim}: {quantile_div[dim]:.4f}")
        else:
            print(f"\nQuantile MAE: N/A (no theoretical quantiles available)")

        # Tail coverage
        print(f"\n--- TAIL EXPLORATION ---")
        tail_stats = self.compute_tail_coverage()
        if tail_stats is not None:
            for dim in range(self.d):
                print(f"\nDim {dim}:")
                stats = tail_stats[f'dim_{dim}']
                for key, val in stats.items():
                    coverage_ratio = val['coverage'] / val['expected']
                    status = "✓" if 0.8 <= coverage_ratio <= 1.2 else "✗"
                    print(f"  {status} {key}: coverage={val['coverage']:.3f}, expected={val['expected']:.3f}")
        else:
            print(f"Tail coverage: N/A (no theoretical quantiles available)")

        # Basic statistics
        print(f"\n--- BASIC STATISTICS ---")
        for dim in range(self.d):
            data = self.samples[:, dim]
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            mean = np.mean(data)
            std = np.std(data)
            theor_median = self.quantile(0.5, dim=dim)
            print(f"\nDim {dim}:")
            print(f"  Mean: {mean:.3f}")
            print(f"  Empirical median: {median:.3f}")
            if not np.isnan(theor_median):
                print(f"  Theoretical median: {theor_median:.3f}")
            print(f"  Std dev: {std:.3f}")
            print(f"  MAD: {mad:.3f}")

        print("\n" + "=" * 70)

    def plot_trace(self, dims=None, max_samples=50000):
        """
        Plot trace plots for specified dimensions.
        """
        if dims is None:
            dims = range(min(self.d, 3))

        n_plot = min(self.n_samples, max_samples)
        samples_to_plot = self.samples[:n_plot, :]

        fig, axes = plt.subplots(len(dims), 1, figsize=(12, 3*len(dims)))
        if len(dims) == 1:
            axes = [axes]

        for idx, dim in enumerate(dims):
            axes[idx].plot(samples_to_plot[:, dim], linewidth=0.5, alpha=0.7)
            axes[idx].axhline(self.quantile(0.5, dim=dim), color='r',
                            linestyle='--', label='Theoretical median')
            axes[idx].set_xlabel('Iteration')
            axes[idx].set_ylabel(f'Dim {dim}')
            axes[idx].set_title(f'Trace Plot - Dimension {dim}')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
