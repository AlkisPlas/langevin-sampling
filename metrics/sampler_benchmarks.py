import numpy as np

from .convergence_metrics import (
    ks_distance,
    quantile_error,
    tail_probability_error,
    truncated_wasserstein_1,
    integrated_autocorrelation_time,
    effective_sample_size,
)


class SamplerBenchmark:
    """
    Sampler-agnostic benchmarking utility for MCMC / Langevin samplers.

    Parameters
    ----------
    cdf : callable
        Target CDF
    ppf : callable
        Target quantile function (inverse CDF)
    tail_prob_fn : callable, optional
        Function R -> P(|X| > R)
    """

    def __init__(
        self,
        *,
        cdf,
        ppf,
        tail_prob_fn=None,
        qs=(0.9, 0.95, 0.99),
        Rs=(5, 10, 20),
        wasserstein_M=10.0,
        ess_max_lag=None,
    ):
        self.cdf = cdf
        self.ppf = ppf
        self.tail_prob_fn = tail_prob_fn

        self.qs = qs
        self.Rs = Rs
        self.wasserstein_M = wasserstein_M
        self.ess_max_lag = ess_max_lag

        self.results = {}

    # --------------------------------------------------
    # Core API
    # --------------------------------------------------

    def evaluate(self, samples, name):
        """
        Evaluate convergence diagnostics for one sampler.

        Parameters
        ----------
        samples : array-like
            Samples from the sampler (after burn-in)
        name : str
            Sampler name (e.g. 'ULA', 'MALA', 'Kinetic')
        """
        samples = np.asarray(samples)

        report = {}

        report["KS_distance"] = ks_distance(samples, self.cdf)

        report["quantile_error"] = quantile_error(
            samples, self.ppf, self.qs
        )

        if self.tail_prob_fn is not None:
            report["tail_probability_error"] = tail_probability_error(
                samples, self.tail_prob_fn, self.Rs
            )

        report["truncated_W1"] = truncated_wasserstein_1(
            samples, self.cdf, M=self.wasserstein_M
        )

        report["IACT"] = integrated_autocorrelation_time(
            samples, max_lag=self.ess_max_lag
        )

        report["ESS"] = effective_sample_size(
            samples, max_lag=self.ess_max_lag
        )

        report["num_samples"] = len(samples)

        self.results[name] = report
        return report

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def summary(self):
        """
        Return full benchmark results.
        """
        return self.results

    def pretty_print(self):
        """
        Human-readable summary.
        """
        for name, report in self.results.items():
            print(f"\n=== {name} ===")
            for key, value in report.items():
                print(f"{key}: {value}")

    def metric_table(self, metric):
        """
        Extract a single metric across samplers.

        Example:
        benchmark.metric_table("ESS")
        """
        return {
            name: report.get(metric)
            for name, report in self.results.items()
        }
