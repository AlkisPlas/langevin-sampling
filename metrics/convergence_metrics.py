# convergence_metrics.py

import numpy as np
from scipy.stats import ks_1samp
from scipy.linalg import sqrtm


# -------------------------------------------------------
# Core metrics
# -------------------------------------------------------

def ks_distance(samples, cdf):
    """
    Kolmogorov–Smirnov distance between samples and target CDF.
    """
    return ks_1samp(samples, cdf).statistic


def quantile_error(samples, ppf, qs):
    """
    Absolute error between empirical and true quantiles.

    Parameters
    ----------
    qs : array-like
        Quantile levels, e.g. [0.9, 0.95, 0.99]
    """
    samples = np.asarray(samples)
    return {
        q: abs(np.quantile(samples, q) - ppf(q))
        for q in qs
    }


def tail_probability_error(samples, tail_prob_fn, Rs):
    """
    Error in tail probabilities P(|X| > R).

    tail_prob_fn(R) should return true P(|X| > R)
    """
    samples = np.asarray(samples)
    return {
        R: abs(
            np.mean(np.abs(samples) > R)
            - tail_prob_fn(R)
        )
        for R in Rs
    }


def truncated_wasserstein_1(samples, cdf, M=10.0, grid_size=5000):
    """
    Truncated Wasserstein-1 distance on [-M, M].

    Integral of |F_emp(x) - F_true(x)| dx over [-M, M].
    """
    samples = np.asarray(samples)
    xs = np.linspace(-M, M, grid_size)

    F_emp = np.array([
        np.mean(samples <= x) for x in xs
    ])
    F_true = np.array([
        cdf(x) for x in xs
    ])

    return np.trapz(np.abs(F_emp - F_true), xs)


def gaussian_w2(mu1, Sigma1, mu2, Sigma2):
    diff = mu1 - mu2
    return np.sqrt(np.sum(diff**2) + np.trace(Sigma1 + Sigma2 - 2*sqrtm(sqrtm(Sigma1) @ Sigma2 @ sqrtm(Sigma1))))


# -------------------------------------------------------
# Autocorrelation & ESS
# -------------------------------------------------------

def autocorrelation(samples, max_lag=None):
    """
    Compute autocorrelation function up to max_lag.

    Returns array rho[0], rho[1], ..., rho[max_lag]
    where rho[0] = 1.
    """
    x = np.asarray(samples)
    x = x - np.mean(x)
    n = len(x)

    if max_lag is None:
        max_lag = min(n // 2, 1000)

    var = np.dot(x, x) / n
    if var == 0:
        return np.ones(max_lag + 1)

    acf = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.dot(x[:n - k], x[k:]) / ((n - k) * var)

    return acf


def integrated_autocorrelation_time(samples, max_lag=None):
    """
    Estimate integrated autocorrelation time (IACT)
    using the initial positive sequence rule.
    """
    rho = autocorrelation(samples, max_lag)

    tau = 1.0
    for k in range(1, len(rho)):
        if rho[k] <= 0:
            break
        tau += 2 * rho[k]

    return tau


def effective_sample_size(samples, max_lag=None):
    """
    Effective Sample Size (ESS).
    """
    n = len(samples)
    tau = integrated_autocorrelation_time(samples, max_lag)
    return n / tau


# -------------------------------------------------------
# Composite report
# -------------------------------------------------------

def convergence_report(
    samples,
    *,
    cdf,
    ppf,
    tail_prob_fn=None,
    qs=(0.9, 0.95, 0.99),
    Rs=(5, 10, 20),
    wasserstein_M=10.0,
    ess_max_lag=None
):
    report = {}

    report["KS_distance"] = ks_distance(samples, cdf)

    report["quantile_error"] = quantile_error(
        samples, ppf, qs
    )

    if tail_prob_fn is not None:
        report["tail_probability_error"] = tail_probability_error(
            samples, tail_prob_fn, Rs
        )

    report["truncated_W1"] = truncated_wasserstein_1(
        samples, cdf, M=wasserstein_M
    )

    report["IACT"] = integrated_autocorrelation_time(
        samples, max_lag=ess_max_lag
    )

    report["ESS"] = effective_sample_size(
        samples, max_lag=ess_max_lag
    )

    return report
