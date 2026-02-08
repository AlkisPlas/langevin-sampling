# Comprehensive MCMC Diagnostics

This module provides comprehensive diagnostics for evaluating Langevin sampling algorithms (MALA and ULA).

## Features

### 1. **Effective Sample Size (ESS)**
Measures the number of "independent" samples in your chain, accounting for autocorrelation.

**Formula:** `ESS = n / (1 + 2 * Σ ρ_k)` where ρ_k is the autocorrelation at lag k

**Interpretation:**
- ESS close to n: Low autocorrelation, efficient sampling
- ESS << n: High autocorrelation, inefficient sampling
- Target: ESS/n > 0.1 (at least 10% efficiency)

### 2. **Autocorrelation Function (ACF)**
Shows how correlated samples are with themselves at various time lags.

**Interpretation:**
- Fast decay to zero: Good mixing
- Slow decay: High autocorrelation, consider thinning or adjusting step size
- Visualized with `plot_autocorrelation()`

### 3. **Acceptance Rate** (MALA only)
Percentage of proposals accepted by the Metropolis-Hastings correction.

**Optimal range:** 0.2 - 0.8 (depends on dimension)
- Too low (< 0.2): Step size too large, reduce `eta`
- Too high (> 0.8): Step size too small, increase `eta`
- ULA has no acceptance step (always accepts)

### 4. **Divergence Metrics**

#### a) Divergent Samples Rate
Fraction of samples with extreme values (|x| > threshold)

**Interpretation:**
- < 1%: Good
- > 1%: Numerical instability, check step size and initialization

#### b) Quantile Divergence
Mean absolute error between empirical and theoretical quantiles.

**Interpretation:**
- Measures accuracy of sampling
- Lower is better
- Per-dimension metric

### 5. **Tail Exploration**

#### a) Tail Coverage
Measures how well the sampler reaches extreme quantiles (1%, 5%, 95%, 99%)

**Interpretation:**
- Coverage ≈ Expected: Good tail exploration
- Coverage < Expected: Under-exploring tails
- Coverage > Expected: Over-exploring tails

#### b) Q-Q Plot
Visual comparison of sample vs theoretical quantiles

**Interpretation:**
- Points on diagonal: Perfect match
- Deviations indicate systematic bias

## Usage

### Basic Usage

```python
from metrics.comprehensive_diagnostics import ComprehensiveDiagnostics

# For MALA (with acceptance rate)
diagnostics = GaussianDiagnostics(
    samples_post,
    d=3,
    acceptance_rate=mala.acceptance_rate
)

# For ULA (no acceptance rate)
diagnostics = GaussianDiagnostics(
    samples_post,
    d=3,
    acceptance_rate=None
)

# Print comprehensive report
diagnostics.print_comprehensive_stats()

# Generate visualizations
diagnostics.plot_trace(dims=[0, 1])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
diagnostics.plot_tail_exploration(dim=0)
```

### Available Methods

| Method | Description |
|--------|-------------|
| `print_comprehensive_stats()` | Print full diagnostic report |
| `compute_ess(dim)` | Compute ESS for dimension(s) |
| `compute_autocorrelation(x, max_lag)` | Compute ACF |
| `compute_divergence_rate()` | Compute divergent samples rate |
| `compute_quantile_divergence()` | Compute quantile MAE |
| `compute_tail_coverage()` | Measure tail exploration |
| `plot_trace(dims)` | Plot trace plots |
| `plot_autocorrelation(max_lag, dims)` | Plot ACF |
| `plot_tail_exploration(dim)` | Plot histogram + Q-Q plot |

## Example Output

```
======================================================================
COMPREHENSIVE MCMC DIAGNOSTICS
======================================================================

Sample size: 90000
Dimensions: 3

--- ACCEPTANCE RATE ---
Acceptance rate: 0.574

--- EFFECTIVE SAMPLE SIZE (ESS) ---
Dim 0: ESS = 8234.5 (9.15% of total samples)
Dim 1: ESS = 8123.2 (9.03% of total samples)
Dim 2: ESS = 8345.7 (9.27% of total samples)

--- DIVERGENCE METRICS ---
Divergent samples rate: 0.0000

Quantile MAE (Mean Absolute Error):
  Dim 0: 0.0124
  Dim 1: 0.0131
  Dim 2: 0.0118

--- TAIL EXPLORATION ---

Dim 0:
  ✓ q_0.01: coverage=0.011, expected=0.010
  ✓ q_0.05: coverage=0.052, expected=0.050
  ✓ q_0.95: coverage=0.049, expected=0.050
  ✓ q_0.99: coverage=0.010, expected=0.010
```

## Recommendations

### For MALA:
1. **Acceptance rate 0.2-0.8**: Good
2. **ESS/n > 0.1**: Adequate efficiency
3. **Quantile MAE < 0.1**: Accurate sampling
4. **Tail coverage within 20% of expected**: Good exploration

### For ULA:
1. **No acceptance rate** (always accepts)
2. **May have discretization bias** (check quantile divergence)
3. **Generally lower ESS than MALA** at same step size
4. **Faster per iteration** (no acceptance computation)

## Files

- `comprehensive_diagnostics.py` - Main diagnostics class
- `median_and_quantile_diagnostics.py` - Legacy simple diagnostics
- `mala_example_gaussian_comprehensive.py` - MALA example with full diagnostics
- `ula_example_gaussian_comprehensive.py` - ULA example with full diagnostics
