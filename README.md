# Langevin Sampling Algorithms

A comprehensive, production-ready implementation of **Overdamped Langevin Sampling** algorithms (MALA and ULA) with extensive diagnostics for MCMC analysis.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-required-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-required-orange.svg)](https://scipy.org/)

---

## 📊 Supported Distributions

This implementation provides **6 distributions** covering a wide range of sampling challenges:

| Distribution | Support | Tail Behavior | Modality | Difficulty | Use Case |
|--------------|---------|---------------|----------|------------|----------|
| **Gaussian** | ℝᵈ | Light (exponential) | Unimodal | ⭐ Easy | Baseline, testing |
| **Exponential** | ℝ₊ᵈ | Light (exponential) | Unimodal | ⭐ Easy | Waiting times, lifetimes |
| **Student-t** | ℝᵈ | Tunable (polynomial) | Unimodal | ⭐⭐ Moderate | Robust statistics, finance |
| **Cauchy** | ℝᵈ | Very Heavy | Unimodal | ⭐⭐⭐ Hard | Heavy-tail modeling |
| **Lognormal** | ℝ₊ᵈ | Light (exponential) | Unimodal | ⭐⭐ Moderate | Positive-valued data |
| **Double-Well** | ℝᵈ | Custom | Bimodal | ⭐⭐⭐ Hard | Multimodal exploration |

### Distribution Details

#### 🔹 **Gaussian** `N(μ, Σ)`
- **Density:** `p(x) ∝ exp(-½(x-μ)ᵀΣ⁻¹(x-μ))`
- **Parameters:** Mean `μ`, covariance `Σ`
- **Properties:** Well-behaved, fast mixing, theoretical baseline
- **Default:** μ=0, Σ=0.5I + 0.5·ones (correlated structure)

#### 🔹 **Exponential** `Exp(λ)`
- **Density:** `p(x) = ∏ᵢ λᵢ exp(-λᵢxᵢ)` for x > 0
- **Parameters:** Rate `λ` (inverse of scale)
- **Properties:** Memoryless, positive support, constant gradient
- **Default:** λ=1 (mean=1, variance=1)
- **Mean:** 1/λ, **Variance:** 1/λ², **Median:** log(2)/λ ≈ 0.693/λ

#### 🔹 **Student's t** `t(ν, μ, σ)`
- **Density:** `p(x) ∝ (1 + ||standardized(x-μ)||²/ν)^(-(ν+d)/2)`
- **Parameters:** Degrees of freedom `ν`, location `μ`, scale `σ`
- **Properties:** Interpolates between Cauchy (ν=1) and Gaussian (ν→∞)
- **Default:** ν=3 (moderate heavy tails), μ=0, σ=1
- **Variance:** Exists for ν>2, equals ν/(ν-2)

#### 🔹 **Cauchy** `Cauchy(location, scale)`
- **Density:** `p(x) ∝ (scale² + ||x-location||²)^(-(d+1)/2)`
- **Parameters:** Location, scale
- **Properties:** No finite moments, very heavy tails, challenging to sample
- **Default:** location=0, scale=1
- **Note:** Special case of Student-t with ν=1

#### 🔹 **Lognormal** `LogNormal(μ, σ)`
- **Density:** `p(x) ∝ (1/∏xᵢ) exp(-½Σ((log(xᵢ)-μᵢ)/σᵢ)²)` for x>0
- **Parameters:** Log-space mean `μ`, log-space std `σ`
- **Properties:** Positive support, skewed, multiplicative processes
- **Default:** μ=0, σ=1 (median=1)
- **Median:** exp(μ)

#### 🔹 **Double-Well** `f(x) = ¼x⁴ - ½x²`
- **Potential:** Creates two wells at x ≈ ±1
- **Properties:** Bimodal, tests mode-switching capability
- **Challenges:** Requires sampler to transition between modes
- **Note:** No closed-form theoretical quantiles

---

## 🎯 Features

### Algorithms Implemented

#### **MALA** (Metropolis-Adjusted Langevin Algorithm)
- ✅ Exact sampling in the limit η→0
- ✅ Metropolis-Hastings correction ensures detailed balance
- ✅ Tracks acceptance rate
- ✅ Better for heavy-tailed distributions
- 📘 **Update:** `x' = x - η∇f(x) + √(2η)ξ`, then accept/reject

#### **ULA** (Unadjusted Langevin Algorithm)
- ✅ Faster per iteration (no acceptance step)
- ✅ Simple Euler-Maruyama discretization
- ✅ O(η) discretization bias
- ✅ Good for quick exploration
- 📘 **Update:** `x_{t+1} = x_t - η∇f(x_t) + √(2η)ξ_t`

### Comprehensive Diagnostics

Our diagnostic suite provides deep insights into sampler performance:

| Metric | Description | Purpose |
|--------|-------------|---------|
| **ESS** | Effective Sample Size | Measures efficiency accounting for autocorrelation |
| **ACF** | Autocorrelation Function | Identifies mixing issues |
| **Acceptance Rate** | % proposals accepted (MALA only) | Tune step size |
| **Divergence Rate** | % numerically unstable samples | Numerical stability check |
| **Quantile MAE** | Mean absolute error vs theory | Sampling accuracy |
| **Tail Coverage** | Exploration of 1%, 5%, 95%, 99% quantiles | Tail behavior assessment |

### Visualizations

- 📈 **Trace plots** - Time series of samples
- 📉 **Autocorrelation plots** - Mixing diagnostics
- 📊 **Q-Q plots** - Sample vs theoretical quantiles
- 📊 **Histograms** - Empirical distributions with theoretical markers

---

## 🚀 Quick Start

### Basic Usage

```python
import numpy as np
from overdamped.mala.mala_example_gaussian import GaussianMALA, GaussianDiagnostics

# 1. Create sampler
sampler = GaussianMALA(
    d=3,                    # dimension
    eta=0.01,              # step size
    n_steps=100000,        # total iterations
    burn_in=10000,         # discard first 10k
    seed=42                # reproducibility
)

# 2. Run sampling
samples = sampler.run()

# 3. Diagnostics
diagnostics = GaussianDiagnostics(
    samples,
    d=3,
    acceptance_rate=sampler.acceptance_rate
)

# 4. Print comprehensive report
diagnostics.print_comprehensive_stats()

# 5. Visualizations
diagnostics.plot_trace(dims=[0, 1])
diagnostics.plot_autocorrelation(max_lag=100, dims=[0, 1])
diagnostics.plot_tail_exploration(dim=0)
```

### Custom Parameters

```python
# Gaussian with custom mean and covariance
from overdamped.mala.mala_example_gaussian import GaussianMALA

mu = np.array([1.0, 2.0, 3.0])
Sigma = np.diag([1.0, 2.0, 3.0])  # diagonal covariance

sampler = GaussianMALA(
    d=3,
    eta=0.01,
    n_steps=100000,
    burn_in=10000,
    mu=mu,
    Sigma=Sigma,
    seed=42
)
```

```python
# Student-t with custom parameters
from overdamped.mala.mala_example_student_t import StudentTMALA

sampler = StudentTMALA(
    d=3,
    nu=5.0,                              # degrees of freedom
    location=np.array([0.0, 0.0, 0.0]),
    scale=np.array([1.0, 1.0, 1.0]),
    eta=0.01,
    n_steps=100000,
    burn_in=10000,
    seed=42
)
```

### Running Examples

All distributions have ready-to-run examples:

```bash
# MALA examples
python overdamped/mala/mala_example_gaussian.py
python overdamped/mala/mala_example_exponential.py
python overdamped/mala/mala_example_student_t.py
python overdamped/mala/mala_example_cauchy.py
python overdamped/mala/mala_example_lognormal.py
python overdamped/mala/mala_example_double_well.py

# ULA examples
python overdamped/ula/ula_example_gaussian.py
python overdamped/ula/ula_example_exponential.py
python overdamped/ula/ula_example_student_t.py
python overdamped/ula/ula_example_cauchy.py
python overdamped/ula/ula_example_lognormal.py
python overdamped/ula/ula_example_double_well.py
```

---

## 📁 Project Structure

```
langevin-sampling/
├── README.md                          # This file
│
├── overdamped/                        # Overdamped Langevin implementations
│   │
│   ├── mala/                         # MALA implementations
│   │   ├── mala_runner.py           # Abstract base class
│   │   ├── mala_example_gaussian.py
│   │   ├── mala_example_exponential.py
│   │   ├── mala_example_student_t.py
│   │   ├── mala_example_cauchy.py
│   │   ├── mala_example_lognormal.py
│   │   └── mala_example_double_well.py
│   │
│   └── ula/                          # ULA implementations
│       ├── ula_runner.py            # Abstract base class
│       ├── ula_example_gaussian.py
│       ├── ula_example_exponential.py
│       ├── ula_example_student_t.py
│       ├── ula_example_cauchy.py
│       ├── ula_example_lognormal.py
│       └── ula_example_double_well.py
│
└── metrics/                           # Diagnostics
    ├── comprehensive_diagnostics.py   # Main diagnostics class
    ├── median_and_quantile_diagnostics.py  # Legacy simple diagnostics
    └── DIAGNOSTICS_README.md         # Diagnostics documentation
```

---

## 🔧 Installation

### Requirements

```bash
pip install numpy scipy matplotlib
```

**Versions tested:**
- Python 3.7+
- NumPy 1.19+
- SciPy 1.5+
- Matplotlib 3.1+

### Setup

```bash
git clone <repository-url>
cd langevin-sampling
# No installation needed - ready to use!
```

---

## 📖 Usage Guide

### 1. Choosing an Algorithm

| Use MALA when: | Use ULA when: |
|----------------|---------------|
| ✓ Need exact sampling | ✓ Quick exploration |
| ✓ Heavy-tailed distributions | ✓ Light-tailed distributions |
| ✓ Final production sampling | ✓ Prototyping |
| ✓ Can afford MH overhead | ✓ Need speed |

### 2. Tuning Step Size (η)

**MALA:** Check acceptance rate
- Too low (< 20%): Decrease η
- Too high (> 80%): Increase η
- Target: 40-70% depending on distribution

**ULA:** Check bias
- Smaller η → less bias, slower mixing
- Larger η → more bias, faster mixing
- Rule of thumb: η ≈ 0.01 to 0.001

### 3. Step Size Recommendations by Distribution

| Distribution | MALA (η) | ULA (η) | Notes |
|--------------|----------|---------|-------|
| Gaussian | 0.01 - 0.05 | 0.01 - 0.02 | Well-behaved |
| Exponential | 0.01 - 0.05 | 0.01 - 0.02 | Simple, constant gradient |
| Student-t (ν>5) | 0.01 - 0.02 | 0.005 - 0.01 | Moderate tails |
| Student-t (ν≤5) | 0.005 - 0.01 | 0.002 - 0.005 | Heavy tails |
| Cauchy | 0.001 - 0.005 | 0.0005 - 0.002 | Very heavy tails |
| Lognormal | 0.01 - 0.02 | 0.005 - 0.01 | Bounded support |
| Double-Well | 0.005 - 0.01 | 0.002 - 0.005 | Multimodal |

### 4. Interpreting Diagnostics

#### ✅ **Good Sampling:**
- ESS ratio > 10%
- Acceptance rate 40-70% (MALA)
- ACF decays to < 0.05 within 100 lags
- Quantile MAE < 0.1
- Tail coverage within 20% of expected

#### ⚠️ **Warning Signs:**
- ESS ratio < 5% → Increase n_steps or adjust η
- Acceptance rate < 20% or > 80% → Adjust η
- ACF remains high (> 0.1) after 200 lags → Decrease η
- Divergence rate > 1% → Decrease η

---

## 🎓 Mathematical Background

### Langevin Dynamics

Sample from distribution `π(x) ∝ exp(-f(x))` using:

```
dX_t = -∇f(X_t)dt + √2 dW_t
```

where `W_t` is standard Brownian motion.

### Discretization

**ULA (Euler-Maruyama):**
```
X_{t+1} = X_t - η∇f(X_t) + √(2η)ξ_t,  ξ_t ~ N(0,I)
```

**MALA (with Metropolis correction):**
1. Propose: `X' = X_t - η∇f(X_t) + √(2η)ξ_t`
2. Accept with probability: `min(1, π(X')q(X_t|X') / (π(X_t)q(X'|X_t)))`

where `q(·|·)` is the proposal density.

---

## 📊 Example Output

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

--- BASIC STATISTICS ---

Dim 0:
  Mean: 0.003
  Empirical median: 0.002
  Theoretical median: 0.000
  Std dev: 0.998
  MAD: 0.673
```

---