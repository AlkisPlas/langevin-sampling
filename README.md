# Langevin Sampling Algorithms

This repository contains the code of a master's thesis. The study compares three
Langevin sampling algorithms on eight target distributions, in dimensions from 1 to 1000.

## What the study does

A sampling algorithm makes a chain of points. After many steps, the points must have
the distribution of a given target. The three algorithms use the gradient of the
log-density of the target. They do not need the normalizing constant.

The study has these steps:

1. Run each algorithm on each target and in each dimension.
2. Change the step size η on a grid. For BAOAB, also change the friction γ.
3. Run each configuration with 20 different random seeds.
4. Find the best configuration for each algorithm, target and dimension.
5. Run the best configurations for 10⁶ steps. This shows if the error decreases with
   more steps, or if it stays (bias).

Each chain in the grid has 50,000 steps. The first 5,000 steps are not used (burn-in).
The grid has 45,720 chains.

## Algorithms

| Algorithm | Step | Bias |
|---|---|---|
| ULA (Unadjusted Langevin Algorithm) | A gradient step plus Gaussian noise. | Yes. The bias is proportional to η. |
| MALA (Metropolis-Adjusted Langevin Algorithm) | The ULA step is a proposal. A Metropolis–Hastings test accepts or rejects it. | No, for each η. |
| BAOAB | The state has a position and a velocity (kinetic Langevin). The step has five parts: B, A, O, A, B. | Yes, but smaller. The bias is proportional to η². On a Gaussian target, the position has no bias. |

Each algorithm uses one new gradient per step.

## Target distributions

| Target | Difficulty | Parameter | Dimensions d |
|---|---|---|---|
| Gaussian | None. It is the reference. | — | 1, 3, 5, 10, 20, 50, 100, 200, 500, 1000 |
| Student-t | Tails that decrease slowly | ν = 5 | 1, 3, 5, 10, 20, 50, 100, 200, 500, 1000 |
| Cauchy | Very heavy tails. No mean and no variance. | ν = 1 | 1, 3, 5, 10, 20, 50, 100, 200, 500, 1000 |
| Anisotropic Gaussian | The variances go from 1 to κ. | κ = 1000 | 2, 4, 10, 20, 50, 100, 200, 500, 1000 |
| Banana | Curved dependence in each pair of coordinates | b = 0.03 | 2, 4, 10, 20, 50, 100, 200, 500, 1000 |
| Gaussian mixture | Three peaks at −a, 0, a on the first coordinate, with low density between them | a = 6 | 2, 4, 10, 20, 50, 100, 200, 500, 1000 |
| Funnel | The scale of the coordinates changes with the first coordinate v | σ_v = 2 | 2, 4, 10, 20, 50, 100, 200, 500, 1000 |
| Double well | Two wells in each coordinate, with a barrier between them | β = 16 | 2, 4, 10, 20, 50, 100, 200, 500, 1000 |

For each target, the code also gives the true mean, variance, quantiles and CDF of each
coordinate. The metrics compare the samples with these values.

## Metrics

| Metric | What it measures | Best value |
|---|---|---|
| KS (Kolmogorov–Smirnov) | The largest distance between the CDF of the samples and the true CDF of a coordinate | 0 |
| q-MAE | The error of the sample quantiles at q = 0.025, 0.5 and 0.975 | 0 |
| Tail coverage | The fraction of samples beyond the true 1 %, 5 %, 95 % and 99 % quantiles, divided by the expected fraction | 1 |
| b² | The squared error of a sample mean (of x_i² and, for the mixture and the double well, of x_i), divided by its true variance. A value below 0.01 is a success. | 0 |
| ESS (effective sample size) | The number of independent samples that give the same information as the chain | High |
| ESS per second | The ESS divided by the run time | High |
| Acceptance rate | The fraction of accepted proposals (MALA only) | — |
| Divergence rate | The fraction of samples that are not finite or larger than 10⁶ | 0 |
| Peak metrics | The rate of jumps between peaks and the error of the mass in each peak (mixture and double well only) | — |

The main metric is KS for the Gaussian, the Student-t and the Cauchy. It is b² for the
other targets, because for some of their coordinates the true CDF has no formula.
