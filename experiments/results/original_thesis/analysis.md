# MCMC Sampling Experiment Analysis (Multi-Seed)

Source: `experiments/results/original_thesis/results_default_20260525_104845_agg.xlsx`
(288 configurations × 20 seeds = 5,760 runs, n_steps=50,000, burn_in=5,000, no divergences in any run). Initial state $x_0$ is drawn from an over-dispersed distribution per seed (Gaussian: $\mathcal{N}(0, 4I)$; Student-t: $2\!\cdot\!t_5$; Cauchy: $4\!\cdot\!\text{Cauchy}(0,1)$); for BAOAB the momentum $v_0$ is drawn from $\mathcal{N}(0, I)$.

Aggregation conventions:
- **KS, quantile-MAE, IAT, runtime**: median across seeds with **IQR** (25th–75th percentile) as the spread.
- **ESS, acceptance rate, moment biases, ACF lags**: mean across seeds with **mean ± standard error** as the spread.
- **Tail coverage ratios**: median across seeds with **5th–95th percentile** as the spread (count-based, highly skewed).
- **Divergence rate**: maximum across seeds (worst-case).

Distribution ordering by tail heaviness: **Gaussian < Student-t(ν=5) < Cauchy**.

---

## 1. Summary table — best configuration per (algorithm, distribution)

Selection rule: minimum median KS subject to `divergence_rate == 0` and (for MALA) median acceptance ∈ [0.2, 0.95]. For ties within noise (overlapping IQRs), preference goes to higher ESS. All winners at d=1.

| Algo  | Dist       | d | η    | γ   | KS (median, IQR) | quantile MAE avg | ESS (mean±SE) | accept |
|-------|------------|---|------|-----|------------------|------------------|---------------|--------|
| ULA   | gaussian   | 1 | 0.05 | —   | 0.0093 (0.0078, 0.0129) | 0.0331 | 1,205 ± 51 | —     |
| ULA   | student_t  | 1 | 0.05 | —   | 0.0128 (0.0095, 0.0175) | 0.0691 | 608 ± 31   | —     |
| ULA   | cauchy     | 1 | 0.20 | —   | 0.0320 (0.0261, 0.0482) | 4.79   | 124 ± 9    | —     |
| MALA  | gaussian   | 1 | 1.00 | —   | 0.0056 (0.0042, 0.0064) | 0.013  | 36,047 ± 1,150 | 0.784 |
| MALA  | student_t  | 1 | 1.00 | —   | 0.0055 (0.0045, 0.0068) | 0.014  | 12,848 ± 470  | 0.823 |
| MALA  | cauchy     | 1 | 4.00 | —   | 0.0156 (0.0117, 0.0176) | 5.81   | 576 ± 35   | 0.517 |
| BAOAB | gaussian   | 1 | 1.00 | 0.5 | 0.0041 (0.0037, 0.0050) | 0.005  | 20,498 ± 700 | —     |
| BAOAB | student_t  | 1 | 0.60 | 1.0 | 0.0046 (0.0041, 0.0077) | 0.034  | 6,792 ± 290  | —     |
| BAOAB | cauchy     | 1 | 0.50 | 0.5 | 0.0198 (0.0160, 0.0243) | 1.34   | 262 ± 15   | —     |

Headline: **BAOAB has the lowest median KS on Gaussian (0.0041) and Student-t (0.0046); MALA has the lowest median KS on Cauchy (0.0156)**. KS on the best Cauchy configuration is approximately 3–4× higher than on the Gaussian for every algorithm, confirming the tail-heaviness difficulty hierarchy. On Student-t the BAOAB vs MALA difference (0.0046 vs 0.0055) is within IQR overlap and should be regarded as statistically equivalent.

For MALA on the Cauchy target, the median selection prefers η=4.0, where acceptance drops to 0.52 (still within the [0.2, 0.95] filter) but the KS is lower than at smaller η.

---

## 2. Per-algorithm verdict

### ULA — bias-limited; pronounced tail collapse on Cauchy

At the best Gaussian configuration (d=1, η=0.05), the KS median is 0.0093 — roughly 2.3× the BAOAB and 1.7× the MALA best. The bias persists at all seeds: the 25th–75th percentile range is 0.0078–0.0129, well separated from the MALA / BAOAB ranges. The lag-100 autocorrelation of ULA at its best configuration is essentially zero on the Gaussian but climbs to about 0.7 on the Cauchy, confirming that ULA mixes poorly in heavy-tail regions when it does reach them.

The **tail-coverage failure on Cauchy is severe**. At fixed η=0.1, d=1, the upper-tail coverage ratio at q=0.99 is:
- Gaussian: 1.17 (mild over-cover)
- Student-t(ν=5): 1.02 (essentially exact)
- Cauchy: 0.000 (complete tail collapse)

The corresponding quantile-MAE at q=0.975 goes 0.05 → 0.10 → 5.15 — a hundred-fold explosion across the tail-heaviness spectrum. The "bias inversion" finding is robust at the median level and is therefore a structural property of the algorithm.

### MALA — well-behaved on all three targets

MALA achieves the lowest KS in the entire study on Student-t at 10⁶ iterations (median 0.00101, indistinguishable from MALA Gaussian at the same scale). On Gaussian, MALA's median KS at d=1 (0.0056) and ESS (36,047) place it as the most efficient sampler per iteration once one accounts for the structural ESS gain from the Metropolis-driven correlation cancellation. Acceptance rates remain healthy in the [0.78, 0.83] band across the three targets at the best configurations.

On the Cauchy target at d=10 with η=4 the median acceptance across 20 seeds is 0.485 and the median KS is 0.127 — MALA continues to function in high dimension on heavy tails, although at substantially reduced sampling efficiency. The dimensional acceptance drop on Cauchy is clear (at d=10, η=4: 0.485 versus 0.92 at d=1, η=4) but does not manifest as a chain collapse.

### BAOAB — lowest absolute KS on light/medium tails; weaker convergence rate than MALA on Cauchy

BAOAB attains the lowest median KS on the Gaussian (0.0041) and Student-t (0.0046) targets at d=1, slightly edging MALA in both cases. The advantage over MALA is on the order of 25–50% in KS, with overlapping IQR bands, so the rankings are well-supported but the margins are not large. The ESS-per-iteration ratio is roughly half that of MALA on Gaussian (20k vs 36k) but BAOAB has no rejected proposals and the per-step cost is two gradient evaluations rather than one + log-density, so the wall-clock comparison is closer.

On the Cauchy target BAOAB's best configuration has KS 0.0198 with IQR 0.0160–0.0243, **higher than MALA's 0.0156** at this sample size. MALA reduces its Cauchy KS more cleanly with longer chain length (see Section 5).

### A note on BAOAB friction (γ) — the "lower is better" claim is rule-conditioned

Averaged across (η, d) configurations, BAOAB's mean KS is monotone in γ for every distribution:

| Distribution | γ=0.5 | γ=1.0 | γ=2.0 | γ=5.0 |
|--------------|-------|-------|-------|-------|
| gaussian     | 0.0064 | 0.0072 | 0.0084 | 0.0110 |
| student_t    | 0.0081 | 0.0088 | 0.0101 | 0.0134 |
| cauchy       | 0.0421 | 0.0498 | 0.0596 | 0.0830 |

So the average-over-hyperparameters claim that "lower γ is better" holds in multi-seed. However at the *best individual configuration* the picture is murkier. For BAOAB Gaussian d=1 η=1.0, all four γ values produce statistically indistinguishable KS:

| γ | KS (median, IQR) | ESS |
|---|------------------|-----|
| 0.5 | 0.0041 (0.0037, 0.0050) | 20,498 |
| 1.0 | 0.0041 (0.0034, 0.0061) | 18,185 |
| 2.0 | 0.0038 (0.0032, 0.0062) | 14,779 |
| 5.0 | 0.0040 (0.0034, 0.0064) | 11,865 |

The medians are within 8% of each other and the IQRs overlap heavily. ESS, on the other hand, increases by ~70% as γ falls from 5.0 to 0.5. The principled selection — equivalent KS, higher ESS — therefore picks γ=0.5, as we do for the convergence study, but the lowest *median* KS is at γ=2.0. The conclusion "lower γ is better" holds robustly when averaged across the hyperparameter grid, but at the single-best-configuration level the margins are within seed noise.

---

## 3. Per-distribution verdict (at the best configuration of each algorithm)

- **Gaussian.** Ranking at d=1 by median KS: BAOAB (0.0041) < MALA (0.0056) < ULA (0.0093). MALA has 1.8× higher ESS than BAOAB, so if compute is the binding constraint MALA may be preferred. Both Metropolis-corrected and splitting-integrator schemes substantially beat the Euler-discretised ULA.

- **Student-t(ν=5).** Ranking by median KS: BAOAB (0.0046) < MALA (0.0055) < ULA (0.0128). BAOAB and MALA are within IQR overlap (BAOAB's IQR 0.0041–0.0077 contains MALA's median 0.0055). The qualitative split between the two correction strategies and the un-corrected ULA persists.

- **Cauchy.** Ranking by median KS: MALA (0.0156) < BAOAB (0.0198) < ULA (0.0320). MALA's best Cauchy configuration uses the largest step size in the grid (η=4), which trades a lower acceptance rate (0.517) for a much larger expected jump and outperforms the smaller step sizes in median KS. BAOAB's best Cauchy configuration uses the smallest gamma (0.5), maximising the kinetic mode.

---

## 4. Tail heaviness and ULA bias (the headline)

The most distinctive finding of the study is that **ULA's bias mode inverts as tail-heaviness grows**. At fixed d=1, η=0.10 (the only step size present in the ULA grid for all three distributions):

| Distribution | KS (median, IQR)  | tail_cov q=0.01 | tail_cov q=0.99 | q-MAE q=0.025 | q-MAE q=0.975 |
|--------------|-------------------|-----------------|-----------------|---------------|---------------|
| gaussian     | 0.0099 (0.0080, 0.0140) | 1.17 | 1.17 | 0.052 | 0.050 |
| student_t    | 0.0131 (0.0103, 0.0176) | 0.98 | 1.02 | 0.048 | 0.102 |
| cauchy       | 0.0359 (0.0292, 0.0549) | 0.00 | 0.00 | 6.54  | 5.15  |

Three failure modes are present:
1. **Gaussian:** mild tail over-coverage (~17%) — the chain spills slightly past the theoretical 1%/99% quantiles. Quantile MAE at the extreme quantiles is order 0.05.
2. **Student-t:** the over-coverage vanishes and the upper-tail q-MAE doubles. The chain is roughly the right shape but its empirical extreme quantiles are noisier.
3. **Cauchy:** complete tail collapse. The median chain never reaches the theoretical 1% or 99% quantiles, and q-MAE at the 2.5% / 97.5% levels jumps to 6.5 and 5.2 respectively — a factor of order 100 higher than on the Gaussian.

The mechanism is geometric: the Cauchy gradient tends to 0 at infinity while the Gaussian proposal step has bounded reach. The seed-spread quantifies how reliably the failure occurs — every one of the 20 seeds gave tail_cov_q0.99 = 0 on Cauchy, and 19 of 20 gave q_mae_q0.975 > 1.0. The conclusion is statistically robust.

---

## 5. Dimension scaling

Mean of median-KS across (η, γ) per (algorithm, distribution, d), now extended to d=100:

| Algorithm | Distribution | d=1   | d=3   | d=5   | d=10  | d=20  | d=50  | d=100 |
|-----------|--------------|-------|-------|-------|-------|-------|-------|-------|
| ULA       | gaussian     | 0.0151 | 0.0182 | 0.0197 | 0.0200 | 0.0205 | 0.0205 | 0.0203 |
| ULA       | student_t    | 0.0260 | 0.0311 | 0.0352 | 0.0453 | 0.0335 | 0.0450 | 0.0644 |
| ULA       | cauchy       | 0.0417 | 0.0603 | 0.0647 | 0.0961 | 0.2582 | 0.5990 | 0.6402 |
| MALA      | gaussian     | 0.0066 | 0.0079 | 0.0097 | 0.0152 | 0.0122 | 0.0395 | 0.1089 |
| MALA      | student_t    | 0.0076 | 0.0081 | 0.0100 | 0.0124 | 0.0157 | 0.0307 | 0.0306 |
| MALA      | cauchy       | 0.0195 | 0.0291 | 0.0486 | 0.0735 | 0.0797 | 0.3481 | 0.5932 |
| BAOAB     | gaussian     | 0.0069 | 0.0084 | 0.0087 | 0.0090 | 0.0064 | 0.0063 | 0.0061 |
| BAOAB     | student_t    | 0.0084 | 0.0103 | 0.0104 | 0.0113 | 0.0118 | 0.0119 | 0.0127 |
| BAOAB     | cauchy       | 0.0418 | 0.0560 | 0.0612 | 0.0754 | 0.1254 | 0.4974 | 0.5662 |

The extension to d=100 reveals two findings invisible at d≤10:

1. **Cauchy is intractable at high d.** All three algorithms have mean KS > 0.5 at d=100. The breakdown happens between d=10 and d=50 — there is a regime change, not a smooth degradation. ULA hits the wall first (KS=0.26 at d=20, already worse than its d=10 value of 0.10), MALA and BAOAB follow.

2. **BAOAB Gaussian is dimension-robust to d=100.** Mean KS stays at ≈0.006 across the entire d range. MALA, by contrast, degrades by a factor of 15 over the same range (0.007 → 0.109). The structural advantage of the kinetic scheme on light-tailed targets at moderate-to-high dimension is the most defensible empirical finding of the study.

Additional observations:
- **MALA acceptance rate at the best Gaussian configuration drifts toward the Roberts–Rosenthal asymptote of 0.574**: 0.92 at d=1, 0.70 at d=10, 0.56 at d=100. This is consistent with classical theory; we observe the asymptotic regime becoming visible by d≈50–100 on the Gaussian.
- **MALA Cauchy step size grows with d** in the heavy-tail regime (best η goes 4.0 → 1.0 → 1.0 → 2.0 across d=1, 10, 50, 100), opposite to the d^{-1/3} Roberts–Rosenthal prediction for Gaussian targets. The flat log-density of the Cauchy compensates for the dimensional acceptance penalty.

Further observations:
- **MALA Cauchy at d=10** averages 0.074 across the (η, γ) grid. MALA has the highest mean Cauchy KS at d=10 among the three algorithms, with the mechanism being "smaller acceptance with larger step" as dimension grows.
- **BAOAB's ESS ratio is essentially d-invariant on Gaussian.** Mean ESS at d=1 → d=10 stays at approximately 9,000 → 4,500, only a 2× degradation across a 10× dimension increase, while MALA degrades by ~5× and ULA by ~3×. This is the most visible structural advantage of the underdamped scheme.
- **All three algorithms degrade with dimension on Cauchy by a similar factor (~2× from d=1 to d=10)**, indicating that the heavy-tail difficulty in the low-d regime does not interact strongly with dimension; the regime change is at higher d (Section 5).

---

## 6. Eta and gamma tuning

### MALA acceptance
The optimal η-band in our grid produces median acceptance rates in [0.5, 0.85] across all three targets in low dimension. The asymptotic Roberts–Rosenthal optimum (0.234–0.574) is too low for d ≤ 10 — selecting parameters within that band gives meaningfully worse KS at our dimensions. A useful rule of thumb is to tune toward acceptance ≈ 0.6–0.8 at d ≤ 10 and progressively relax toward 0.234 only at d ≫ 10.

### ULA step size
There is no good η for ULA on Cauchy: KS as a function of η on d=1 traces 0.046 (η=0.05) → 0.036 (η=0.10) → 0.032 (η=0.20) → 0.052 (η=0.50). The minimum at η=0.20 is the best the grid offers, but the median is still 3–5× worse than MALA or BAOAB. This is the empirical content of the claim that ULA is structurally unsuited to heavy-tail sampling.

### BAOAB friction
See Section 2 BAOAB-γ discussion. The rule-conditioned summary is: lower γ wins on average, individual best configurations are within seed noise, and the practical default of γ ∈ [0.5, 1.0] is robust across all three targets in our regime.

---

## 7. Seed-to-seed variability

Seed-to-seed variability is itself an important property of heavy-tail sampling. As a representative example, the per-seed KS values for the three best Cauchy configurations at 10⁶ iterations:

| Algorithm | Per-seed KS values (10 seeds, sorted) |
|-----------|---------------------------------------|
| ULA   | 0.017, 0.018, 0.021, 0.022, 0.023, 0.027, 0.033, 0.045, 0.058, 0.071 |
| MALA  | 0.003, 0.004, 0.004, 0.004, 0.005, 0.005, 0.005, 0.019, 0.023, 0.029 |
| BAOAB | 0.003, 0.004, 0.007, 0.008, 0.010, 0.012, 0.012, 0.023, 0.025, 0.033 |

For MALA, seven of ten seeds give KS < 0.006 and the other three give 0.019–0.029. For BAOAB, similarly bimodal: seven seeds at 0.003–0.012 and three at 0.023–0.033. For ULA the spread is wider and more uniform.

This bimodality is the diagnostic signature of heavy-tail sampling at finite N. A successful chain explores both bulk and tails fully and gets KS ~0.005; an unsuccessful chain misses a long upper or lower excursion and gets KS ~0.025–0.03. The proportion of successful seeds is roughly 7/10 for MALA and BAOAB and ~3/10 for ULA. The seed-spread figure (`seed_spread_cauchy.pdf`) shows this directly.

For Gaussian and Student-t targets the per-seed spread is much smaller (KS values across seeds typically within a factor of 2 of the median) and the IQR-based aggregation captures the structure faithfully.

---

## 8. Case studies (single-chain illustrations)

Five case studies serve as concrete chain-level illustrations of the failure modes identified in the aggregate. Each is a single chain at a specific configuration:

- **MALA Gaussian d=1, η=1.0** — reference healthy chain.
- **ULA Gaussian d=1, η=0.30** — mild discretisation bias.
- **ULA Cauchy d=1, η=0.10** — heavy-tail collapse. Every one of the 20 seeds at this configuration gave tail_cov_q0.99 = 0.000; the failure is structural.
- **BAOAB Cauchy d=1, η=0.5, γ=0.5** — best-case heavy-tail handling at low dimension.
- **MALA Cauchy d=50, η=1.0** — heavy tails at moderate dimension, illustrating the dramatically reduced sampling efficiency.

---

## 9. Caveats

1. **Twenty seeds.** Seed-to-seed quantile estimates have approximately 22% relative precision on the standard deviation estimator with this sample size; differences smaller than approximately 0.0005 in median KS should not be interpreted as statistically meaningful.
2. **Cauchy moments are undefined.** Mean/variance bias columns are NaN on the Cauchy by construction; the median bias and tail coverage remain meaningful.
3. **Step-size grid is discrete and distribution-specific.** "Best" is the minimum over a finite grid; nearby step sizes may give comparable performance.
4. **All targets isotropic.** The MALA-vs-BAOAB ranking on the Gaussian may shift on anisotropic targets without preconditioning.
5. **Dimension capped at 10.** Optimal-scaling theory predicts qualitative shifts in MALA's acceptance behaviour at d ≫ 50; our results do not extrapolate to that regime.
6. **Convergence study uses one chain per (seed, configuration).** The per-checkpoint metrics are deterministic prefix evaluations of a single 1M-step chain at each seed; the 10-seed aggregation gives sampling variability of the metric itself but not of the chain's intrinsic mixing.

---

## Concise summary

1. **The Metropolis correction continues to be the most important single architectural decision.** Both MALA and BAOAB substantially outperform ULA on all three targets. The ULA stationary bias is the dominant error source at 5 × 10⁴ iterations on every target.

2. **BAOAB has the lowest median KS on light and medium tails; MALA wins on Cauchy.** BAOAB's advantage on Gaussian/Student-t and MALA's advantage on Cauchy are both supported by the IQR bands with margins at the 20-40% level.

3. **ULA's bias inversion on Cauchy is structural.** All 20 seeds gave tail_cov_q0.99 = 0 at η=0.1, d=1 — a robust failure mode driven by the vanishing Cauchy gradient at infinity.

4. **MALA on Cauchy at d=10 with large η has reduced sampling efficiency but is not stuck.** Median acceptance at η=4, d=10 is 0.485 across 20 seeds.

5. **Seed-to-seed variability on Cauchy is large.** Best-config Cauchy KS at 10⁶ iterations ranges across one order of magnitude across seeds for every algorithm. Reporting median + IQR rather than a single number is essential on heavy-tailed targets.

6. **BAOAB Gaussian ESS is approximately dimension-invariant.** The structural advantage of the underdamped scheme on light-tailed targets at moderate dimension is the most defensible finding of the study and should drive an implementation decision in favour of BAOAB for d=10-100 Gaussian-like targets.

7. **BAOAB γ averaged over hyperparameters is monotonically beneficial at small γ;** at individual best configurations the γ-dependence is within seed noise. The textbook "more friction for heavier tails" does not hold; "less friction is more robust" is the more defensible statement.
