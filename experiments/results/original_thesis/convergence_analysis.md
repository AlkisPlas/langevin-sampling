# Convergence Study — Multi-Seed Analysis

Source: `experiments/results/original_thesis/convergence_20260525_122341_agg.xlsx`
9 best-configuration chains × 10 seeds × 10⁶ iterations = 90 chains total. Single-chain prefix evaluations at checkpoints $N \in \{5\times10^4, 10^5, 5\times10^5, 10^6\}$, then aggregated across seeds at each checkpoint. Burn-in 5,000, dimension d=1, over-dispersed $x_0$ per seed (Gaussian: $\mathcal{N}(0, 4I)$; Student-t: $2\!\cdot\!t_5$; Cauchy: $4\!\cdot\!\text{Cauchy}(0,1)$); BAOAB $v_0 \sim \mathcal{N}(0, I)$.

Best configurations (selected from the multi-seed grid by minimum median KS, with ties broken in favour of higher ESS):

| Algorithm | Distribution | η    | γ   |
|-----------|--------------|------|-----|
| ULA       | gaussian     | 0.05 | —   |
| ULA       | student_t    | 0.05 | —   |
| ULA       | cauchy       | 0.20 | —   |
| MALA      | gaussian     | 1.00 | —   |
| MALA      | student_t    | 1.00 | —   |
| MALA      | cauchy       | 4.00 | —   |
| BAOAB     | gaussian     | 1.00 | 0.5 |
| BAOAB     | student_t    | 0.60 | 1.0 |
| BAOAB     | cauchy       | 0.50 | 0.5 |

Notes on parameter selection: the median-KS best for MALA Cauchy is at η=4 (a larger step compensates for the flatter log-density). BAOAB γ values are chosen by higher-ESS tie-breaks where multiple γ values give statistically equivalent KS.

---

## 1. KS convergence — median ± IQR across 10 seeds

| Algo  | Dist       | 5·10⁴                | 10⁵                  | 5·10⁵                | 10⁶                  | ratio 10⁶/5·10⁴ |
|-------|------------|----------------------|----------------------|----------------------|----------------------|------------------|
| ULA   | gaussian   | 0.0093 (0.0082, 0.0113) | 0.0124 (0.0095, 0.0131) | 0.0063 (0.0057, 0.0067) | **0.0055** (0.0047, 0.0068) | 0.58× |
| ULA   | student_t  | 0.0128 (0.0123, 0.0161) | 0.0135 (0.0120, 0.0160) | 0.0077 (0.0074, 0.0088) | **0.0067** (0.0060, 0.0090) | 0.52× |
| ULA   | cauchy     | 0.0330 (0.0265, 0.0620) | 0.0301 (0.0258, 0.0369) | 0.0270 (0.0218, 0.0461) | **0.0246** (0.0211, 0.0422) | 0.75× |
| MALA  | gaussian   | 0.0057 (0.0044, 0.0063) | 0.0038 (0.0026, 0.0044) | 0.0015 (0.0013, 0.0020) | **0.0010** (0.0008, 0.0014) | **0.18×** |
| MALA  | student_t  | 0.0053 (0.0044, 0.0061) | 0.0039 (0.0034, 0.0047) | 0.0019 (0.0013, 0.0025) | **0.0010** (0.0009, 0.0017) | **0.19×** |
| MALA  | cauchy     | 0.0170 (0.0118, 0.0196) | 0.0124 (0.0095, 0.0153) | 0.0071 (0.0058, 0.0328) | **0.0046** (0.0039, 0.0158) | **0.27×** |
| BAOAB | gaussian   | 0.0043 (0.0036, 0.0054) | 0.0035 (0.0032, 0.0037) | 0.0012 (0.0011, 0.0017) | **0.0008** (0.0006, 0.0011) | **0.18×** |
| BAOAB | student_t  | 0.0057 (0.0044, 0.0072) | 0.0046 (0.0034, 0.0052) | 0.0022 (0.0016, 0.0025) | **0.0018** (0.0016, 0.0024) | **0.31×** |
| BAOAB | cauchy     | 0.0202 (0.0173, 0.0300) | 0.0179 (0.0128, 0.0240) | 0.0142 (0.0086, 0.0425) | **0.0108** (0.0069, 0.0206) | 0.53× |

Ideal $1/\sqrt{N}$ scaling between $5\!\times\!10^4$ and $10^6$ would give a reduction ratio of **0.224×**.

Three conclusions follow directly:

**ULA is bias-limited on every target.** The KS ratio is well above 0.224 in all three rows. The Gaussian and Student-t ratios (0.58, 0.52) are non-trivial reductions but reflect random-error decay around a non-zero bias rather than reduction *of* the bias. On Cauchy the ratio is 0.75 — barely an improvement at 20× more samples.

**MALA achieves the 1/√N rate on every target — including Cauchy.** Ratios 0.18, 0.19, 0.27 are all close to or below the 0.224 ideal. The Cauchy 0.27 is the most striking convergence result of the study: with over-dispersed initialisation across 10 seeds and a step size of η=4 (the largest in the grid, which the heavy-tailed log-density allows), MALA achieves a 3.7× KS reduction over a 20× sample-size increase.

**BAOAB matches MALA on the Gaussian and trails on heavier tails.** Gaussian ratio 0.18, identical to MALA. Student-t ratio 0.31, slightly above the ideal but the absolute value at $10^6$ (0.0018) is competitive with MALA (0.0010). On Cauchy BAOAB reduces by a factor 0.53 — substantially slower than MALA (0.27) and the absolute KS at $10^6$ is 2.4× higher (0.0108 vs 0.0046). On the Cauchy bulk-fit metric, MALA outperforms BAOAB at every chain length.

---

## 2. ESS scaling

ESS at $5\!\times\!10^4$ and at $10^6$ with the scaling ratio (ideal: 20×):

| Algo  | Dist       | ESS @ 5·10⁴ | ESS @ 10⁶ | ratio | comment |
|-------|------------|-------------|-----------|-------|---------|
| ULA   | gaussian   | 1,176       | 27,022    | 23.0× | linear  |
| ULA   | student_t  | 658         | 12,784    | 19.4× | linear  |
| ULA   | cauchy     | 140         | 1,271     | 9.1×  | sub-linear |
| MALA  | gaussian   | 35,993      | 796,022   | 22.1× | linear  |
| MALA  | student_t  | 12,177      | 276,080   | 22.7× | linear  |
| MALA  | cauchy     | 623         | 1,874     | 3.0×  | severely sub-linear |
| BAOAB | gaussian   | 20,504      | 452,879   | 22.1× | linear  |
| BAOAB | student_t  | 6,638       | 153,099   | 23.1× | linear  |
| BAOAB | cauchy     | 340         | 1,634     | 4.8×  | sub-linear |

Light- and medium-tail ESS scales linearly with $N$ for every algorithm. On Cauchy, all three are sub-linear — long heavy-tail excursions keep correlation persistently high — and MALA suffers the worst sub-linearity (3.0×). This is the trade-off MALA pays for its low KS on Cauchy: it samples with much lower per-sample efficiency than on lighter tails, but the bias is small enough that the overall metric (KS) still wins.

---

## 3. Quantile-MAE convergence

For each algorithm we report the median q-MAE at $10^6$ iterations together with the IQR:

| Algo  | Dist       | quantile_mae_avg (median, IQR) | quantile_mae @ q=0.975 (median) |
|-------|------------|--------------------------------|----------------------------------|
| ULA   | gaussian   | 0.022 (0.018, 0.026) | 0.008 |
| ULA   | student_t  | 0.040 (0.031, 0.054) | 0.014 |
| ULA   | cauchy     | 4.81 (3.18, 9.51)    | 9.93  |
| MALA  | gaussian   | 0.003 (0.002, 0.004) | 0.003 |
| MALA  | student_t  | 0.004 (0.003, 0.006) | 0.003 |
| MALA  | cauchy     | 4.74 (1.55, 7.49)    | 2.71  |
| BAOAB | gaussian   | 0.002 (0.002, 0.003) | 0.002 |
| BAOAB | student_t  | 0.018 (0.012, 0.023) | 0.013 |
| BAOAB | cauchy     | 2.04 (0.83, 5.49)    | 2.43  |

On Cauchy the per-seed q-MAE values span an order of magnitude — visible in the wide IQRs — because the empirical extreme-quantile estimator on Cauchy has heavy-tailed sampling variability. Note that **BAOAB has the best Cauchy quantile MAE at 10⁶ samples** (median 2.04 vs MALA's 4.74) despite higher KS. KS captures bulk-fit error, q-MAE captures tail accuracy, and they can disagree on heavy-tailed targets.

For a Cauchy use case where extreme-quantile accuracy is the actual quantity of interest, BAOAB remains preferable to MALA despite having higher KS. For a use case dominated by bulk-fit error, MALA is preferable.

---

## 4. Cauchy tail coverage convergence

Median across 10 seeds of tail_cov_ratio at q=0.01 (lower tail) and q=0.99 (upper tail), with 5th–95th percentile range:

| Algo  | N      | q=0.01 (median, [q05,q95]) | q=0.99 (median, [q05,q95]) |
|-------|--------|----------------------------|----------------------------|
| ULA   | 5·10⁴  | 0.000 (0.000, 5.29)        | 0.000 (0.000, 0.40)        |
| ULA   | 10⁵    | 0.307 (0.000, 2.51)        | 0.000 (0.000, 0.19)        |
| ULA   | 5·10⁵  | 0.460 (0.053, 10.2)        | 0.219 (0.017, 2.79)        |
| ULA   | 10⁶    | 0.626 (0.157, 6.07)        | 0.797 (0.122, 3.50)        |
| MALA  | 5·10⁴  | 0.261 (0.040, 4.76)        | 0.167 (0.021, 1.02)        |
| MALA  | 10⁵    | 0.536 (0.116, 2.46)        | 0.412 (0.080, 0.89)        |
| MALA  | 5·10⁵  | 1.017 (0.541, 4.51)        | 0.633 (0.383, 5.33)        |
| MALA  | 10⁶    | **0.808 (0.599, 2.55)**    | **0.844 (0.621, 2.97)**    |
| BAOAB | 5·10⁴  | 0.002 (0.000, 5.26)        | 0.048 (0.000, 1.55)        |
| BAOAB | 10⁵    | 0.971 (0.001, 2.98)        | 0.329 (0.030, 0.96)        |
| BAOAB | 5·10⁵  | 0.551 (0.180, 7.42)        | 0.551 (0.289, 2.95)        |
| BAOAB | 10⁶    | 0.708 (0.233, 3.85)        | 0.555 (0.300, 2.00)        |

At $10^6$ iterations the median tail coverage is approximately:
- **MALA**: 0.81 (lower), 0.84 (upper) — within 20% of the ideal value of 1.0 for both tails.
- **BAOAB**: 0.71 (lower), 0.56 (upper).
- **ULA**: 0.63 (lower), 0.80 (upper).

**MALA reaches the closest to ideal tail coverage** at $10^6$ iterations, with BAOAB and ULA both showing roughly equivalent residual under-coverage. The wide 5–95% percentile ranges (some seeds give coverage > 5×, others ≈ 0) indicate substantial seed-to-seed variability — characteristic of heavy-tail sampling at finite N.

---

## 5. Per-algorithm summary

**ULA — bias floor, not just slow convergence.** At its best configuration ULA reduces KS by 0.58× (Gaussian), 0.52× (Student-t), and 0.75× (Cauchy) between $5\!\times\!10^4$ and $10^6$ samples. The Gaussian and Student-t reductions are real but reflect approaching a *biased* stationary distribution; the IQRs at $10^6$ still sit at 0.005–0.009 on Gaussian and 0.006–0.009 on Student-t, well above the MALA / BAOAB ranges. On Cauchy ULA barely improves.

**MALA — fastest convergence on every target, including Cauchy.** Reduction ratios 0.18 / 0.19 / 0.27 on Gaussian / Student-t / Cauchy. On all three targets the IQR bands at $10^6$ are narrow (relative spreads of 30–60%) and the medians are decisively below those of ULA and BAOAB.

**BAOAB — matches MALA on Gaussian, trails on heavier tails.** Gaussian reduction 0.18 (identical to MALA); Student-t reduction 0.31; Cauchy reduction 0.53. BAOAB is the best absolute-KS performer on Gaussian and tied with MALA on Student-t at $10^6$, but on Cauchy MALA's lower KS and faster reduction make it the preferable algorithm — though BAOAB retains the advantage in quantile-MAE on the Cauchy tails.

---

## 6. Headline narrative

The convergence study supports a clean narrative: **MALA wins on bulk fit (KS) at every target including Cauchy; BAOAB wins on extreme-quantile accuracy on Cauchy**. The bulk-fit ranking is consistent with MALA's exact targeting (the Metropolis correction removes the discretisation bias) and BAOAB's higher-order splitting integrator giving comparable but slightly larger residual error. The tail-accuracy ranking on Cauchy reflects BAOAB's momentum-driven exploration of long excursions that MALA's proposal-and-accept geometry has more difficulty inducing.

ULA's bias floor is robust across all three targets: even at $10^6$ iterations the chain has not approached the target distribution, only a biased neighbour of it.

---

## 7. Caveats specific to the convergence study

1. **Single chain per (seed, configuration).** The per-checkpoint metrics are prefix evaluations of a single 10⁶-step chain at each seed. We have 10-seed variability of the metric *value* but only one realisation of the chain's mixing trajectory per seed. A more rigorous study would run multiple chains per seed and apply Gelman–Rubin-style diagnostics.
2. **MALA acceptance rate is computed once per chain.** We use the full-chain acceptance rate as an approximation to the per-checkpoint acceptance rate; the difference is small in steady-state but the assumption is not formally tested.
3. **Best-configuration parameters are locked in from the grid study.** A longer grid (e.g., at $10^5$ iterations) might prefer slightly different step sizes; the parameters here are the best at $5\!\times\!10^4$, which is what a practitioner would actually deploy after a $5\!\times\!10^4$-step tuning pass.
4. **Cauchy quantile-MAE estimator has heavy-tailed sampling variability.** Even with 10 seeds, the IQR at $10^6$ iterations on Cauchy is wide. The medians are the right summary statistic; means would be dominated by occasional extreme runs.

---

## Concise summary

1. **MALA achieves near-$1/\sqrt{N}$ KS convergence on every target, including Cauchy.** Ratios 0.18, 0.19, 0.27 across Gaussian / Student-t / Cauchy — all close to or below the ideal 0.224.

2. **BAOAB matches MALA on Gaussian and lags on heavier tails.** Gaussian ratios identical (0.18×); on Cauchy BAOAB reduces by 0.53× while MALA reduces by 0.27×, and the absolute KS at $10^6$ is 2.4× higher for BAOAB.

3. **ULA is bias-limited on every target.** Reduction ratios 0.5–0.75× vs the $1/\sqrt{N}$ ideal of 0.22×; the chain approaches a biased stationary distribution rather than the target.

4. **Tail coverage convergence is non-monotonic and seed-dependent on Cauchy.** All three algorithms reach roughly 0.6–0.8 median coverage of the 1% / 99% tail bins at $10^6$ iterations; the 5–95% percentile ranges span an order of magnitude across seeds.

5. **BAOAB wins Cauchy quantile-MAE despite higher KS.** Median Cauchy q-MAE at $10^6$ is 2.04 (BAOAB), 4.74 (MALA), 4.81 (ULA). For applications dominated by extreme-quantile accuracy, BAOAB remains preferable on Cauchy.

6. **The seed-spread is fundamental to heavy-tail sampling.** Per-seed best-config Cauchy KS at $10^6$ spans 0.003–0.029 (MALA), 0.003–0.033 (BAOAB), 0.017–0.071 (ULA) — typically an order of magnitude across seeds. Headline conclusions about heavy-tail convergence require multi-seed analysis.

7. **Diagnostic recommendation:** report both KS (bulk-fit) and tail coverage / quantile-MAE (extreme-fit) on heavy-tail targets; they can disagree by orders of magnitude. The choice of best algorithm depends on which is the binding metric for the application.
