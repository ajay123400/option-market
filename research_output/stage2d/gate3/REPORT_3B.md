# Stage 2D — Gate 3b report: T11 synthetic known-truth validation (Tier A and Tier B)

Scope: T11 only. Measurement/statistics validation on **synthetic** data. No trading strategy, entry/exit rule, P&L, signal, sizing or Fyers link. Nothing here is evidence about
the real market; it tests whether the Stage 2D *inference machinery* recovers known truths and covers them at the stated rate. Plan: `GATE3B_IMPLEMENTATION_PLAN.md`
(written before the code). Baseline: Gate 3a commit `b2a7584` (HEAD, unchanged). Not committed, not pushed.

## 0. Review decision and required interpretation (T11 approved as implemented; methods not retuned to hit 93–97%)
1. **Tier A** is a successful known-truth *pipeline-recovery* validation (plumbing, definitions, annualization).
2. **Tier B** is evidence that the current percentile-bootstrap uncertainty procedure does **not** achieve nominal 95% coverage uniformly. This is a preserved research finding, not a defect to be tuned away.
3. **Spread-median estimands** reached about 93.7–94.7% coverage in B0 with b = 5.
4. **Ratio estimands** reached only about 88.7–91.7% in B0 and **must carry an explicit under-coverage warning** wherever they are quoted.
5. **Strong-dependence B1** produced materially worse coverage (best 78–87%).
6. **Row-level i.i.d. inference is invalid** for this dependent, overlapping structure.
7. The synthetic coverage percentages must **not** be extrapolated directly to the real dataset.
8. The simulation-calibration limitations are documented in section 6 and are preserved unchanged.

## 1. Headline

* **Tier A (full-pipeline recovery) passes.** The real, unmodified Stage 2A → 2B/2C → annualization-audit builders reproduce the generator's truth to numerical precision.
* **Tier B does NOT meet the 93–97% coverage target for the full set of estimands.** That is a result, not a defect I patched away. In the calibrated process the best cluster/block
  procedure (expiry block bootstrap, b = 5) covers the **spread-median** estimands at 93.7–94.7% (inside the band) but the **ratio** estimands (R1, R1e, R2, R2e, R3) at only
  88.7–91.7% (below the band; Wilson upper bounds still reach ~0.94 for several). Under a deliberately much more persistent process (B1) every method under-covers badly (best 78–87%).
* **Row-level inference is invalid, as expected and now quantified:** treating the 7,676 rows of a history as independent covers only 24–45% (calibrated process) and 13–20% (B1).
* **Date-block bootstrap (the Stage 2C method) under-covers** in the calibrated process (mean 89.7%, 3/5 estimands flagged `undercovers`); the expiry-level block bootstrap is better.
* **The heterogeneity Wald test is only partly calibrated:** placebo-by-row false-rejection 3.7% (S1) / 7.0% (R1); placebo-by-date 10.0% (S1) / 8.3% (R1) — the by-date placebo exceeds 5%.
* All limits are tied to a simulation that is **more dependent than the observed data** in some respects and heavier-skewed in others (§6), so the numbers are indicative, not definitive.

## 2. Methods

### 2.1 Tier A — small full-pipeline known-truth recovery (`synth_pipeline.py`, `build_gate3b.py`)
Three synthetic worlds (38 weeks, 36 weekly expiries, 954 snapshot rows each; implied-variance multiplier c = 1.00 / 1.30 / 0.80) are written in the on-disk format the real stack reads
(1-minute index parquet, one option parquet per expiry on a 50-point strike grid with a skewed smile and 0.05 tick rounding, a participant-OI date directory used only as a trading calendar).
Volatility is deterministic (sinusoid + step) so that the generator-side conditional expectation F = E[V | information at t] is closed-form; returns carry Gaussian diffusion, a daily
jump (p = 0.02, lognormal size) and overnight/weekend variance. The real `build_surface`, `build_iv_rv` and `build_annualization_audit` run on the files, and the EXP/hybrid aligned rows are compared
row-by-row (`obs_id`) with the generator truth (V, F, W, IV), and the nine Gate 1 estimands from pipeline rows are compared with the same estimands from the truth rows.

### 2.2 Tier B — calibrated synthetic coverage/dependence study (`synth_process.py`, `coverage_study.py`)
Process (all parameters in `DGP`): six-state Markov volatility regimes with an *exact* stationary law (Metropolis neighbour kernel + stay probability + uniform regime-shift jumps),
lognormal variance noise, daily jumps, four intraday variance segments (Dirichlet), overnight and weekend variance, random holidays, weekly Thursday expiries with windows of up to 14 days
(so consecutive windows **overlap**), exact conditional expectation F through the n-step transition matrix, and implied total variance W = c·F·exp(persistent AR(1) mispricing + row noise − mean).
Each history has 260 expiries (≈29.5 rows/expiry). **Truth** for every estimand is the same estimand evaluated on a 40,000-expiry long simulation of the same process (population value).
Scenarios (300 histories each, 500 bootstrap draws per interval, fixed seeds):

| id | description |
|---|---|
| B0_calibrated | calibrated process: regime shifts, jumps, heavy tails, persistent mispricing, overlapping windows (primary) |
| B1_strong_dependence | much more persistent vol regimes (stay 0.97) and mispricing (phi 0.93): a stress test |
| B2_no_overlap_iid | i.i.d. vol states, no mispricing persistence, windows ≤ 5.5 days: no overlap, no cross-expiry dependence |
| B4_overlap_only | i.i.d. vol states, no persistence, ≤ 14-day windows: dependence arises only from overlapping windows |
| B3_extreme_tails | frequent large jumps and fatter-tailed variance noise |

Methods compared for every estimand: `row_iid` (rows resampled as if independent), `date_block10` (Stage 2C), `expiry_iid` (cluster bootstrap over expiries), and expiry **moving-circular-block**
bootstraps with b = 2, 5, 8. Coverage reported with Wilson 95% intervals; flags `in_target_band_93_97`, `consistent_with_band` (Wilson interval intersects the band), `undercovers` (Wilson upper < 0.93).
Dev-vs-holdout contrast (Gate 2 method, calendar split, 170 dev / 90 holdout expiries): under a **true null** and under a **true shift** (holdout implied variance ×1.15; true difference from the long simulation).
Heterogeneity Wald test: placebo strata assigned at random **by row** and **by date** (true null), and a **planted** ×1.25 effect in the 1–3 d DTE stratum (power).

## 3. Tier A results (`t11a_*.csv`)

| world | c | max rel. error V | max IV error (vol pt) | median IV error | max ratio error | annualization identity error | max estimand difference |
|---|---|---|---|---|---|---|---|
| 1 | 1.00 | 9.6e-13 | 0.0060 | 0.0009 | 0.00096 | 8.9e-12 | 0.0010 |
| 2 | 1.30 | 8.9e-13 | 0.0057 | 0.0008 | 0.00107 | 7.8e-12 | 0.0006 |
| 3 | 0.80 | 9.8e-13 | 0.0056 | 0.0008 | 0.00091 | 7.3e-12 | 0.0011 |

All stated criteria hold (V ≤ 1e-9; IV ≤ 0.05 vol pt; estimands within 0.05 of the truth rows — actually ≤ 0.0011; session/calendar identity exact). The closed-form
generator expectation is validated by Monte Carlo: simulated 4.578e-5 vs closed form 4.599e-5 (−0.44%, 20,000 sessions).

**Deviation from the pre-stated criterion (recorded, not hidden).** The plan required "R3 within ±3 bootstrap SE of c". It fails (R3 − c = 3.4 SE for c = 1, 5.8 SE for c = 0.8) — but the
pipeline R3 equals the oracle R3 computed on the truth rows to 4e-6, so the pipeline is not at fault: a single-path **ratio of sums** Σ W / Σ V is not equal to c in expectation (Jensen term of a ratio
of random sums), and the bootstrap SE does not account for that. The criterion was mis-specified. It was replaced by (a) pipeline-vs-oracle estimand agreement ≤ 0.001 (met) and (b) a generator-level
unbiasedness check over 200 worlds: mean oracle ΣW/ΣV = 1.0195 (MC SE 0.0075) vs c = 1; the +2% is the ratio-of-random-sums Jensen term (`t11a_generator_check.csv`). I did not change any code to pass the original criterion.

## 4. Tier B results (`t11b_*.csv`, 300 histories per scenario; coverage SE ≈ 1.3 points)

### 4.1 Coverage of the 95% interval (all estimands; full table `t11b_coverage.csv`)
Mean coverage over estimands (and number flagged `undercovers`):

| scenario | row_iid | date_block10 | expiry_iid | expiry_block2 | expiry_block5 | expiry_block8 |
|---|---|---|---|---|---|---|
| B0 calibrated | 0.334 | 0.897 | 0.843 | 0.901 | **0.920** | 0.916 |
| B1 strong dependence | 0.158 | 0.717 | 0.609 | 0.720 | 0.829 | 0.849 |
| B2 no overlap, iid | 0.547 | 0.915 | 0.928 | 0.923 | 0.923 | 0.920 |
| B4 overlap only | 0.331 | 0.899 | 0.874 | 0.914 | 0.923 | 0.921 |
| B3 extreme tails | 0.282 | 0.897 | 0.861 | 0.906 | 0.921 | 0.924 |

Block-5 coverage in B0 by estimand: S1 session 0.937, S2 session 0.940, S1 calendar 0.937, S2 calendar 0.947 (all in 93–97%); R1 0.917, R1e 0.917, R2 0.893, R2e 0.887, R3 0.910 (below band).
Bias of the point estimates is small (≤ 0.02 vol pt for spreads, ≤ 0.009 for ratios), so the shortfall is interval width/shape, not bias.

* **Target 93–97%:** met only by the expiry-block procedures for spread-median estimands in the calibrated world; not met for ratio estimands, and not met anywhere in the stress world B1.
* **Dependence from overlapping windows (B4 vs B2):** with overlap only, `expiry_iid` drops from 0.928 (B2) to 0.874 and the block procedures recover to 0.914–0.923: overlapping windows alone induce enough cross-expiry dependence to hurt the iid-expiry bootstrap.
* **Block length b = 2, 5, 8:** b = 5 is best in the calibrated process; b = 8 is better under strong persistence (B1: 0.849 vs 0.720 for b = 2) — longer blocks are needed when persistence grows, and no tested block fully repairs it.
* **Row-level vs cluster-independent under-coverage:** row-iid 0.28–0.55 (0.16 under B1) versus cluster/block 0.84–0.93 (calibrated, B2–B4). Even with no dependence between expiries (B2), rows within an expiry are strongly correlated (ICC 0.71), so row-level intervals fail by a wide margin; cluster resampling fixes most but not all of it.
* Even i.i.d. B2 shows ~92–93% (not 95%) for several estimands: this is consistent with percentile bootstrap intervals for ratio/median statistics at 260 clusters being slightly narrow even without dependence (not separately diagnosed).

### 4.2 Dev-vs-holdout contrast (`t11b_split_contrast.csv`; B0)
* **True null (difference 0):** the interval for the difference covers 0 at 91.9–93.1% on average (b = 2/5/8: 0.919 / 0.931 / 0.926); false-rejection (p < 0.05) 7.4% / 6.0% / 6.7%. Spread medians 3.7–6.7% (near nominal); ratio statistics 5.7–12.7% (R3 worst, 10–13%).
* **True shift (holdout implied variance ×1.15):** the interval covers the true difference at 91.8–93.4%; power (p < 0.05) is high for the calendar spread (0.89–0.92), moderate for R1/R1e/R2/R2e (0.62–0.79) and low for R3 (≈ 0.49), since a ×1.15 shift on a noisy ratio is hard to detect. In B1 (strong dependence) the false-rejection rate under the null is 11–27% depending on estimand and block — inflated.
* A significant contrast is a statement about this synthetic process, not about tradability.

### 4.3 Heterogeneity Wald test (B0, 300 histories; `t11b_wald.csv`)
| test | statistic | rejection rate (α = 0.05) | Wilson 95% | near 5%? |
|---|---|---|---|---|
| placebo by row | S1 calendar | 0.037 | 0.021–0.064 | yes |
| placebo by row | R1 | 0.070 | 0.046–0.105 | yes |
| placebo by date | S1 calendar | 0.100 | 0.071–0.139 | **no (over-rejects)** |
| placebo by date | R1 | 0.083 | 0.057–0.120 | **no** (interval just above 5%) |
| planted ×1.25 in 1–3 d | S1 calendar | 1.000 | 0.987–1.000 | (power) |
| planted ×1.25 in 1–3 d | R1 | 1.000 | 0.987–1.000 | (power) |

## 5. Failures and what they imply for earlier gates (stated plainly)
1. **Ratio-estimand intervals from the expiry-block bootstrap are somewhat too narrow** (≈ 89–92% actual for nominal 95%). Gate 1/2 intervals for R1–R3 should be read as lower-bound uncertainty; the spread-median intervals are closer to nominal.
2. **The Stage 2C date-block bootstrap (block 10) is anti-conservative** here (≈ 90% mean, 85–94% by estimand).
3. **The `expiry_iid` bootstrap under-covers (≈ 84%)** whenever windows overlap or volatility persists; Gate 1 correctly reports block variants alongside it.
4. **Under strong persistence (B1) no tested procedure reaches nominal coverage.** If the real data are that persistent, all reported intervals are too narrow. The observed ACF(1) of the per-expiry median spread (0.128) is far below the B1 value (0.465), so B1 is a stress case rather than a description of the data.
5. **The Wald placebo-by-date false-rejection (8–10%) exceeds 5%:** the Gate 2 heterogeneity p-values are somewhat anti-conservative when strata are formed by dates.
6. The R3 contrast is the weakest (power ≈ 0.49 for a ×1.15 shift; false-rejection 10–13%).
Not fixed here: methodology of T1–T8 is unchanged (the instruction was to test, not to retune). Possible remedies (studentised/BCa bootstrap, longer block, wider blocks under persistence) are candidates for a later gate only if approved.

## 6. Limitations
* **Calibration is imperfect** (`t11b_calibration.csv`, 27 moments; observed value vs simulated 5–95%): inside for 18/27, outside for rows/expiry (24.96 vs 29.5), IV/RV 5th percentiles (within 5%), session spread (2.65 vs 2.98), ln-ratio skew (−1.28 vs −0.74: real data are more negatively skewed), ICC (0.575 vs 0.49: the simulation is **less** clustered), ACF(1) of expiry median spreads (0.128 vs 0.275: the simulation is **more** serially dependent), Ljung–Box p (0.15 vs 0.03). Mixed direction → coverage on the real data could be better or worse than here.
* The truth is a 40,000-expiry population value; its own MC error is small relative to coverage SE but not zero.
* B2 and B4 are isolating designs, not realistic worlds. B1 is a stress case.
* Percentile intervals only (no BCa/studentised); 500 draws; R = 300 histories (coverage SE ≈ 1.3 points, so differences of ≤ 2 points are not resolved).
* The implied side of Tier B is generated through W = c·F·(mispricing) directly (the IV solver is not re-run for 300 histories × 5 scenarios); the IV/price chain is covered by Tier A only.
* Tier A uses deterministic volatility; it validates plumbing/annualization/definitions, not statistical behaviour. Tier B validates statistics, not plumbing.
* Fixed Black-76 assumptions (r = 6.5%, no dividends), no bid/ask, no market microstructure.

## 7. Verification
* `tests/test_stage2d_gate3b.py` — 25 tests (process invariants incl. exact stationary law, nesting and slot-wise unbiasedness of F; hand-computed `to_aligned`, DTE bucket edges, Wilson values, coverage/flags/edges with fake results; worker-count determinism; a row-iid under-coverage property; Tier A generator expectation and a real-stack recovery; builder refusal outside `stage2d`).
* Mutation check (`validation/mutation_check_output.txt`): 15 deliberate faults in the process, to_aligned, Wilson, coverage flags, contrast sign, pipeline offsets and the output guard — all killed after strengthening four tests that initially let mutants survive (slot-0 segment weights, DTE bucket edge, closed-interval coverage, band edge, remainder offset).
* Independent recomputation (`validation/independent_gate3b_check.py`, output stored): realised variance recomputed from the raw 1-minute bars; IV by scipy `brentq` on written option prices (within 0.003 vol pt of the generator smile); every coverage, Wilson interval, flag, contrast coverage/rejection and Wald rejection recomputed from the per-history CSVs — all agree.
* Deterministic reproducibility, immutability and full pytest: see the final status in the accompanying hand-off message (`validation/rerun_determinism_output.txt`, `validation/immutability_check_3b_output.txt`).
