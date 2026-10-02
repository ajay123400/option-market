# Stage 2D — Gate 2: stability and sensitivity (T5–T8)

Status: **measurement and statistics only.** No strategy, signal, P&L, threshold search or Fyers link; no Stage 2A/2B/2C or Gate 1 file is modified (`validation/immutability_check_output.txt`).
Design frozen in `PREREGISTRATION.md` before any Gate 2 number existed (SHA-256 `108d3375…88e8` recorded in `run_metadata.json`). Baseline for this gate: Gate 1 commit `f2a9a2f` (parent `3b2a821`, the pushed Stage 2C).

> **Reading guide.** Intervals use the expiry moving-block bootstrap (b = 5, 5,000 replications, fixed seeds); p-values are percentile-bootstrap or Wald-bootstrap p-values, Holm-adjusted inside the two pre-declared families.
> A small p-value or an interval excluding the null describes this historical sample under stated dependence assumptions. It is **not** evidence of a tradable edge, of out-of-sample stability, or of the absence of selection effects (T9, not yet done).
> **The holdout was displayed in Stage 2C and Gate 1; here it is "previously viewed", never "out-of-sample".** Only expiries after 2026-09-30 would be untouched.

Population (unchanged): EXP / hybrid, 6,490 rows, 260 expiries; the builder reproduces the committed Stage 2C baseline at 1e-9 before running.

---

## 1. T5 — development vs holdout (family A: 4 pre-declared contrasts)

Clusters = expiries within a split (dev 170, holdout 92; 2 expiries straddle 2025-01-01 and appear in both). Independent block bootstraps per split; difference = holdout − dev.

| Estimand | Dev | Holdout | Holdout − dev | 95% interval | p (bootstrap) | p (Holm, family A) |
|---|---:|---:|---:|---|---:|---:|
| S1 calendar spread | 1.889 | 1.686 | −0.203 | −0.853 – +0.494 | 0.512 | 1.00 |
| S2 calendar (median of expiry medians) | 1.894 | 1.747 | −0.146 | −0.951 – +0.585 | 0.675 | 1.00 |
| R1 geometric-mean ratio | 1.297 | 1.280 | −0.017 | −0.199 – +0.168 | 0.877 | 1.00 |
| R2 median ratio | 1.367 | 1.378 | +0.011 | −0.147 – +0.160 | 0.912 | 1.00 |
| *(not in family A)* S1 session spread | 2.942 | 2.141 | −0.800 | −1.617 – +0.009 | 0.052 | – |
| *(not in family A)* R3 ratio of sums | 1.177 | 1.011 | −0.167 | −0.457 – +0.210 | 0.378 | – |

* **No contrast is distinguishable from zero** on the calendar basis or for the ratios. The intervals are wide (about ± 0.7 vol pts, ± 0.18 in the ratio): this is absence of evidence of a difference, not evidence of equality.
* The session-basis spread difference (−0.80) is borderline and **depends on the block length**: p = 0.011 (b = 2), 0.052 (b = 5), 0.080 (b = 8) (`block_length_sensitivity.csv`). Calendar-basis and ratio contrasts stay non-significant at b = 2 and 8.
* **Level vs spread shift.** The holdout has more low-IV observations (Q1: 44% vs 27%; Q4: 12.5% vs 19.7%). Re-weighting the holdout to the dev IV-quartile composition: S1 calendar 1.987 vs 1.889 (difference +0.098, −0.530 – +0.819); S1 session −0.361 (−1.095 – +0.538); R1 +0.019 (−0.208 – +0.239).
  About half of the raw session-basis difference (−0.80 → −0.36) is a composition effect. Within every IV quartile the dev-vs-holdout difference interval contains 0 (the Q4 holdout cell holds 31 expiries, just above the 30 minimum).

## 2. T6 — strata and heterogeneity (family B: 12 pre-declared tests; `t6_*.csv`)

Point estimates and expiry-block intervals for S1 calendar and R1 (strata under 30 expiries are point-only):

| Dimension | Level | Expiries | S1 calendar | R1 |
|---|---|---:|---|---|
| DTE | 1–3 d | 253 | 2.34 (2.04–2.76) | 1.368 (1.300–1.441) |
| | 3–7 d | 259 | 1.72 (1.35–2.11) | 1.291 (1.192–1.383) |
| | 7–14 d | 257 | 1.70 (1.39–2.03) | 1.264 (1.179–1.345) |
| Snapshot time | 10:00 / 13:00 / 15:00 | 260 each | 1.87 / 1.82 / 1.74 (1.61–2.21 / 1.56–2.21 / 1.46–2.05) | 1.293 / 1.296 / 1.284 |
| IV quartile (dev cut-offs) | Q1 low / Q2 / Q3 / Q4 high | 149 / 194 / 168 / 114 | 1.03 / 1.85 / 2.88 / 4.05 | 1.197 / 1.263 / 1.374 / 1.418 |
| Recent-20-session RV regime | low / mid / high | 109 / 109 / 100 | 1.26 / 2.02 / 2.73 | 1.198 / 1.316 / 1.358 |
| | unknown | 28 | 2.41 (point only) | 1.389 |
| Expiry year | 2021 (point only) | 14 | 1.93 | 1.228 |
| | 2022 / 2023 / 2024 / 2025 / 2026 | 52 / 50 / 52 / 53 / 39 | 2.98 / 1.56 / 1.69 / 1.81 / 1.49 | 1.353 / 1.304 / 1.249 / 1.364 / 1.181 |
| Weekend in window | no / yes | 254 / 257 | 2.29 / 1.67 | 1.360 / 1.265 |

| Heterogeneity (bootstrap Wald) | S1 calendar p (Holm) | R1 p (Holm) |
|---|---|---|
| DTE bucket | 0.005 (**0.042**) | 0.027 (0.145) |
| snapshot time | 0.024 (0.145) | 0.002 (**0.020**) |
| IV quartile | 0.0002 (**0.002**) | 0.053 (0.211) |
| recent-RV regime | 0.001 (**0.012**) | 0.168 (0.475) |
| expiry year (2022–2026) | 0.158 (0.475) | 0.662 (0.662) |
| weekend in window | 0.0006 (**0.007**) | 0.018 (0.127) |

* Five of the twelve tests survive Holm adjustment (bold). Equal-strata is **not** rejected across expiry years for either estimand.
* **Statistical vs practical size.** The snapshot-time R1 test is "significant" for differences of about 0.01 in the ratio (1.284–1.296): the three snapshots share the same expiries, so their contrasts are tightly correlated and tiny differences have small bootstrap variance. The spread differences across DTE and weekend windows are 0.6–0.7 vol pts.
* **Interpretation caution.** The IV-quartile and recent-RV strata condition on variables that enter the spread mechanically (a high IV mechanically widens IV − RV); DTE, weekend and time-of-day patterns overlap with the annualization effect found in the Stage 2C audit. These are descriptive; no mechanism is claimed.
* **Mincer–Zarnowitz (descriptive)**: RV_cal = a + b·IV, all rows: a = 1.45 (−0.01–3.08), **b = 0.80 (0.68–0.92)**; on logs b = 0.89 (0.80–0.96). Dev b = 0.85 (0.70–0.98), holdout 0.72 (0.51–1.00). Noise in IV attenuates b; no forecasting claim and no correction for that.
* **Expiry-day stratum** (through the F1/F5 hybrid targets only; the IV belongs to an expiring contract and the RV to the sessions after it, so this is a horizon-mismatched comparison): S1 calendar F1 8.07 (6.77–9.22) vs 2.45 (2.10–2.83) on other days; F5 9.02 (7.87–10.16) vs 2.29 (1.95–2.66); Wald p = 0.0002 for both. Expiry-day wing IVs are resolution-limited (Stage 2C), and these rows are never pooled.

## 3. T7/T8 — convention and filter sensitivity (31 specifications; `t7_t8_specification_curve.csv`)

Median spread and ratio estimates (95% intervals in the CSV; nothing is selected by outcome):

| Specification | n | S1 calendar | S1 session | R1 | R3 |
|---|---:|---:|---:|---:|---:|
| **baseline** | 6,490 | 1.805 | 2.651 | 1.291 | 1.116 |
| risk-free rate 5.5% / 7.5% (Stage 2A re-run) | 6,490 / 6,489 | 1.802 / 1.806 | 2.649 / 2.653 | 1.291 / 1.291 | 1.115 / 1.116 |
| session days 250 / 256 | 6,490 | 1.805 | 2.693 / 2.580 | 1.291 | 1.116 |
| calendar days 365.25 | 6,490 | 1.806 | 2.655 | 1.291 | 1.116 |
| RV sampling 5 min / 15 min | 6,490 | **1.672 / 1.714** | 2.571 / 2.542 | 1.274 / 1.273 | 1.099 / 1.082 |
| partial-session intraday-variance profile | 6,490 | 1.805 | 2.579 | 1.291 | 1.116 |
| intraday-only RV (subset; overnight excluded) | 6,490 | undefined | 4.438 | 1.920 | 1.906 |
| forward gate 3 / 8 / 12 bps | 6,373 / 6,511 / 6,517 | 1.796 / 1.810 / 1.811 | 2.650 / 2.661 / 2.662 | 1.291 / 1.292 / 1.292 | 1.112 / 1.119 / 1.121 |
| stale-quote age 2 / 15 min | 6,482 / 6,491 | 1.806 / 1.804 | 2.660 / 2.650 | 1.291 / 1.291 | 1.117 / 1.116 |
| DTE ≤ 7 d | 2,916 | **2.013** | 3.815 | 1.325 | 1.102 |
| DTE ≤ 21 d (strike coverage biased beyond 14 d) | 9,084 | 1.746 | 2.335 | 1.283 | 1.125 |
| exclude expiry year 2021 / 2026 | 6,182 / 5,505 | 1.802 / 1.855 | 2.617 / 2.824 | 1.294 / 1.312 | 1.113 / **1.172** |
| drop snapshot 10:00 / 13:00 / 15:00 | 4,326 / 4,321 / 4,333 | 1.772 / 1.800 / 1.845 | 2.456 / 2.646 / 2.850 | 1.290 / 1.289 / 1.295 | 1.112 / 1.113 / 1.123 |

The 3 × 3 factorial (rate × sampling) shows no interaction: the rate cells differ by ≤ 0.003 in S1 calendar at every sampling step.

* **Rate**: ±1 percentage point changes S1 calendar by ≤ 0.003 vol pts and leaves ratios unchanged to 3 decimals. Independently, a Black-76 re-solve of 300 reliable points in each of the 5.5% / 6.5% / 7.5% runs reproduces the stored IVs to ≤ 1.2e-11.
* **Annualization conventions** move only the session-basis spread (250 → 2.693, 256 → 2.580); the calendar basis and the ratios are invariant; 365.25 changes S1 calendar by +0.001 (and leaves implied total variance unchanged).
* **RV sampling frequency is the most consequential convention tested for the headline spread**: with 5- or 15-minute returns, realized variance is about 1.3–3% higher (depending on the estimator) and the calendar spread falls by 0.09–0.13 vol pts (−5% to −7%). Higher variance at coarser sampling is the signature of positive serial correlation of 1-minute index returns, so the 1-minute baseline understates RV slightly. The conclusion that the spread and the ratio exceed their nulls is unchanged.
* **DTE cap** moves the level most (≤ 7 d: 2.013; ≤ 21 d: 1.746), as the Stage 2C DTE pattern implies; the ≤ 21 d row mixes in the biased-coverage region.
* **Excluding 2026** raises the ratio of sums to 1.172, whose interval (1.053–1.306) then excludes 1; in all other specifications the ratio-of-sums interval contains 1.

### 3.1 Pre-declared robustness reading (descriptive; not a decision rule)
| Estimand | Specs | Baseline | Min – max across specs | Intervals excluding the null | Sign unchanged | Widest departure |
|---|---:|---:|---|---:|---|---|
| S1 calendar | 30 | 1.805 | 1.671 – 2.013 | 30 of 30 | yes | DTE ≤ 7 d (+0.208) |
| R1 | 31 | 1.291 | 1.272 – 1.325 (1.920 incl. the intraday-only subset) | 31 of 31 | yes | intraday-only subset (+0.629; subset, overnight excluded) |
By the pre-declared wording both are "not sensitive to the tested conventions" **in sign**; magnitudes move by −7% to +12% (S1 calendar) and −1.5% to +2.6% (R1) across all non-subset specifications. R3 is not robust in that sense: its interval contains 1 in 29 of 31 specifications.

## 4. Forking-paths ledger (`forking_paths_ledger.csv`)
17 analysis choices recorded with the stage, when decided and whether results had been seen. Three were decided **after** results were seen: singling out the hybrid measure for the aligned comparison, introducing the calendar basis and total-variance ratio (after the session-basis results exposed the time-basis mismatch), and choosing EXP/hybrid as the Stage 2D population; the partial-session weighting is "partly". Each alternative is evaluated in T7/T8 (or flagged), and none changes a sign.

## 5. Deviations from the pre-registration
1. Mincer–Zarnowitz and the within-IV-quartile contrasts use 2,500 replications (not 5,000).
2. The block-length sensitivity (b = 2, 8) covers the T5 contrasts and the pooled baseline estimates, not every stratum or specification.
3. The T7 "RV measure" row is reported with the intraday-only subset included in the min/max reading; both ranges are shown above.
No other deviation. Seeds and tables are reproducible: re-running the builder reproduced every CSV byte-for-byte.

## 6. Limitations
* The holdout is previously viewed; the T5 contrasts have little power (interval half-widths ≈ 0.7 vol pts, ≈ 0.18 in R1) and cannot show equality of the periods.
* Wald tests use a block bootstrap from 260 expiries; with 5 + 5 + 3 strata many tests are correlated; Holm controls family-wise error only inside each family.
* Strata overlap (DTE, weekend, time of day) and two of them (IV quartile, RV regime) condition on variables tied to the spread.
* Variant pipelines re-use the committed builders, so any bias in strike coverage, last-trade pricing, forward filtering or the missing bid/ask is common to every specification; this gate cannot detect it (T9, T10).
* The "robustness reading" is about sign across the tested grid, not about regimes that are not in the data.
* No selection-on-outcome audit, quote-noise scenario, whole-pipeline synthetic recovery or claim ledger yet (T9–T12).

## 7. Tests and validation
* `tests/test_stage2d_gate2.py` — 32 tests: Holm known values/monotonicity, bootstrap-p floor, split-contrast recovery of a known shift and null behaviour, determinism, small-split guard, straddling expiries, weighted median, post-stratification (re-weights to the target; removes a pure composition shift), within-level point-only flag, strata views, small-stratum point-only rule, Wald calibration (≤ 6/40 false rejections) and power (≥ 9/10), covariance between strata sharing expiries, manual Wald statistic, Mincer–Zarnowitz slope recovery, alt-RV k = 1 equals the Stage 2C target exactly for 10:00/13:00/15:00, k = 5/15 hand values, no look-ahead (violent past inside/before the snapshot bar leaves V unchanged; a later move changes it), flat profile ≡ uniform weighting, annualization and 365.25 conversions (IV² T invariant), unavailable windows not filled, the 23 pre-registered specification names and the filters (including the ≤ 7 d boundary), robustness-reading logic, refusal to write outside `stage2d`, restoration of the DTE constant after a failed variant, cached variants not rebuilt, end-to-end determinism and the pre-registration SHA.
  Ten mutants were applied; nine were caught immediately and the surviving `dte ≤ 7` boundary mutant was caught after the test was strengthened.
* `validation/independent_gate2_check.py` (no package imports; csv/pandas/scipy only) — dev/holdout estimates and differences, Holm (both families), post-stratified estimates, all 44 stratum estimates and row counts (max difference 2.7e-13), Mincer–Zarnowitz slope and intercept, the F5 expiry-day stratum, **5- and 15-minute RV recomputed from the raw 1-minute bars for all 6,490 rows** (S1 calendar and R1 agree to 1e-9), and a Black-76 re-solve of 900 points across the three rate runs: **all agree**.
* `validation/immutability_check_output.txt` — baseline `f2a9a2f` unchanged, 0 tracked files modified, every new file under Stage 2D paths.
* Full suite: **1183 passed** (1151 + 32), 0 failed.

## 8. Files and reproduction
Code: `optionsengine/research/stage2d/{contrasts,strata,alt_rv,spec_curve,build_gate2}.py`; tests: `tests/test_stage2d_gate2.py`; outputs: this directory (13 tables, `PREREGISTRATION.md`, `IMPLEMENTATION_PLAN.md`, `validation/`); gitignored variant runs under `variants/` (≈ 1 GB: Stage 2A re-runs for 5.5% / 7.5% and Stage 2C pipelines for each variant).
```
python -m optionsengine.research.stage2d.build_gate2 --out research_output/stage2d/gate2     # ~15 min for the variant pipelines (cached afterwards), ~4 min for the tables
python research_output/stage2d/gate2/validation/independent_gate2_check.py
python research_output/stage2d/gate1/validation/immutability_check.py f2a9a2fb37a30a5307676b2ba46edff5da43539f
python -m pytest -q
```
