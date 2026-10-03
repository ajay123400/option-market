# Stage 2D — Gate 2 pre-registration (T5–T8): frozen before any Gate 2 result is computed

Written after Gate 1 was committed (`f2a9a2f`, parent `3b2a821`) and before any Gate 2 code or number exists. Anything not listed here is **exploratory** and will be labelled so.
Measurement only: no strategy, signal, P&L, threshold search or Fyers link. A confidence interval or p-value never means "tradable edge".

## 1. Frozen population and estimands
* Population: Stage 2C expiry-aligned sample, hybrid RV, 6,490 rows / 260 expiries (baseline reproduced at 1e-9 by the builder before anything else runs).
* Headline estimands (Gate 1 definitions): **S1_cal** (all-observation median spread, calendar basis), **S1_sess** (session basis), **R1** (geometric-mean total-variance ratio), **R2** (median ratio), **R3** (ratio of summed variances, reported but never headline).
  Per-expiry versions (S2, R1e, R2e) are reported alongside.
* Uncertainty: expiry moving-block bootstrap, **b = 5** (Gate 1 rule), 5,000 replications, percentile 95% intervals, base seed 20260101. Sensitivity to b ∈ {2, 8} is reported, never used to pick an answer. < 30 clusters ⇒ no interval.
* Dev = snapshot day ≤ 2024-12-31, holdout ≥ 2025-01-01 (Stage 2A split). **The holdout was already displayed in Stage 2C and Gate 1: it is "previously viewed", not out-of-sample.** Data after 2026-09-30 (expiries not yet in the sample) is reserved as the only untouched test; the frozen specification below is the one to be applied to it.
* Regime cut-offs: the Stage 2C development-only values in `research_output/stage2c/regime_cutoffs.json` (IV quartiles 11.68/14.50/18.49; recent-20-session RV terciles 10.77/13.45), used unchanged.

## 2. Test families and multiplicity
* **Family A (T5)**: 4 contrasts, holdout − dev, for S1_cal, S2_cal, R1, R2. Holm-adjusted within the family; unadjusted values also shown.
* **Family B (T6)**: heterogeneity (equal-strata) tests for 6 dimensions × 2 estimands (S1_cal, R1) = 12 tests: DTE bucket (1–3 d / 3–7 d / 7–14 d), snapshot time, IV quartile, recent-RV regime, expiry year, weekend-in-window. Holm within the family; unadjusted shown.
* T7/T8 specification curve: **no hypothesis tests**; point estimates and intervals for every specification, nothing selected by outcome.
* Everything else (Mincer–Zarnowitz slopes, expiry-day stratum, per-stratum intervals) is descriptive.

## 3. T5 — dev vs holdout (pre-specified design)
* Difference of each estimand between holdout and dev; clusters = expiries within split (an expiry straddling 2025-01-01 contributes one cluster to each split; their number is reported); expiry-block bootstrap run independently in the two splits (adjacent at the boundary: ignored, stated).
* Level-adjusted contrast: holdout re-weighted to the dev IV-quartile composition (post-stratification, dev cut-offs) and the within-quartile contrasts; purpose: separate a level shift (holdout IV ≈ 1.6 pts lower) from a spread shift.
* Forking-paths ledger: every analysis choice made in Stages 2A–2C and Gate 1 with the date of decision and whether results had been seen; each alternative is evaluated in T7/T8.

## 4. T6 — strata
* Per-stratum estimates with expiry-block intervals; strata with < 30 expiries get point estimates only.
* Heterogeneity test: bootstrap Wald test of equal stratum estimates (contrasts against the first level, joint expiry-block bootstrap covariance, bootstrap-centred null distribution).
* Descriptive Mincer–Zarnowitz regression RV = a + b·IV (calendar-basis RV vs IV; also log scale), slope interval by expiry-block bootstrap; errors-in-variables attenuation is stated as a caveat. No forecasting model.
* Expiry-day rows have no expiry-aligned target; they are examined through the F1/F5 hybrid calendar-basis targets only (horizon mismatch stated) and kept out of every pooled statistic.
* IV-quartile and RV-quartile strata are conditioned on a variable that enters the spread mechanically; they are descriptive.

## 5. T7 — convention sensitivity (one-at-a-time around the baseline, plus a small factorial)
| Dimension | Values | How |
|---|---|---|
| risk-free rate | 5.5%, **6.5%**, 7.5% | Stage 2A builder re-run with `--rate` into a Stage 2D directory, then the Stage 2C and audit builders (no Stage 2A/2B/2C file modified) |
| session annualization days | 250, **252**, 256 | arithmetic on stored V and session-equivalents |
| calendar days per year | **365**, 365.25 | arithmetic (T and span) |
| RV sampling | **1-min**, 5-min, 15-min returns | new Stage 2D RV recomputation from the 1-min spot bars via read-only use of the Stage 2B session index |
| partial-session weighting | **uniform**, development intraday variance profile | session-equivalents recomputed (Stage 2C exploratory script logic, re-implemented in Stage 2D) |
| RV measure | **hybrid**, intraday-only (subset, labelled not interval-complete) | existing columns |
Factorial: rate × RV sampling (9 cells). Each specification reports n, expiries, S1_cal, S1_sess, R1, R2, R3 and intervals.

## 6. T8 — filter and sample restrictions
Forward-gate 3 / **5** / 8 / 12 bps and stale-quote age 2 / **5** / 15 min (existing Stage 2A sensitivity smiles are reused, then the Stage 2C and audit builders); DTE cap ≤ 7 / **≤ 14** / ≤ 21 (the ≤ 21 variant is flagged: Stage 2A strike coverage beyond 14 DTE is biased; run with a runtime override of the Stage 2C DTE constant, no file edit); exclude the first and last calendar year; exclude 10:00 / 13:00 / 15:00 one at a time. The sample size of every variant is reported.

## 7. Pre-declared robustness reading (descriptive; not a decision rule or a threshold to tune)
For S1_cal and R1: report the minimum and maximum point estimate across all T7/T8 specifications, the share of specifications whose interval excludes the null (0 for spreads, 1 for ratios), and the specification with the widest departure from baseline. A finding is described as **"not sensitive to the tested conventions"** only if its sign is unchanged in every specification; otherwise the sensitive dimension is named. These words never upgrade a finding beyond the tiers below.

## 8. Evidence tiers (carried to the claim ledger in Gate 3)
* Tier 0 — descriptive association in this sample.
* Tier 1 — statistical evidence: clustered/block interval excludes the null, sign stable across dev/holdout, years and specifications, no sign of selection on the outcome (T9).
* Tier 2 — tradable edge: **not assessable here** (no bid/ask, costs, margin, execution); no claim is made.

## 9. Reproducibility
Fixed seeds; builders refuse output paths outside `stage2d`; the pre-registration file's SHA-256 is recorded in the Gate 2 run metadata; large variant outputs are gitignored.
Deviations from this document must be listed in the Gate 2 report under "Deviations".
