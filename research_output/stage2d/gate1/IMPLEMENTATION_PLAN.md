# Stage 2D — Gate 1 (statistical inference core, T1–T4): implementation plan

Written before any Gate 1 code. Baseline: HEAD `3b2a8218e5aeb8b394705e3f1c3b7b9290211403`. Scope: T1–T4 only. Measurement/statistics only: no strategy, P&L,
signal, Fyers link, threshold search. Stage 2A/2B/2C code and outputs are read, never edited. All new files live under
`optionsengine/research/stage2d/`, `tests/test_stage2d_*.py` and `research_output/stage2d/gate1/`.

## Data and population
Input: `research_output/stage2c/annualization_audit/aligned_observations.csv` (local, gitignored; produced by the committed Stage 2C pipeline),
restricted to `horizon == EXP` and `measure == hybrid` (n = 6,490 rows, 260 expiries, 1,224 snapshot dates). The build script first
reproduces the committed Stage 2C baseline (median session spread 2.651, calendar spread 1.805, median total-variance ratio 1.371, n 6,490)
from `original_vs_aligned.csv` / `expiry_total_variance.csv` and aborts on a mismatch. No new IV, RV or target is computed.

## Estimands (reported separately, never merged)
For each basis b in {session, calendar} spread = IV − RV_b (vol points):
* **S1** all-observation median of spread (the Stage 2C statistic);
* **S2** median of per-expiry medians of spread (each expiry counts once).

Total-variance ratio r = W_imp / V (valid rows only, V > 0):
* **R1** geometric mean of r over all rows, exp(mean ln r);
* **R1e** expiry-weighted geometric mean, exp(mean over expiries of the per-expiry mean ln r);
* **R2** median of r over all rows;
* **R2e** median of per-expiry median r;
* **R3** ratio of summed total variances, Σ W_imp / Σ V over all rows (dominated by long, high-variance windows by construction).

Each answers a different question; none is preferred. A confidence interval excluding the null (spread 0, ratio 1) is a statement about a
historical association in this sample under stated dependence assumptions, not evidence of tradability.

## Independence unit and overlapping windows
* The unit of inference is the **expiry** (a snapshot's outcome is the path to its expiry, so all ≈26 snapshots of one expiry share it;
  measured ICC ≈ 0.58, design effect ≈ 15).
* Expiries are **not** independent of each other: a ≤14-day window of expiry E overlaps the window of the next weekly expiry. Three handling strategies
  are compared, each with its own assumption:
  1. *expiry-iid cluster bootstrap* — resample whole expiries with replacement; assumes no dependence between expiries (known violation: overlap);
  2. *expiry-block bootstrap* (moving circular and stationary) — resample runs of consecutive expiries in calendar order, block length from
     Politis–White (2004; Patton–Politis–White 2009) on the per-expiry series, with a sensitivity grid; assumes weak stationarity of the expiry series
     and dependence shorter than the block;
  3. *date-block bootstrap* (the Stage 2C method, block 10 dates) — resample runs of consecutive snapshot dates; same row-level statistics as
     Stage 2C; the per-expiry estimands S2/R1e/R2e are undefined there and not reported.
  A naive *row-iid bootstrap* is shown only as a reference that treats the 6,490 rows as independent (expected to be far too narrow).
* The overlap is measured, not assumed: per-expiry window [first snapshot, expiry close], overlap with the following expiry's window, and the
  number of expiries sharing each calendar day. A **non-overlapping subsample** (one pre-specified 10:00 snapshot per expiry, the largest
  T ≤ 6 days; an expiry is dropped if its window overlaps the previously kept window) gives an estimate whose clusters overlap by construction
  far less than in the full sample. The rule uses only DTE and time, never outcomes.
* Percentile intervals (2.5/97.5), fixed seeds, 5,000 replications for headline tables, 2,000 for sensitivity grids.
* Small-cluster guard: any procedure with fewer than 30 clusters (or a block longer than the number of clusters) raises instead of returning an interval.

## T2 diagnostics
ICC (one-way ANOVA, unbalanced), design effect and effective n; ACF and Ljung–Box on the per-expiry series; Politis–White block length;
Newey–West (Bartlett) long-run SE of the mean of the per-expiry series; window-overlap map; CI-width vs block-length sensitivity.

## T3 ratio inference
Estimands R1–R3 with the intervals above; exact two-sided sign test on per-expiry median ratios (independence assumed — stated); expiry-block sign-flip
permutation test of H0: E[ln r] = 0 on the per-expiry mean log ratio (blocks of consecutive expiries are flipped together); HAC t-statistic. Rows with V = 0
are counted and excluded from log/ratio statistics only.

## T4 influence
Leave-one-expiry-out (260 estimates per estimand) with the delete-1 jackknife SE; delete-block jackknife; leave-one-year-out (by expiry year);
removal of the k most influential expiries (k = 1, 5, 10, 20, ranked by their leave-one-out effect on the same estimand) compared with removal of k random expiries;
10% trimmed and 5% winsorized means of the per-expiry series. Influence ranking is descriptive; it is not an exclusion rule.

## Tests and validation (all deterministic)
Hand-computed estimands; seed reproducibility and row-order invariance; small-cluster and oversized-block failures; ACF/Ljung–Box/ICC/Politis–White/Newey–West on
known processes; synthetic dependence studies (iid, ICC-only, AR(1) across clusters): coverage of row-iid vs expiry-iid vs block procedures; jackknife exactness;
overlap and subsample rules on synthetic windows; baseline reproduction; no-look-ahead (inputs are Stage 2C outcome/feature columns only; no spot data, no new fields); immutability of Stage 2A/2B/2C files.
An independent pure-Python recomputation of the estimands, leave-one-out and ACF is stored with its output.

## Not in Gate 1
T5–T12 (dev-vs-holdout inference, strata/heterogeneity, specification curve, filter sensitivity, selection audit, quote-noise scenarios, full-pipeline synthetic recovery, claim ledger).
