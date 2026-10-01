# Stage 2D — Gate 1: statistical inference core (T1–T4)

Status: **measurement and statistics only.** Gate 1 re-expresses the uncertainty of the Stage 2C expiry-aligned findings under explicit dependence handling.
It builds no strategy, signal, P&L or Fyers link, and it changes no Stage 2A/2B/2C code or output. Baseline HEAD
`3b2a8218e5aeb8b394705e3f1c3b7b9290211403`; `validation/immutability_check_output.txt` confirms 0 tracked files changed.

> **How to read the intervals.** An interval that excludes the null (spread 0, ratio 1) says the *historical sample difference* is unlikely to be zero under the stated
> dependence assumptions. It does **not** show the difference is stable out of sample, free of selection effects, or tradable; those questions belong to later gates
> (T5–T12, not implemented) and, for tradability, to information this dataset does not contain (bid/ask, costs, execution, margin).

Population: Stage 2C expiry-aligned target, hybrid RV, **6,490 snapshot rows from 260 expiries (1,224 snapshot dates)**. The builder first reproduces the committed Stage 2C
baseline (n 6,490; median session spread 2.651; median calendar spread 1.805; geometric-mean ratio 1.291; median ratio 1.3705; tolerance 1e-9) and aborts on any mismatch.

---

## 1. Methods

### 1.1 Estimands (reported separately; none preferred)
| Id | Definition | What it answers |
|---|---|---|
| S1 | all-observation median of spread (IV − RV, vol points), session basis and calendar basis | the Stage 2C statistic; expiries with more snapshots weigh more |
| S2 | median of the 260 per-expiry median spreads | one vote per expiry |
| R1 | geometric mean of r = W_imp / V over rows, exp(mean ln r) | typical multiplicative gap |
| R1e | exp of the mean over expiries of the per-expiry mean ln r | R1 with one vote per expiry |
| R2 | median of r over rows | |
| R2e | median of the per-expiry median r | |
| R3 | ΣW_imp / ΣV over all rows | volume-weighted: long, high-variance windows dominate by construction |

Rows with V = 0 (none occur) would be excluded from R1/R1e/R2/R2e only; R3 uses every row's sums.

### 1.2 Independence unit and overlapping expiry windows
* **Unit = expiry.** All ≈ 25 snapshots of an expiry share the same realized path, so rows are not independent (ICC 0.50–0.62, §3).
* **Expiries overlap each other.** An expiry's ≤ 14-day window overlaps the next weekly expiry's: the mean window is 12.9 days, **46% of it (5.9 days) is shared with the next expiry**, each window overlaps a
  median of 2 (max 3) other expiries' windows. Dependence between neighbouring expiries is therefore expected and is handled in three ways, compared side by side:

| Method | Resamples | Assumption | Main limitation |
|---|---|---|---|
| row-iid (naive reference) | single rows | rows independent | ignores both within-expiry and cross-expiry dependence; shown only to quantify the damage |
| date-block (the Stage 2C method, block 10 dates) | runs of 10 consecutive snapshot dates | dependence shorter than ~10 dates; stationarity | ignores the expiry structure; per-expiry estimands undefined |
| expiry-iid cluster | whole expiries, with replacement | expiries independent | ignores the overlap between neighbouring expiries |
| **expiry moving-block (headline)** | circular runs of b = 5 consecutive expiries | dependence between expiries shorter than b; weak stationarity | block length is estimated from 260 points; percentile intervals can be biased for skewed statistics |
| expiry stationary bootstrap (mean b = 5) | circular runs of geometric length | same as above, smoother block-length choice | same |

Block length rule (fixed before looking at intervals): the largest rounded Politis–White (2004; Patton–Politis–White 2009) circular length over the three per-expiry series = **5**
(5.09 session spread, 2.53 calendar spread, 4.67 log ratio). A grid b ∈ {1,2,3,4,5,6,8,12} shows the sensitivity (§2.3). Percentile 95% intervals, fixed seeds, 5,000 replications (2,000 for grids).
Small-cluster guard: designs with < 30 clusters, or a block longer than the number of clusters, raise instead of returning an interval.

### 1.3 Non-overlapping subsample (pre-specified, outcome-free)
Per expiry keep the **10:00** snapshot with the largest T ≤ 6 days (windows ≈ 3.2 days); walk expiries in order and drop one whose window starts before the previously kept window ends.
Result: 259 of 260 expiries kept, windows do not overlap. It is a different stratum (10:00, ≈ 3-day windows), so its level is not comparable with the full sample; it checks whether the qualitative picture survives when
windows do not overlap.

---

## 2. T1 — clustered uncertainty

### 2.1 Point estimates and 95% intervals (vol points for S, ratio for R)
| Estimand | Estimate | row-iid (naive) | date-block (Stage 2C method) | expiry-iid | **expiry-block b=5** | expiry-stationary |
|---|---:|---|---|---|---|---|
| S1 session | 2.651 | 2.56 – 2.74 | 2.34 – 2.96 | 2.41 – 2.91 | **2.26 – 3.06** | 2.23 – 3.10 |
| S2 session | 2.523 | n/a | n/a | 2.27 – 2.69 | **2.23 – 2.88** | 2.21 – 2.89 |
| S1 calendar | 1.805 | 1.74 – 1.87 | 1.57 – 2.08 | 1.61 – 2.05 | **1.54 – 2.16** | 1.54 – 2.18 |
| S2 calendar | 1.860 | n/a | n/a | 1.56 – 2.12 | **1.53 – 2.22** | 1.53 – 2.22 |
| R1 geometric mean | 1.291 | 1.275 – 1.307 | 1.224 – 1.356 | 1.229 – 1.352 | **1.215 – 1.369** | 1.216 – 1.362 |
| R1e expiry-weighted | 1.290 | n/a | n/a | 1.228 – 1.351 | **1.214 – 1.367** | 1.217 – 1.360 |
| R2 median ratio | 1.371 | 1.356 – 1.385 | 1.307 – 1.430 | 1.310 – 1.425 | **1.301 – 1.437** | 1.303 – 1.431 |
| R2e median of expiry medians | 1.383 | n/a | n/a | 1.309 – 1.437 | **1.304 – 1.442** | 1.308 – 1.439 |
| R3 ratio of sums | 1.116 | 1.087 – 1.146 | 0.990 – 1.248 | 1.002 – 1.236 | **0.978 – 1.268** | 0.980 – 1.258 |

* Treating the rows as independent makes the intervals **4.8–5.0× too narrow** for S1 calendar and R2 (width 0.129 vs 0.616; 0.028 vs 0.136).
* The Stage 2C date-block interval is reproduced here (S1 calendar 1.57–2.08 vs the published 1.59–2.09; different seed/replications). The expiry-block interval is about **20% wider** (0.616 vs 0.507), i.e. the Stage 2C interval was somewhat optimistic but of the right order.
* The null (spread 0, ratio 1) lies outside every interval for S1, S2, R1, R1e, R2 and R2e under every cluster/block method. **R3's interval contains 1 under both block methods and the Stage 2C date-block method** (0.98–1.27, 0.98–1.26, 0.99–1.25) and only just excludes it under expiry-iid (1.002).
* The all-observation and per-expiry versions agree closely (S1/S2 calendar 1.805 vs 1.860; R1/R1e 1.291 vs 1.290; R2/R2e 1.371 vs 1.383); the session-basis pair differs by 0.13.

### 2.2 Summed versus typical ratio
R1/R2 (typical row or expiry) ≈ 1.29–1.38 but R3 (ratio of summed variances) = 1.116. R3 weights each window by its realized variance, so a few large-variance windows with r ≤ 1 pull it towards 1 (§5).
These are different, both valid, summaries of the same rows.

### 2.3 Block-length sensitivity (all estimands in `block_length_sensitivity.csv`)
S1 calendar interval width: expiry-block b = 1/2/3/4/5/6/8/12 → 0.457/0.540/0.539/0.569/0.599/0.615/0.646/0.667; date-block b = 1/5/10/20/40 → 0.244/0.438/0.513/0.594/0.664.
Widths rise quickly until b ≈ 5 and slowly after; the date-block width at 20–40 dates matches the expiry-block width at b ≈ 5–12, as expected because an expiry spans ≈ 10 sessions.
R3 and R1 behave the same way (R3: 0.234 at b = 1 → 0.273 at b = 5 → 0.296 at b = 8).

---

## 3. T2 — dependence diagnostics (`dependence_diagnostics.csv`, `acf_expiry_series.csv`, `window_overlap.csv`)

| Series (260 expiries) | ICC by expiry | design effect | effective n (of 6,490 rows) | ACF lag 1 | Ljung–Box p (10 lags) | PW circular b | Newey–West variance inflation (4 / 8 lags) |
|---|---:|---:|---:|---:|---:|---:|---|
| spread, session basis | 0.505 | 13.1 | 495 | 0.232 | 0.004 | 5.09 | 1.56 / 1.86 |
| spread, calendar basis | 0.575 | 14.8 | 439 | 0.128 | 0.150 | 2.53 | 1.26 / 1.34 |
| ln total-variance ratio | 0.616 | 15.8 | 412 | 0.229 | 0.0003 | 4.67 | 1.48 / 1.60 |

* **Within-expiry dependence dominates**: the effective sample is about 6–7% of the row count; whole-date clusters give ICC 0.73–0.75 but a smaller design effect (≈ 4.2) because dates hold ≈ 5 rows each.
* **Cross-expiry dependence is weaker but real**: lag-1 autocorrelation 0.13–0.23; Ljung–Box rejects independence for the session spread and the log ratio, not for the calendar spread. This is partly mechanical (46% shared window with the next expiry).
* **Independent cross-check on the mean of each per-expiry series**: Newey–West SE (0.286, 0.250, 0.0296) agrees with the block bootstrap SE (0.274, 0.242, 0.0292) and exceeds the expiry-iid SE (0.226, 0.222, 0.0242) by 15–25%, supporting b ≈ 5 over b = 1 (`mean_se_crosscheck.csv`).
* **Non-overlapping subsample** (259 expiries, no shared windows; different stratum): S1 calendar 2.02 (1.66–2.32 expiry-iid; 1.64–2.41 block), R1 1.308 (1.23–1.39), R2 1.359 (1.30–1.45), R3 1.105 (0.93–1.31). The calendar spread stays positive, the ratio stays above 1, and R3 again spans 1.
  Session-basis S1 is 4.51 in this stratum, consistent with the large session-basis spread of ≈ 3-day windows seen in Stage 2C; it is not a robustness replacement for the full sample.

---

## 4. T3 — total-variance ratio inference (`ratio_inference.csv`)

| Test (H0: ratio centred on 1) | Result | Assumption |
|---|---|---|
| exact sign test on per-expiry median ratio | 206/260 = **79.2%** of expiries have median r > 1, p = 4.3e-22 | independent expiries (violated mildly by overlap) |
| sign-flip permutation, mean per-expiry ln r, blocks of 1 / 5 / 8 expiries | p < 5.0e-5 in all three (the resolution of 20,000 flips) | symmetry under H0; dependence shorter than the block |
| Newey–West t of mean per-expiry ln r (4 / 8 lags) | t = 8.6 / 8.3 | stationary series, Bartlett kernel |

All three reject "ratio centred on 1" for the **typical** (R1/R2-type) ratio. R3 is a different estimand and its interval spans 1. Zero rows were excluded for V = 0.
These are statements about the historical sample under the stated assumptions: they do not address selection (T9), convention choices (T7/T8) or regime change (T5/T6), nor tradability.

---

## 5. T4 — influence and jackknife (`influence_*.csv`, `leave_one_year_out.csv`, `topk_removal.csv`, `trimmed_winsorized.csv`)

* **Leave-one-expiry-out** (260 re-estimates each): the median-type estimands barely move (max |Δ| ≤ 0.021 vol pts; R2/R2e ≤ 0.003), R1 ≤ 0.011, R3 ≤ 0.034 (largest effect expiry 2026-02-03, then 2025-04-09, 2024-06-06 — each removal raises R3).
* **Delete-1 jackknife SE** equals the bootstrap SE for the smooth estimands (R1 0.032 vs 0.031; R3 0.061 vs 0.061 expiry-iid). The delete-block (b = 5) jackknife agrees with the block bootstrap (R1 0.039 vs 0.039; R3 0.077 vs 0.074).
  The delete-1 jackknife is **not a consistent variance estimator for medians**; its median-type SEs/biases are tabulated but should not be interpreted (e.g. the reported jackknife "bias" of S1 session is meaningless).
* **Leave-one-expiry-year-out** (by expiry year, 14–53 expiries removed): S1 calendar 1.66–1.93 (lowest without 2022), S1 session 2.40–2.82, R2 1.351–1.381, R3 1.07–1.17. No single year reverses any sign or pushes a typical-ratio estimate to 1.
  Single-year point estimates (descriptive context only; no inference, deferred to T5/T6): S1 calendar 1.93 / 2.98 / 1.56 / 1.69 / 1.81 / 1.49 for 2021…2026; R3 1.17 / 1.23 / 1.25 / 1.06 / 1.16 / **0.91** (2026 only).
* **Removing the k most influential expiries** (ranked on the same estimand) versus k random expiries: R3 rises 1.116 → 1.244 (k = 5) and 1.264 (k = 10) against a random-removal median 1.114/1.113 (5–95%: 1.108–1.133 / 1.104–1.145): R3 is **concentrated in a handful of high-variance expiries**.
  S1 calendar moves −0.056 (k = 5), −0.102 (k = 10); R2 moves +0.012, +0.027; both are small relative to the ±0.3 interval half-widths.
* **Trimmed / winsorized** (per-expiry series; calendar spread): mean 1.47, median 1.86, 10% trimmed mean 1.78, 5% winsorized 1.73 — the per-expiry distribution is left-skewed, so mean-based and median-based summaries differ by ≈ 0.4 vol pts; geometric-mean ratio 1.29 vs trimmed 1.33.
  The influence ranking is descriptive; it is not an exclusion rule.

---

## 6. Synthetic dependence validation (`validation/coverage_simulation_output.txt`)

Known-truth study of the resampling procedures: 260 clusters x 25 rows (the observed shape), within-cluster correlation 0.58 (observed ICC of the calendar spread),
cluster effects AR(1) across consecutive clusters with coefficient phi; estimand = all-observation median (true value 0); 300 replications x 300 bootstrap draws; nominal coverage 95%.
phi = 0.25 is close to the observed lag-1 autocorrelation of the per-expiry series (0.13-0.23).

| phi | row-iid (ignores clustering) | cluster-iid | block b=2 | **block b=5 (headline)** | block b=8 | stationary (mean 5) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.443 | 0.943 | 0.930 | **0.923** | 0.917 | 0.930 |
| 0.25 | 0.397 | 0.860 | 0.897 | **0.923** | 0.923 | 0.910 |
| 0.60 | 0.277 | 0.683 | 0.770 | **0.847** | 0.893 | 0.863 |

(Monte-Carlo SE of each coverage 0.013-0.029.) Mean interval widths at phi = 0.25: row-iid 0.093, cluster-iid 0.301, block b=5 0.358.
* Row-level resampling covers the truth only 28-44% of the time: the naive intervals in section 2 are not usable.
* With no cross-cluster dependence the whole-cluster and block procedures are near nominal (0.92-0.94; blocks cost about 2 points of coverage).
* At an observed-like dependence level, cluster-iid under-covers (0.86) and b = 5 restores about 92%; under strong dependence (0.6) even b = 5 covers 85% and a longer block (b = 8) 89%.
  The headline b = 5 is therefore reasonable for the observed dependence but not conservative if dependence is stronger than the sample suggests; the b grid in section 2.3 shows how much wider longer blocks are (+8% at b = 8).
* This study validates the procedures under a stylised DGP; it does not validate any claim about the real data-generating process.

---

## 7. Assumptions and limitations

1. **Percentile intervals** from 260 clusters can be biased for skewed statistics and are discretised for medians; BCa/studentised intervals were not used.
2. **Block length** is estimated from one short noisy series; the grid shows widths still creeping up at b = 12 (0.667 vs 0.599 at b = 5). The headline choice is a documented rule, not an optimum; the conservative direction is a longer block.
3. **Overlap is only partly captured**: expiry blocks respect calendar order of expiries, but the rows of one expiry span up to 13 days and interleave in calendar time with neighbours; the date-block and expiry-block methods bracket this and agree within ≈ 20%.
4. **Stationarity**: volatility regimes (2022 stands out) violate it; no regime-dependent dependence structure was modelled.
5. **Sign-flip and sign tests** assume symmetry/independence as stated; the permutation p-value floor is 5e-5.
6. **Jackknife for medians** is inconsistent (see §5); only the smooth estimands (R1, R1e, R3) should use jackknife SEs.
7. **Scope**: pooled sample only; dev-vs-holdout, strata, specification sensitivity, selection and quote-noise questions are deliberately **not** addressed here (T5–T12). The holdout has already been reported in Stage 2C and is not pristine.
8. Intervals describe uncertainty of the historical sample statistics; they are not predictive intervals and do not establish a tradable edge or a risk premium.

---

## 8. Tests and validation

* `tests/test_stage2d_gate1.py` — 37 tests: hand-computed estimands (all nine, unbalanced clusters, zero-V rows), per-expiry estimands undefined for row/date clusters, row-order invariance, **deterministic seeds** (same seed identical, different seeds differ),
  **small-cluster failures** (29 clusters, oversized or non-positive blocks, unknown scheme, 15 row-clusters, too few sign-flip blocks, k ≥ n), whole-cluster integrity of every scheme, moving-block runs, stationary mean block length,
  ACF/χ²/Ljung–Box (vs known quantiles and scipy)/ICC/Politis–White/Newey–West on known processes, window-overlap and subsample rules (outcome-free, drops overlapping/missing expiries), exact sign test, sign-flip calibration and the permutation-p correction,
  jackknife exact closed forms, leave-one-out/leave-group-out, top-k vs random removal, trimmed/winsorized means, **synthetic dependence studies** (row-iid under-covers with clustered rows; block resampling repairs coverage under serial dependence),
  builder baseline-reproduction guard, refusal to write outside `stage2d`, end-to-end determinism and no files created outside the output directory.
  Mutation checks (7 mutants: `rows_for` offset, ACF divisor, jackknife factor, sign-flip p, S2 as row median, non-circular blocks, design effect, plus an equivalent control): five were caught immediately; two survived at first (the hand-built S2 calendar value coincided with S1; no test pinned the permutation "+1" correction), the tests were strengthened and both are now caught. Only the equivalent control survives.
* `validation/independent_gate1_check.py` (csv + math + scipy, no package imports): all nine estimands, ICC/design effects, ACF, Ljung–Box (scipy χ²), Newey–West, exact sign test (scipy), all 260 leave-one-out vectors (max difference 7.5e-13), jackknife SEs, block jackknife, window-overlap and the subsample selection — **all agree at 1e-9**.
* `validation/immutability_check.py`: baseline HEAD unchanged, 0 tracked files modified, all new paths are Stage 2D paths, Stage 2A/2B/2C and protected files identical to baseline.
* Full suite (`python -m pytest -q`): **1151 passed** (1114 existing + 37 new), 0 failed.

## 9. Files and reproduction
New code: `optionsengine/research/stage2d/{__init__,clusters,estimands,dependence,ratio_inference,influence,build_gate1}.py`; new tests: `tests/test_stage2d_gate1.py`;
outputs: `research_output/stage2d/gate1/` (`IMPLEMENTATION_PLAN.md`, this report, 19 CSV/JSON tables, `validation/`).
```
python -m optionsengine.research.stage2d.build_gate1 --aligned research_output/stage2c/annualization_audit/aligned_observations.csv --stage2c research_output/stage2c --out research_output/stage2d/gate1
python research_output/stage2d/gate1/validation/independent_gate1_check.py
python research_output/stage2d/gate1/validation/immutability_check.py
python research_output/stage2d/gate1/validation/coverage_simulation.py 300 300
python -m pytest -q
```
