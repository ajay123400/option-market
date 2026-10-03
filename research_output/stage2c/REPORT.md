# Stage 2C — Implied Volatility vs Subsequent Realized Volatility (descriptive measurement)

Status: **measurement only**. No strategy, signal, threshold, P&L or live-app link is built or implied.
Terminology: "IV minus future RV" and "implied-minus-realized spread" are plain differences of two measured
quantities. Nothing here is a claim about profitability, mispricing or any "premium".

All volatilities are in annualized volatility points (vol pts, i.e. percent). Sample: NIFTY index options
10:00 / 13:00 / 15:00 snapshots, 2021 → 2026-09-30. Development ≤ 2024-12-31, holdout ≥ 2025-01-01 (the Stage 2A split, unchanged).

---

## 1. Look-ahead rules (explicit)

| Item | Information used | Cut-off |
|---|---|---|
| Features at snapshot *t* (ATM IV, OTM smile IV, forward, DTE, moneyness, IV quartile, recent 20-session RV regime) | Stage 2A outputs, which use option and spot bars **at or before** the snapshot bar; recent 20-session RV uses only sessions strictly before the snapshot day | ≤ *t* |
| Regime cut-offs (IV quartiles, recent-RV terciles) | **Development period only**; applied unchanged to holdout | dev only |
| Outcome (future RV) | 1-minute spot returns whose **both** endpoints lie strictly after the snapshot observation point | > *t* |

Snapshot bars are labelled by bar start (10:00, 13:00, 15:00). The observation point is **bar start + 60 s**
(the bar must be complete). The reference price for the first future return is the **close of the snapshot bar**
(observable at the observation point); the first future return therefore runs from the observation point forward.
The remaining returns after the snapshot in the snapshot session: 329 (10:00), 149 (13:00), 29 (15:00).
A snapshot-session return is never paired with an IV measured on a window that contains it. The 13:00 IV is **not**
compared with the same-day full-session RV (that would partly include returns before 13:00). Fixed-horizon targets
F1/F5/F10/F20 use only **full sessions after the snapshot session** (the remainder of the snapshot day is excluded
from F-targets and enters only the expiry-aligned target).

Evidence: `validation/lookahead_check_stage2c.py` recomputes 12 observations with all data after *t* altered
(features unchanged, targets changed) and a boundary probe confirms 40/40 timestamp decisions.

## 2. Target construction

| Target | Window start | Window end | Valid-session rule | Overnight | Annualization |
|---|---|---|---|---|---|
| F1/F5/F10/F20 | next regular session after the snapshot session | k-th following regular session | all k sessions must be regular and complete; otherwise **unavailable** (nothing is filled) | hybrid: yes (overnight gap squared included); intraday: no; close_to_close: closes only | 100·√(252·mean daily variance) |
| EXP (expiry-aligned) | snapshot observation point | expiry session close (15:29 last bar; spot ends 15:30) | partial first session + every later regular session through expiry day must be complete; **expiry-day snapshots are not forced** (only a partial session remains: `expiry_day_partial_session_only`) | hybrid / intraday | session-equivalents = n_returns/375 + later sessions, annualized by 252 |

Unavailability reasons recorded per row: `expiry_day_partial_session_only`, `beyond_data_end`,
`incomplete_or_missing_session_in_window`, `snapshot_session_bars_missing_after_snapshot`,
`snapshot_session_not_regular`, `expiry_session_not_in_regular_chain`. Special sessions are excluded from the
regular chain (Stage 2B calendar). Real data ends 2026-09-30, so late observations are censored.

Three variants are always reported separately (none is called best): **hybrid**, **intraday-only**
(overnight excluded, so it is mechanically lower), **close-to-close**.

## 3. Filtering and sample (Q1, Q8, Q7)

| Stage | All | Dev | Holdout |
|---|---:|---:|---:|
| S0 Stage 2A snapshot × expiry groups | 21,723 | 14,718 | 7,005 |
| S1 option + spot bars present | 21,079 | 14,104 | 6,975 |
| S2 forward status OK | 13,901 | 8,760 | 5,141 |
| S3 ≤ 14 calendar DTE | **7,384** | 4,771 | 2,613 |
| S4 reliable, interpolated strict ATM IV (**Primary**) | **7,384** | 4,771 | 2,613 |
| S4b Loose-only (loose ATM where strict unavailable) | 0 | 0 | 0 |

**Q1 — survivors:** 7,384 primary IV observations (1,239 unique dates, 262 expiries, median 6 / max 9 snapshots per
date; dev 4,771 / holdout 2,613). The Loose population adds **nothing** (0 loose-only rows ≤ 14 DTE), so Primary and
Loose are identical for ATM IV; they are still kept as separate columns/counts.

**Q8 — effect of ≤ 14 DTE restriction:** it removes 6,517 forward-OK groups (47% of S2) (5,230 of which had a strict
ATM IV). These are only a diagnostic (`data_coverage.csv`, row "DIAG … > 14 DTE") and are **not** in any primary
statistic, because Stage 2A strike coverage beyond 14 DTE depends on the traded-strike path. Primary rows by year:
2021 417, 2022 1,450, 2023 1,431, 2024 1,473, 2025 1,505, 2026 1,108.

**Q2 — rows with valid future RV (primary):**

| Target | hybrid / intraday | close-to-close | Share unavailable |
|---|---:|---:|---:|
| F1 | 7,347 | 7,369 | 0.5% |
| F5 | 7,240 | 7,345 | 2.0% |
| F10 | 7,093 | 7,315 | 3.9% |
| F20 | 6,796 | 7,255 | 8.0% |
| EXP | 6,490 | n/a | 12.1% (894 rows: 779 expiry-day snapshots with only a partial session left, 81 incomplete/missing session, 18 beyond data end, 12 snapshot session not regular, 4 snapshot-session bars missing) |

Hybrid availability is lower than close-to-close because hybrid needs every session in the window complete.
Full reasons: `target_availability.csv`.

**Q7 — expiry-day resolution uncertainty.** ATM IV is never lost to price resolution (strict ATM IV exists for 260/260/259 expiry-day
snapshots at 10:00/13:00/15:00; ATM tick half-width median 0.0022 vol pts non-expiry, 0.011/0.017/0.052 at expiry day 10/13/15, max 0.107).
The exploratory flag `resolution_sensitive` (half-width ≥ 0.5 vol pt) is therefore **never true**; the primary sample
loses **0 rows** to resolution. What *is* resolution-limited are OTM wing points on expiry day only: 1.9% / 21.9% / 76.2% of
converged OTM points at 10:00 / 13:00 / 15:00 (median point IV 88% / 109% / 171%, half-width 1.1 / 1.4 / 3.6 vol pts).
Those points are excluded from the primary smile, shown separately in `summary_resolution_limited.csv` and
`resolution_data_loss.csv`, and no replacement IV is invented. 15:00 expiry-day IV is not treated as precise.
Expiry-day snapshots cannot support an EXP target (only a partial session remains: 779 rows). They do have F-targets, which
cover the sessions after the contract has expired; they are reported separately in the `expiry_day` breakdown (and as the ≤ 1 d DTE bucket), not blended silently.

## 4. Distribution of IV − future RV (Q3)

Hybrid, all primary rows, vol pts. CI = date-block bootstrap (see §9) on the median.

| Target | n | median IV | median RV | median diff | 95% CI median diff | p05 / p95 | frac IV>RV | Spearman |
|---|---:|---:|---:|---:|---|---|---:|---:|
| EXP | 6,490 | 13.54 | 10.71 | +2.65 | 2.35 – 2.97 | −4.19 / +9.22 | 0.81 | 0.66 |
| F1 | 7,347 | 13.94 | 10.39 | +3.43 | 3.19 – 3.68 | −5.37 / +12.95 | 0.82 | 0.56 |
| F5 | 7,240 | 13.92 | 11.22 | +2.45 | 2.16 – 2.74 | −4.53 / +11.90 | 0.79 | 0.61 |
| F10 | 7,093 | 13.87 | 11.43 | +2.21 | 1.90 – 2.66 | −4.82 / +12.21 | 0.77 | 0.60 |
| F20 | 6,796 | 13.85 | 11.81 | +2.00 | 1.48 – 2.53 | −7.94 / +12.25 | 0.75 | 0.54 |

Other variants (medians of IV − RV): intraday-only +4.4 (EXP) to +5.0 (F1), frac IV>RV 0.92–0.95, which largely
reflects that overnight variance is dropped; close-to-close F1 +6.1 (Spearman only 0.23, a single close-to-close
return is a very noisy one-day measure), close-to-close F5/F10/F20 ≈ +2.2 to +2.6.
IV/RV ratio (invalid when RV = 0), absolute and squared error are per observation in `iv_rv_observations.csv` and summarized in
`summary_by_horizon.csv`. These are descriptive; the table does not say which horizon is "best".

### 4.1 Annualization time-basis confound — read before interpreting any table
IV is annualized on calendar time (T in calendar days / 365), whereas RV here is annualized on trading-session time
(252 sessions). A window that contains a weekend therefore has more calendar time per session, which shifts the
session-basis comparison without any change in the market's behaviour. Stage 2C reports both bases:

* EXP total-variance comparison (IV²·T vs realized Σr²): median ratio 1.37; frac IV²T > realized total variance 0.75
  (`summary_expiry_total_variance.csv`).
* EXP calendar-basis IV − RV median +1.80 vs +2.65 on session basis.
* Weekend-containing vs not (EXP, session basis): +1.89 vs +5.20; on calendar basis 1.7 vs 2.2 (dev) and 1.56 vs 2.46 (holdout).
  Rank correlation of IV−RV with `time_basis_ratio`: −0.52 on session basis vs −0.13 on calendar basis.

(The follow-up measurement of this effect is in §14, *Annualization Alignment Audit*; the numbers in §3–§13 are unchanged.)

Consequently a large part of the DTE, weekday and (partly) time-of-day pattern in the session-basis spread is a time-basis
artefact. No convention is declared correct; both are provided.

## 5. Variation with DTE (Q4) — F5 hybrid unless noted

| DTE bucket | n (F5) | median diff | frac IV>RV | Spearman | dev | holdout |
|---|---:|---:|---:|---:|---:|---:|
| ≤ 1 d (expiry day) | 764 | +9.10 | 0.88 | 0.31 | +10.21 | +7.03 |
| 1–3 d | 1,284 | +4.21 | 0.89 | 0.66 | +4.34 | +3.90 |
| 3–7 d | 1,598 | +1.78 | 0.72 | 0.69 | +2.27 | +1.03 |
| 7–14 d | 3,594 | +1.88 | 0.76 | 0.72 | +2.04 | +1.62 |

EXP: 1–3 d +5.50, 3–7 d +2.44, 7–14 d +2.03 (≤1 d unavailable by construction). The shape (larger at short DTE) is present in
both dev and holdout but is partly the time-basis effect of §4.1 and, at expiry day, partly resolution-limited IV
dynamics; ≤1 d rows are 764 observations from only ~260 expiries.

## 6. Snapshot time (Q6) — hybrid, all

| Snapshot | F5 n | median IV | median RV | F5 diff | EXP diff | dev F5 | holdout F5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10:00 | 2,417 | 14.33 | 11.23 | +2.79 | +3.05 | +3.12 | +2.17 |
| 13:00 | 2,418 | 13.94 | 11.23 | +2.52 | +2.68 | +2.82 | +1.91 |
| 15:00 | 2,405 | 13.39 | 11.21 | +2.05 | +2.26 | +2.38 | +1.56 |

The ordering 10:00 > 13:00 > 15:00 appears in both periods; the entire movement comes from median IV falling through the day while
future RV stays almost constant. On calendar basis the dev EXP differences narrow to about 1.96 / 1.95 / 1.77. The 13:00 and 15:00 EXP targets include fewer
snapshot-session returns (149 and 29), whose weight is a convention question; an exploratory check
(`validation/exploratory_sensitivities.py`) shows a uniform-vs-development-profile intraday weighting changes EXP RV by median
+0.06 vol pts (10:00 +0.18, 13:00 +0.06, 15:00 +0.01).

## 7. Dev vs holdout (Q5)

Same sign for every horizon, measure and bucket. Holdout magnitudes are smaller by roughly 0.5–1 vol pt:

| Target (hybrid) | dev median diff | holdout median diff | dev IV median | holdout IV median |
|---|---:|---:|---:|---:|
| EXP | +2.94 | +2.14 | 14.00 | 12.37 |
| F1 | +3.73 | +2.94 | 14.49 | 12.71 |
| F5 | +2.78 | +1.90 | 14.41 | 12.71 |
| F10 | +2.54 | +1.80 | 14.33 | 12.77 |
| F20 | +2.10 | +1.86 | 14.24 | 12.87 |

For F1 hybrid, F1 intraday and F5 hybrid the median-CI intervals of dev and holdout do not overlap; for the others they do.
Holdout median IV is also ~1.6 vol pts lower, so the levels of the two samples differ; the holdout was not used to choose anything.

## 8. Regime breakdowns (descriptive; cut-offs from dev only)

F5 hybrid:

* **IV quartile** (cut-offs 11.68 / 14.50 / 18.49): median diff Q1 → Q4 = +1.06 → +7.40 (holdout +0.77 → +7.45). Higher IV is associated with a larger
  difference; RV is only partially higher, mechanically so for a spread that subtracts a smaller-variance quantity.
* **Future-RV quartile**: Q1–Q3 ≈ +2.5 to +2.8, Q4 ≈ −0.1. This conditions on the outcome and is mechanical; it is shown for completeness, not as an insight.
* **Recent 20-session RV known at t** (terciles 10.77 / 13.45): low / mid / high = +1.79 / +2.65 / +3.37; 579 observations are "unknown" (insufficient history).
  No external dataset (e.g. VIX) is used.
* No threshold was tuned on holdout; all cut-offs are fixed from the development sample and are exploratory labels.

## 9. Overlap and uncertainty method

Snapshots overlap heavily (up to 9 per date; forward windows of consecutive snapshots overlap), so the raw n overstates independent
information. Reported next to n: unique dates, unique expiries, snapshots per date. Confidence intervals use a
**moving circular block bootstrap over dates** (all snapshots from a date move together), block length 5 for F1/F5, 10 for F10/EXP and 20 for
F20 dates, 1,000 resamples for the horizon table and 300 for secondary tables. These intervals are approximate: they do not remove
dependence at longer lags, they are not adjusted for multiple comparisons, and the effective sample is closer to the number
of unique expiries (262) than to 7,384.

## 10. Sampling and selection bias audit (Q9)

1. Observed IV requires traded strikes in Stage 2A with a reliable forward; S1→S2 removes 7,178 groups on forward status. Surviving
   snapshots are those with liquid enough chains and are not representative of all NIFTY option days.
2. ≤ 14 DTE is imposed because of Stage 2A strike-universe bias beyond 14 DTE. Results do not describe longer-dated options.
3. Primary rows per year differ (2021 has 417 vs ~1,430–1,505 for 2022–2025), so the DTE and expiry composition changes over time.
4. Data end 2026-09-30 right-censors late observations in F10/F20/EXP (and expiry-day and near-expiry rows are not missing at random).
5. Risk-free rate fixed at 6.5% and carry-implied yield from parity: IV level depends modestly on these assumptions.
6. Post-2026-08-03 expiry close is 15:40 while spot stops at 15:30: T includes 10 minutes without a spot print.
7. Annualization mismatch (§4.1), overnight inclusion (hybrid vs intraday), and the single index series are all convention choices.
8. Dev and holdout cover different volatility levels (median IV ≈ 14.0 vs 12.4), so cross-period comparisons are confounded with level.
9. The sample is one index and one 5-year path; it cannot be taken as representative of other periods or underlyings.

## 11. What is / is not robust across dev and holdout (Q10)

Consistent in both dev and holdout:
* median IV is above median subsequent RV for every horizon and variant (sign of median difference);
* the ordering of the difference by DTE bucket (≤1 d largest, 1–3 d next, 3–14 d smallest) and by snapshot time (10 > 13 > 15);
* IV and subsequent RV are positively rank-correlated (≈ 0.5–0.7 for hybrid; ≤ 1 d weaker at 0.13–0.37).

Not consistent or not robust:
* the **magnitude** of the difference (holdout about 0.5–1 vol pt smaller; CIs for several horizons do not overlap);
* the fraction of observations with IV > RV is lower in holdout (0.72–0.79) than in dev (0.76–0.84) for the hybrid horizons;
* tails: the 5th percentile of the difference is much lower in holdout for F10/F20 (−11.9 / −12.3 vs −3.8 / −3.1);
* the ≤ 1 d and weekend-containing patterns largely depend on the time-basis convention;
* the effective number of independent observations is small (262 expiries), so the intervals are wider than the raw n suggests.

No statement about an "edge", "premium" or tradability follows from these tables, and none is made.

## 12. Validation

Run in `validation/` (outputs saved):
* `independent_stage2c_check.py`: independent pure-Python (F-targets) and pandas (EXP) recomputation. Max RV disagreement 9e-13 vol pts, total variance 1.1e-16,
  IV−RV arithmetic 1.4e-14; availability agreement in both directions (880 sampled available rows, 400 sampled unavailable rows: 0 disagreements).
* `lookahead_check_stage2c.py`: look-ahead probe on 12 observations (features unchanged, targets change); timestamp boundary 40/40;
  wrong-boundary variant would change variance by median 0.016% (max 0.94%), showing the boundary matters little numerically but is implemented exactly.
* Synthetic tests (`tests/test_iv_rv.py`, `test_iv_rv_stats.py`, `test_build_iv_rv.py`): 10:00/13:00/15:00 snapshots, missing future bars, incomplete sessions, expiry boundary, zero future RV (ratio invalid), overlapping snapshots, no-look-ahead.
* Full suite (`python -m pytest -q`): **1114 passed**, 0 failed (45 s), including the 41 Stage 2C tests and the 30 annualization-audit tests (§14).

## 13. Files

`iv_rv_observations.csv` (gitignored, 103,376 long rows), `summary_by_horizon.csv`, `summary_by_dte.csv`,
`summary_by_snapshot_time.csv`, `summary_by_iv_regime.csv`, `summary_expiry_total_variance.csv`, `dev_vs_holdout.csv`,
`data_coverage.csv`, `target_availability.csv`, `resolution_data_loss.csv`, `summary_resolution_limited.csv`,
`sample_observations.csv`, `regime_cutoffs.json`, `run_metadata.json`, `validation/`.

Reproduce:
```
python -m optionsengine.research.build_iv_rv --stage2a research_output/stage2a/full \
  --spot data/hist1m/NIFTY50_1m.parquet --out research_output/stage2c
python research_output/stage2c/validation/independent_stage2c_check.py
python research_output/stage2c/validation/lookahead_check_stage2c.py
python -m pytest -q
```

---

## 14. Annualization Alignment Audit

Purpose: measure how much of the §4 spread comes from comparing an IV annualized on calendar time with an RV annualized on a
252-session year. Measurement only. §3–§13 are **kept unchanged** and are called the **original / session-basis** comparison below.
Nothing in `iv_rv_observations.csv` or any earlier Stage 2C table was overwritten; the audit writes only to
`research_output/stage2c/annualization_audit/` and reads the original observation table.

### 14.1 Exact mathematical convention

| Quantity | Definition |
|---|---|
| T (the IV's time) | ACT/365: `(expiry instant − observation instant).total_seconds() / (365·86400)` (Phase 1 `analytics.time_to_expiry_years`, unchanged). Not ACT/365.25, not trading-day time. Observation = bar start + 60 s. Expiry instant = 15:30 IST, 15:40 IST for expiries on/after 2026-08-03. |
| Implied total variance | `W_imp = (IV/100)² · T` |
| Realized total variance V | sum of squared 1-minute log returns (+ squared overnight/weekend/holiday gap returns for `hybrid`) strictly after the observation point; nothing filled |
| Session basis (original) | `RV_sess = 100·√(252·V / S)`, S = session-equivalents (n_partial/375 + later sessions) |
| Calendar basis (aligned) | `RV_cal = 100·√(365·V / D)`, D = elapsed calendar days of the RV window (window end − window start); `D/365` is its ACT/365 year fraction |
| Relation | `RV_sess / RV_cal = √κ`, `κ = (D/365)/(S/252)` (calendar years per session year of that window). Exactly: `spread_sess − spread_cal = RV_cal − RV_sess = RV_cal·(1−√κ)` (checked on all 99,216 rows, max error 1.4e-14) |
| Windows | EXP: observation point → expiry-session close (15:30). F1/F5/F10/F20 hybrid and close-to-close: snapshot-session close (15:30) → k-th following session close (15:30). |

The two bases are not interchangeable: κ = 1 only when the window happens to contain 365/252 ≈ 1.45 calendar days per session.
Over the whole sample κ is below 1 for a 1-day weekday window (0.690), above 1 across a weekend (2.071 for Friday→Monday), 0.967 for a
5-session window without holidays (7 calendar days) and about 1.00 for 20 sessions.

### 14.2 Which comparison is aligned

* **Aligned (same interval, no annualization convention involved):** EXP / hybrid, implied total variance `IV²·T` versus realized total variance V over
  [observation, expiry close]. For expiries before 2026-08-03 the window span equals T to < 1e-11 minutes, so
  `W_imp / V = (IV/RV_cal)²` exactly. For the 237 EXP rows with expiry on/after 2026-08-03, T runs 10 minutes past the last spot print
  (15:40 vs 15:30): reported in `t_minus_span_minutes`; excluding them changes the median spread by +0.01 (calendar basis) to +0.04 (session basis) vol pts (below).
* **Conventional annualized comparison only:** the session-basis spread (§4), and every fixed-horizon (F1/F5/F10/F20) comparison on either basis,
  because the IV belongs to an option expiring at T (a different horizon) and is compared with RV over k sessions; calendar-basis RV only fixes the clock, not the horizon.
  The implied variance "scaled to the window span" is reported in the table file as `implied_variance_basis = window_span_flat_term_structure_assumed`.
* **Not interval-complete:** EXP / intraday and F-intraday exclude overnight/weekend/holiday intervals, so no calendar-basis RV is defined; EXP/intraday total variance is a subset quantity (shown, flagged).

### 14.3 Primary diagnostic: total variance, expiry-aligned sample (EXP / hybrid, n = 6,490)

| Group | n | median W_imp | median V | median W_imp − V | median ratio | geometric-mean ratio | frac W_imp > V | vol-space equivalent (median IV − RV_cal) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 6,490 | 3.007e-4 | 2.305e-4 | +5.48e-5 (CI 4.5e-5 – 6.6e-5) | 1.371 | 1.291 (CI 1.227 – 1.360) | 0.752 | +1.80 |
| 1–3 d | 1,301 | 9.65e-5 | 7.19e-5 | +2.38e-5 | 1.423 | 1.368 | 0.778 | +2.34 |
| 3–7 d | 1,615 | 2.11e-4 | 1.61e-4 | +4.62e-5 | 1.368 | 1.291 | 0.746 | +1.72 |
| 7–14 d | 3,574 | 4.34e-4 | 3.38e-4 | +9.32e-5 | 1.351 | 1.264 | 0.746 | +1.70 |
| 10:00 | 2,164 | 3.17e-4 | 2.41e-4 | +6.00e-5 | 1.376 | 1.293 | 0.750 | +1.87 |
| 13:00 | 2,169 | 3.01e-4 | 2.30e-4 | +5.51e-5 | 1.378 | 1.296 | 0.755 | +1.82 |
| 15:00 | 2,157 | 2.83e-4 | 2.20e-4 | +4.93e-5 | 1.362 | 1.284 | 0.753 | +1.74 |
| Dev | 4,183 | 3.14e-4 | 2.41e-4 | +5.62e-5 (CI 4.5e-5 – 7.0e-5) | 1.367 | 1.297 (CI 1.231 – 1.363) | 0.760 | +1.89 |
| Holdout | 2,307 | 2.78e-4 | 2.13e-4 | +5.19e-5 (CI 3.1e-5 – 7.3e-5) | 1.378 | 1.280 (CI 1.135 – 1.433) | 0.739 | +1.69 |
| Window without weekend | 1,849 | 1.23e-4 | 9.05e-5 | +2.90e-5 | 1.412 | 1.360 | 0.775 | +2.29 |
| Window with weekend | 4,641 | 3.86e-4 | 2.99e-4 | +7.82e-5 | 1.355 | 1.265 | 0.743 | +1.67 |
| Excl. expiries ≥ 2026-08-03 (T-span mismatch) | 6,253 | 3.07e-4 | 2.33e-4 | +5.57e-5 | 1.370 | 1.287 | 0.752 | +1.81 |

Total variance is additive in time, so these rows are directly comparable across windows of different length; the
ratio is nearly flat across DTE bucket (1.35–1.42), snapshot time (1.36–1.38), period (1.37–1.38) and weekend/no-weekend windows (1.36–1.41), whereas the session-basis volatility
spread (next sections) varies strongly across the same groups. Expiry-day snapshots have no EXP target, so they are not in this table.
EXP / intraday (subset, overnight excluded): median ratio 1.96, frac W_imp > V 0.93; shown in `expiry_total_variance.csv`, not an aligned quantity.

### 14.4 Original (session basis) vs calendar-aligned, volatility space — median IV − RV, vol points

EXP / hybrid (the only horizon whose window equals the option's life). 95% intervals: date-block bootstrap, 1,000 resamples.

| Metric | Original / session basis | Calendar-aligned | Difference (aligned − original) |
|---|---:|---:|---:|
| All (n 6,490) | +2.65 (2.35–2.97) | +1.80 (1.59–2.09) | −0.85 |
| DTE 1–3 d | +5.50 | +2.34 | −3.16 |
| DTE 3–7 d | +2.44 | +1.72 | −0.72 |
| DTE 7–14 d | +2.03 | +1.70 | −0.33 |
| 10:00 | +3.05 | +1.87 | −1.19 |
| 13:00 | +2.68 | +1.82 | −0.86 |
| 15:00 | +2.26 | +1.74 | −0.52 |
| Dev | +2.94 | +1.89 | −1.05 |
| Holdout | +2.14 | +1.69 | −0.46 |
| Window without weekend | +5.20 | +2.29 | −2.91 |
| Window with weekend | +1.89 | +1.67 | −0.22 |
| Expiry day | no EXP target (partial session only) | n/a | n/a |
| Share of observations with IV > RV | 0.811 | 0.752 | −0.059 |

F5 / hybrid (conventional comparison only; 5 sessions = 7 calendar days when no holiday, κ = 0.967):

| Metric | Original / session basis | Calendar-aligned | Difference |
|---|---:|---:|---:|
| All (n 7,240) | +2.45 (2.16–2.74) | +2.54 (2.26–2.81) | +0.10 |
| ≤ 1 d (expiry day, n 764) | +9.10 | +9.02 | −0.08 |
| 1–3 d | +4.21 | +4.22 | +0.01 |
| 3–7 d | +1.78 | +1.91 | +0.12 |
| 7–14 d | +1.88 | +1.94 | +0.06 |
| 10:00 / 13:00 / 15:00 | +2.79 / +2.52 / +2.05 | +2.88 / +2.60 / +2.13 | +0.09 / +0.08 / +0.08 |
| Dev / Holdout | +2.78 / +1.90 | +2.83 / +1.96 | +0.05 / +0.06 |
| Expiry day vs not | +9.10 vs +2.20 | +9.02 vs +2.29 | −0.08 / +0.08 |

F1 / hybrid: all +3.43 → +2.74 (−0.69); dev +3.73 → +3.02, holdout +2.94 → +2.15; 10:00 / 13:00 / 15:00 +3.79 / +3.50 / +3.08 → +3.13 / +2.83 / +2.18;
expiry day (n 779) +9.42 → +8.07; non-expiry +3.19 → +2.45. F10 / F20 move by +0.07 / +0.07 (all) and by between −0.05 and +0.18 in every row.
All 9 horizon/measure combinations and every breakdown are in `original_vs_aligned.csv`. This table does not rank the two methodologies.

### 14.5 Fixed-horizon targets: same sessions, different calendar time (F1/F5/F10/F20, hybrid)

| Horizon | Elapsed calendar days (share of rows) | κ | Median RV session basis | Median RV calendar basis | Median annualization effect (RV_cal − RV_sess) |
|---|---|---:|---:|---:|---:|
| F1 | 1 d (75.9%) / 2 d (3.0%) / 3 d (18.4%) / 4 d (2.5%) / 5 d (0.2%) | 0.690 / 1.381 / 2.071 / 2.762 | 10.17 / 12.50 / 10.84 / 12.28 | 12.24 / 10.63 / 7.53 / 7.39 | +2.07 / −1.86 / −3.31 / −4.89 |
| F5 | 7 d (74.3%) / 8 d (18.0%) / 9–12 d (7.6%) | 0.967 / 1.105 / 1.24–1.66 | 10.96 / 11.56 / see file | 11.15 / 11.00 / see file | +0.19 / −0.56 / −1.4 to −4.7 |
| F10 | 14 d (53.0%) / 15 d (29.8%) / 16–20 d (17.3%) | 0.967 / 1.036 / 1.10–1.38 | 10.95 / 11.61 / see file | 11.14 / 11.41 / see file | +0.19 / −0.20 / −0.6 to −3.6 |
| F20 | 28 d (27.2%) / 29 d (35.9%) / 30–35 d (37.0%) | 0.967 / 1.001 / 1.04–1.21 | 10.67 / 12.16 / see file | 10.85 / 12.15 / see file | +0.18 / −0.01 / −0.2 to −1.9 |

So "5 sessions" is 7, 8, 9 or more calendar days and "1 session" is 1, 2, 3 or 4; the same session count annualized by sessions
differs from the calendar annualization by `√κ` (down to −4.9 vol pts for a 4-day one-session window). The four horizons are never combined into one measure; full rows per elapsed-day value are in `fixed_horizon_calendar_span.csv`.
(Sparse windows with many calendar days also contain few sessions' worth of ordinary variance plus special events, so individual cells with n < 200 are noisy.)

### 14.6 How much of the earlier positive spread is mechanically an annualization effect

Annualization effect = `RV_cal − RV_sess` per observation (exact), shown against the original median spread. "Difference of medians" is
`(spread_sess_median − spread_cal_median) / spread_sess_median`; "median of pairwise effects" is `median(effect) / spread_sess_median`.
The two differ because the statistic is not additive across medians; both are given and neither is preferred.

| Target (hybrid) | Original median spread | Residual IV − RV (calendar basis) | Annualization effect: difference of medians | Share of original median | Median of pairwise effects (share) |
|---|---:|---:|---:|---:|---:|
| EXP all | +2.65 | +1.80 | 0.85 | 31.9% | +0.63 (23.6%) |
| EXP dev / holdout | +2.94 / +2.14 | +1.89 / +1.69 | 1.05 / 0.46 | 35.8% / 21.3% | +0.84 / +0.26 (28.6% / 12.1%) |
| F1 all | +3.43 | +2.74 | 0.69 | 20.1% | +1.77 (51.6%) |
| F5 all | +2.45 | +2.54 | −0.10 | −4.0% | +0.16 (6.6%) |
| F10 all | +2.21 | +2.28 | −0.07 | −3.0% | +0.13 (5.7%) |
| F20 all | +2.00 | +2.07 | −0.07 | −3.6% | −0.01 (−0.4%) |

For the one window that equals the option's life (EXP), roughly one quarter to one third of the original median spread (about 0.6–0.85 vol pts of 2.65) is
the annualization effect; a residual median IV − RV difference of about +1.8 vol pts (CI 1.6–2.1) remains on the calendar basis, and `W_imp > V` for 75% of observations (median ratio 1.37).
The explained share is larger in the development period (≈ 36%) than in holdout (≈ 21%); about 56% of the spread in 1–3 d windows and in no-weekend windows is annualization effect, but only ≈ 12% in weekend windows.
For F5/F10/F20 the annualization effect is small in the median (κ ≈ 1) and has mixed sign; for F1 it is large and bimodal (κ 0.69 vs 2.07). The residual is described only as a
residual IV − RV difference; no explanation of its origin is attempted here.

### 14.7 Effect on the earlier patterns (descriptive)

* DTE: the 1–3 d vs 7–14 d gap in EXP medians shrinks from 3.47 to 0.64 vol pts; ordering is preserved (1–3 d largest).
* Snapshot time: the 10:00 − 15:00 gap shrinks from 0.79 to 0.13 vol pts on the calendar basis (EXP); ordering preserved. The total-variance ratio is flat across times (1.36–1.38).
* Dev vs holdout: dev − holdout gap shrinks from 0.80 to 0.20 vol pts; sign and ordering preserved. Dev and holdout CIs for EXP calendar basis overlap.
* Weekend windows: the no-weekend vs weekend gap shrinks from 3.3 to 0.6 vol pts.
* Share of observations with IV > RV: 0.81 → 0.75 (EXP).

### 14.8 Look-ahead protection and validation of the audit

* Implied side (IV, T, DTE, expiry, `W_imp`) uses only the observation instant and the contract's expiry date/time; verified by test that changing the outcome columns does not change it.
* Outcome side (V, window start/end) unchanged from Stage 2C: EXP starts at bar start + 60 s, F targets start at the snapshot-session close. Tests: changing returns at or before the snapshot bar (10:00/13:00/15:00) leaves V unchanged; changing a later return changes it.
* `tests/test_annualization_audit.py` (30 tests): ACT/365 vs 365.25 (mutation-checked), 15:30 vs 15:40 expiry instant, hand-computed `W_imp`, V→RV conversions, the `√κ` identity, κ = 1 case, Friday→Monday and holiday-gap windows, 5 sessions over 7 vs 8 days, observation-point boundary for 10:00/13:00/15:00 under both close schedules, post-2026-08-03 10-minute mismatch, intraday rows without calendar basis, zero-variance (ratio invalid), unavailable rows not filled, fixed-horizon span, no-look-ahead.
  Mutating the year length, the seconds-per-day constant or the κ formula makes 6, 12 and 2 tests fail respectively.
* `validation/independent_annualization_check.py` (pure Python from the raw 1-minute file, no `optionsengine` imports) recomputes T, V, span, S, RV_sess, RV_cal, κ, W_imp, ratios and spreads for 13 named cases
  (normal expiry ×2, Friday/weekend ×2, holiday-gap ×2, expiry-day snapshot ×3, 10:00 / 13:00 / 15:00, post-2026-08-03 expiry; two rows each), 1,200 random rows, and T/span for all 6,490 EXP rows:
  max deviations T 9.7e-17 years, V 1.0e-16, span 1.8e-15 days, RV_sess 1.4e-14, RV_cal 7.4e-12 vol pts, κ 2.2e-16; all agree at 1e-9.
* Full test suite: 1114 passed, 0 failed.

### 14.9 Remaining limitations

1. The calendar basis aligns the clock, not the economics: IV²·T assumes variance accrues uniformly in calendar time, while realized variance accrues mostly in trading hours. Whether quoted IV is calendar- or trading-time scaled cannot be tested with these data; the flat total-variance ratio across weekend and non-weekend windows is the only evidence offered, and it is descriptive.
2. ATM IV is not a model-free expected variance (skew/convexity and the choice of strike matter); `W_imp` is "ATM IV squared times T", not a variance-swap strike.
3. V is a sum of squared 1-minute index returns of a computed index; microstructure and the uniform partial-session weighting are conventions (see the §6 sensitivity, median +0.06 vol pts).
4. For the 237 EXP rows with expiry on/after 2026-08-03 the option's T includes 10 minutes without a spot print (effect on medians ≤ 0.04 vol pts).
5. Fixed-horizon comparisons remain horizon-mismatched (IV for expiry T vs RV over k sessions) on both bases.
6. Overlapping windows, the small number of independent expiries (262), data end 2026-09-30 and the Stage 2A selection limits (§10) all remain; intervals here use the same date-block bootstrap.
7. No statement about an edge, premium or tradability is made or implied.

### 14.10 Was any existing Stage 2C result materially affected?

No number in §3–§13 changed and no earlier output file was modified or regenerated (`iv_rv_observations.csv`, `summary_*.csv` keep their original timestamps). What changes is **interpretation**: the original session-basis spread for the expiry-aligned target
(+2.65 vol pts) overstates the aligned calendar-basis difference (+1.80) by 0.85 vol pts, with larger overstatement in short-DTE, no-weekend, 10:00 and development-period subsets; the DTE, weekend, time-of-day and
dev-vs-holdout gradients in §5–§7 are largely compressed on the aligned basis. The fixed-horizon F5/F10/F20 results are not materially affected (differences ≤ 0.2 vol pts); F1 is materially affected by the κ split.

Files: `annualization_audit/{aligned_observations.csv (gitignored), aligned_sample_observations.csv, original_vs_aligned.csv, annualization_effect.csv, expiry_total_variance.csv, fixed_horizon_calendar_span.csv, audit_metadata.json}`, `validation/independent_annualization_check{.py,_output.txt}`.
Reproduce: `python -m optionsengine.research.build_annualization_audit --stage2c research_output/stage2c` then `python research_output/stage2c/validation/independent_annualization_check.py`.
