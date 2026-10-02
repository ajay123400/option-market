# Stage 2D — prospective-test protocol (frozen)

Frozen on 2026-10-02 (the session date) with the Gate 3c deliverables; its SHA-256 is recorded in `run_metadata_3c.json`. Applies only to **expiries that have not yet occurred or whose
snapshot window starts after 2026-09-30**. As of this document **no prospective expiry has been analysed**; no result is computed, simulated or fabricated here, and
nothing below is a trading rule, signal, entry/exit condition, sizing rule or P&L target. It describes how a measurement will be repeated on new data.

## 1. Data and pipeline (unchanged code)
* Same Stage 2A → 2B/2C → annualization-audit pipeline at the commit recorded at freeze (the Gate 3c commit hash will be written into the analysis log), run on index and option files that contain
  data after 2026-09-30. No parameter, filter, rate (6.5%), DTE cap (≤ 14), freshness gate, or convention may be changed after this freeze. A pipeline bug found later is reported separately and the analysis is run both ways.
* Population: EXP / hybrid, ≤ 14 calendar DTE, expiry-aligned targets, identical definitions of S1 (calendar and session), R1, R2, R3 (R3 never headline).
* Eligibility of expiries is decided by calendar and data availability only (a target window must be complete); never by outcome.

## 2. Pre-declared estimands and uncertainty
* Headline: **S1 calendar** (all-observation median spread) and **R1** (geometric-mean total-variance ratio). Also reported: S1 session, S2 calendar, R2, R1e, R2e, R3.
* Uncertainty: expiry moving-block bootstrap, **b = 5 and b = 8 both reported**, 5,000 replications, percentile 95% intervals, base seed 20261001; fewer than 30 expiries ⇒ no interval is produced.
* **T11 reading rule:** the percentile expiry-block interval under-covered in the synthetic study (ratio estimands about 88.7–91.7%, worse under strong dependence). Every prospective ratio interval is therefore reported with
  that warning, and conclusions are never based on an interval edge falling just outside the null.

## 3. Minimum sample before any analysis
* At least **52 prospective expiries** (about one year of weekly expiries) with at least 20 snapshot rows each on average. The analysis is run **once**, after that date, by the frozen specification; there is no interim look.
  If fewer expiries exist, the result is reported as "insufficient data", not as a weaker version of the test.

## 4. Pre-declared descriptive reading of the result (no thresholds to tune)
For S1 calendar and R1 separately, report: the point estimate, the b = 5 and b = 8 intervals, the share of prospective expiries whose median spread is positive / whose median ratio exceeds 1, and the difference from the pooled
2021–2026 estimate with its interval (same method as T5). The wording used is fixed:
* "consistent with the historical sign" if the estimate has the historical sign and the b = 8 interval excludes the null;
* "not consistent" if the estimate has the opposite sign or the b = 8 interval contains the null;
* "inconclusive" otherwise (including insufficient data).
None of these phrases means tradable, profitable or an edge; each is a statement about the measured spread on new expiries.

## 5. What a prospective result can never show
Tradability: there is still no bid/ask, cost, margin or execution information. A "consistent with the historical sign" result leaves the same-direction last-trade bias bound (Rs about 20, claim C12) untested; only bid/ask data can address it.

## 6. Audit trail
Before the analysis: record the commit hash, the data file hashes and the expiry list. After: store every table with the seed, and publish the claim ledger update with the same status vocabulary. Any deviation from this protocol is listed explicitly.
