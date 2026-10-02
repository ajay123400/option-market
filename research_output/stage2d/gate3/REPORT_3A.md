# Stage 2D — Gate 3a: selection audit (T9) and quote-noise sensitivity (T10)

Status: **measurement and statistics only.** No strategy, signal, P&L or Fyers link; Stage 2A/2B/2C and Gates 1–2 are untouched (`validation/immutability_check_output.txt`: 0 tracked files changed).
Design frozen in `GATE3A_IMPLEMENTATION_PLAN.md` before any Gate 3a number existed. Baseline HEAD `b29dc2c` (Gate 2) over `f2a9a2f` (Gate 1); neither is pushed.

> **Reading guide.** A bound, interval or p-value here describes the historical sample; none of it is evidence of tradability. **There is no bid/ask in this dataset.** Every T10 scenario is an
> assumption or a bound about how far last-trade prices might sit from a fair mid price; none is observed spread data. T9 uses only information known at the snapshot time t in the inclusion model.

Population (unchanged): EXP / hybrid, 6,490 rows, 260 expiries; the builder reproduces the committed Stage 2C baseline at 1e-9 before running. Intervals: expiry moving-block bootstrap, b = 5.

---

## 1. T9 — selection-on-outcome / rejected-observation audit

### 1.1 What is missing, and why
Within ≤ 14 calendar DTE, Stage 2A attempted **7,458** snapshot × expiry groups.

| Layer | Groups | Meaning |
|---|---:|---|
| Observed (EXP/hybrid rows) | 6,490 | the analysis sample |
| Expiry-day, by construction | 780 | no expiry-aligned target exists (a population definition, not selection) |
| **L1 snapshot-quality rejections** | **73** | forward LOW_CONFIDENCE 40, no spot bar at the snapshot 29, forward UNAVAILABLE 3, no option bars 1 |
| **L2 target-availability losses** | **115** | incomplete/missing session in the window 81, beyond data end 18, snapshot session not regular 12, snapshot-session bars missing 4 |

So 188 of 6,678 non-expiry-day groups (2.8%) are missing: 1.1% by L1, 1.7% by L2. The rejection pattern is **not uniform**: L1 is concentrated in 2022 (40 of 73; 3.0% of that year's groups vs 0.2–0.5% in 2023 and 2025–2026), and L2 in 2021 (46 groups, 12.1% of the year's non-expiry-day groups: early-data session gaps), 2026 (24, mostly the end-of-data censoring) and 2023 (27). By DTE the missing share rises with DTE (1–3 d: 0.5%, 7–14 d: 3.9%).

### 1.2 Raw balance (descriptive; standardized difference, rejected minus kept, expiry-block 95% interval)
| Contrast | DTE (days) | ln RV20 | previous-session range | \|overnight gap\| | ln of later realized volatility (ln Y1) |
|---|---|---|---|---|---|
| missing (L1+L2) vs observed (53 vs rest, complete covariates) | +1.28 (0.99–1.54) | +0.50 (−0.07–1.24) | +0.76 (0.35–1.24) | +0.40 (−0.05–0.85) | **+0.80 (0.33–1.31)** |
| L1 rejected vs observed (31) | +1.25 (0.94–1.54) | +1.10 (0.62–1.65) | +0.92 (0.54–1.40) | +0.59 (0.15–0.97) | **+1.12 (0.67–1.59)** |
| L2 target-unavailable vs observed (22) | +1.31 (−0.02–2.16) | −0.11 (−1.44–2.59) | +0.53 (−0.35–2.33) | +0.15 (−0.60–5.57) | +0.41 (−0.54–2.69) |

Rejected groups sit in higher-volatility conditions *and* are followed by higher realized volatility. L2 versus observed: IV level 15.1 vs 14.6 (std. diff. +0.09, −1.36–1.02) and F5 calendar spread 1.15 vs 2.16 (−0.23, −1.09–0.43): no detectable difference, with wide intervals (22 groups).

### 1.3 Does rejection depend on the *unforeseen* part of later volatility? (the information rule)
Ridge logistic model of "missing" on covariates known at t only (year, snapshot time, weekday, DTE, ln RV20, previous-session range, |overnight gap|; no option-quote or snapshot-bar column, enforced by a test), with ln Y1 added as one extra term. Null distribution: circular shift of the date-level ln Y1 series (offset ≥ 20 sessions, 2,000 draws; keeps its autocorrelation, breaks alignment with rejections).

| Model | Groups (flagged) | Standardized coefficient of ln Y1 | 95% interval (1,000 refits) | Permutation p |
|---|---:|---:|---|---:|
| **primary**: missing (L1+L2) vs observed | 5,976 (53) | +0.304 | −0.044 – +0.664 | **0.30** |
| secondary: L1 rejected vs observed | 5,954 (31) | +0.240 | −0.274 – +0.768 | 0.34 |

After conditioning on what was known at t, the later realized volatility adds **no detectable** information about rejection: the coefficients are about one null standard deviation (0.31 and 0.26) from zero. The strong raw association in §1.2 is carried by the known-at-t variables (DTE, year, range), which are the model's largest coefficients. **Power is low** (53 and 31 flagged groups): this is a bound on what could be seen, not a proof that no dependence exists. 702 groups lack covariates (mostly the first 20 sessions of 2021, before ln RV20 exists) and are outside the model.

### 1.4 Inverse-probability weighting (covariate-only model, refit in every bootstrap draw; weights 1/p̂, cap 10; max weight 4.3, none capped)
| Estimand | Complete-case, unweighted | IPW | Difference | IPW 95% interval |
|---|---:|---:|---:|---|
| S1 calendar spread | 1.766 | 1.761 | −0.005 | 1.45 – 2.08 |
| S1 session spread | 2.560 | 2.546 | −0.014 | 2.16 – 2.91 |
| R1 geometric-mean ratio | 1.2824 | 1.2813 | −0.0011 | 1.199 – 1.356 |
| R2 median ratio | 1.3645 | 1.3618 | −0.0027 | 1.281 – 1.426 |

Weighting for the observable selection moves nothing material. (The complete-case baseline, 6,102 groups, is 1.766 vs the full-sample 1.805 only because the 576 groups before ln RV20 exists are dropped.)

### 1.5 Worst-case (Manski) bounds and break-down shares
If the 188 missing groups could take any value, the **median** calendar spread lies in **[1.733, 1.881]** (L1 only: [1.777, 1.836]; L2 only: [1.757, 1.851]) against the observed 1.805; the session-basis median in [2.547, 2.758] (observed 2.651); the median ratio in [1.353, 1.388] (observed 1.371).
**Break-down**: for the median spread (or the median ratio) to reach its null the missing groups would have to be **33.6%** of the population (session basis 38.4%), all at the extreme, against 2.8% actually missing. R1 is unbounded under worst-case values; the 188 groups would need an average ratio of about 1.5e-4 for R1 to equal 1.
The bounds cover the **188 missing groups**. They do not cover the structural restrictions below.

### 1.6 Structural layers (limitations, not selection of observations)
* **Strike universe at t**: every one of the 611,194 points used in an ATM interpolation has a quote age within [1.0, 5.0] minutes (none negative, none above the 5-minute freshness gate), i.e. no quote from after t enters the universe at t. Whether a *strike was listed* because of later trading cannot be audited with this data.
* **14–30 DTE comparison — DIAGNOSTIC / POPULATION-COMPOSITION CONTEXT ONLY** (no outcomes, spreads or ratios; no extrapolation to the primary universe): 8,907 attempted 14–30 DTE groups, forward-OK 56.4% and strict-ATM 46.4%, against 99.0% / 99.0% for ≤ 14 DTE; year, snapshot-time and ln RV20 composition are essentially the same (2021–22 share 26.6% vs 25.7%; mean ln RV20 2.50 vs 2.50). The ≤ 14 DTE universe is a coverage-driven subpopulation; its results are conditional on it.

**Reading of T9.** Within the primary universe the observable selection is small (2.8%), concentrated in volatile and early periods, adjusts to nothing material under IPW, and — given its power — shows no detectable dependence on unforeseen later volatility. The audit cannot see selection on unobservables, the strike-listing process, or anything outside ≤ 14 DTE.

---

## 2. T10 — bid/ask and quote-noise sensitivity (ASSUMPTIONS and BOUNDS)

### 2.1 Reconstruction
The two points bracketing the forward are re-solved with an independent Black-76 solver (r = 6.5%): the interpolation of the stored point IVs reproduces the stored ATM IV to 3e-16, and the re-solved IVs reproduce the stored ones to 1.9e-11 (all 6,490 rows, none unsolved). A zero-shift scenario therefore reproduces the baseline exactly (S1 calendar 1.805).

### 2.2 Systematic price bias on both bracketing quotes (an assumption; calendar spread, 95% interval; R1)
| Scenario | S1 calendar | interval | S1 session | R1 |
|---|---:|---|---:|---:|
| baseline | 1.805 | 1.53 – 2.12 | 2.651 | 1.291 |
| +₹1 / +₹2 / +₹5 / +₹10 | 1.907 / 2.006 / 2.299 / 2.796 | 1.62–2.27 / 1.75–2.39 / 2.04–2.68 / 2.49–3.20 | 2.746 / 2.839 / 3.120 / 3.564 | 1.311 / 1.331 / 1.392 / 1.497 |
| −₹1 / −₹2 / −₹5 / −₹10 | 1.716 / 1.614 / 1.310 / **0.796** | 1.46–2.05 / 1.35–1.92 / 1.02–1.59 / 0.49–1.10 | 2.567 / 2.472 / 2.186 / 1.696 | 1.271 / 1.251 / 1.193 / 1.097 |
| +1% / +2% / +5% of price | 1.921 / 2.047 / 2.393 | 1.65–2.30 / 1.76–2.42 / 2.12–2.78 | 2.772 / 2.883 / 3.241 | 1.315 / 1.340 / 1.415 |
| −1% / −2% / −5% / −10% | 1.693 / 1.577 / 1.232 / **0.659** | 1.44–2.04 / 1.32–1.91 / 0.97–1.52 / 0.36–0.91 | 2.538 / 2.419 / 2.055 / 1.444 | 1.267 / 1.243 / 1.173 / 1.060 |
| calls +ε, puts −ε (ε = ₹0.5 – 5) | 1.814 – 1.815 | 1.52 – 2.19 | 2.656 – 2.667 | 1.291 |

Rule of thumb: a uniform +₹1 adds about 0.10 vol pt (median vega ₹11.3 per vol point); +1% of price adds about 0.12. A bias that moves calls and puts in **opposite** directions has almost no effect on the interpolated ATM IV.

### 2.3 Zero-mean noise (200 seeded draws per scenario)
Independent noise on each bracketing price, σ = ₹0.5 / 1 / 2 / 5: the mean calendar spread is 1.808 / 1.811 / 1.813 / 1.811 (sd 0.004 / 0.006 / 0.008 / 0.014); R1 stays 1.289–1.291; the Mincer–Zarnowitz slope moves from 0.803 to 0.803 / 0.803 / 0.802 / 0.796. Relative noise of 0.5 / 1 / 2% behaves the same (spread sd ≤ 0.010). Random quote noise of these sizes neither creates nor removes the spread.

### 2.4 Tipping bias — how large a *systematic* bias would be needed
The median calendar spread of 1.805 vol pts equals, at median vega, a systematic last-trade price bias on the bracketing quotes of **₹20.5 (interquartile ₹14.3 – 25.3), i.e. 14.2% of the median ₹140 bracketing price**; the geometric-mean ratio would equal 1 if every implied volatility were multiplied by 0.880, i.e. a **relative reduction of 12.0% of the IV level** (for example an IV of 14.0% would become 12.3%). This is a relative percentage reduction of the IV, **not 12 volatility points**; the table column `iv_reduction_pct_for_unit_ratio` is the same relative percentage. These are bounds on how large a bias would have to be to explain the whole spread, not estimates of any actual bias.

### 2.5 What the data can say about noise (observed proxy)
A seeded sample of 298 snapshots (60 expiries across 2021–2026) was re-read from the raw option bars. At 2,495 strikes within ±1% of the forward with both call and put quotes fresh (≤ 5 min), the call-minus-put implied volatility has **median −0.001 vol pt**, robust sd 0.057, median |difference| 0.038 (about ₹0.4 at median vega), 90th percentile of |difference| 0.147 (5–95%: −0.154 to +0.142); year medians within ±0.004. This is an **upper bound on quote noise** (it also contains forward-estimation error and ITM-side time-value effects) and **no call-vs-put disagreement was detected in this observable proxy** (the median call-minus-put difference is indistinguishable from zero at the 0.001 vol pt level).
**Limitation retained: a common-mode systematic last-trade bias (calls and puts biased in the same direction, for example every last trade at the ask) cannot be detected by this proxy and cannot be detected without bid/ask data.** That is exactly the situation the tipping bias in §2.4 describes.

### 2.6 Quote age and errors-in-variables
99.8% of rows have both bracketing quotes at most 1 minute old (99.9% ≤ 2, 100% ≤ 5), so there is no quote-age gradient to measure: S1 calendar 1.806 / 1.806 / 1.805 for ≤ 1 / ≤ 2 / ≤ 5 min. Correcting the Mincer–Zarnowitz slope (0.803) for classical IV noise of 0.03 – 0.10 vol pt (the call–put proxy) gives 0.803 – 0.803; for noise equivalent to ₹5 of price (0.44 vol pt) 0.809 (a scenario, not an estimate).

**Reading of T10.** Within the range of random noise and call-versus-put disagreement that can be observed in the data (none was detected beyond small noise), the spread and ratio are stable. They would change materially only under a *systematic, same-direction* bias of several rupees (each ₹5 ≈ 0.5 vol pt) and would vanish only at about ₹20 (14% of price). Whether such a bias exists cannot be settled without bid/ask data.

---

## 3. Deviations from the plan
1. The outcome-dependence test is reported for two flagged sets (primary: all 188 missing; secondary: the 73 L1 rejections), the plan named "rejections" generally.
2. The outcome model and IPW use complete cases (5,976 and 6,102 groups): groups before ln RV20 exists are not imputed.
3. The quote-age gradient is degenerate (99.8% of rows ≤ 1 minute), so it carries no information beyond the baseline.
4. The call/put proxy covers 298 snapshots (two of the 300 sampled had no usable strikes).
5. `t9_groups.csv` (one row per attempted group, known-at-t covariates, layer, reasons, ln Y1) was added so the model can be refit independently.

## 4. Limitations
* T9 power is low (53 / 31 flagged groups); no statement about selection on unobservables (liquidity) or about the strike-listing process; no inference beyond ≤ 14 DTE. The 14–30 DTE comparison is composition context only.
* T10 scenarios are assumptions; the call/put proxy is an upper bound for noise and blind to same-direction bias; no bid/ask exists, so the true sign and size of any last-trade bias are unknown.
* The expiry-day population (780 groups) is outside T9/T10 by construction.
* Intervals use the expiry moving-block bootstrap with b = 5; the Gate 1 caveats about block length apply.
* No claim about edge, premium or tradability is made or implied.

## 5. Tests and validation
* `tests/test_stage2d_gate3a.py` — 31 tests: universe layers/reasons and counts by hand, the design matrix contains only known-at-t columns (the outcome only on request, last), spot features known at t (changing the snapshot-day bars after the open or a later session leaves the covariates unchanged; only a later session moves the outcome; only an earlier one moves ln RV20), ridge logit vs closed form, scipy optimiser, shrinkage and perfect separation, permutation test calibration (≤ 20% false rejections over 24 null datasets) and power (≥ 7/8 planted), the "+1" p floor, determinism, short-series guard, balance (planted difference detected, null covered, < 30 clusters raises), Manski/break-down by hand and against the closed form, IPW (removes selection on a known covariate, neutral otherwise, weights 1 / capped, deterministic, < 30 clusters raises), Black-76 known value/parity/vega/round trip/no-arbitrage bounds/bad initial guess, bracketing and interpolation, zero shift reproduces the baseline, +₹1 ≈ 1/vega, scenario kinds/floor/determinism/errors, solved-row dropping, tipping identities, EIV correction, call/put proxy filters, builder refusal outside `stage2d`, and an end-to-end run on the real artifacts (deterministic, nothing else touched).
  **Mutation checks: 17 mutants** (ridge intercept penalty, no standardization, permutation p, swapped Manski bounds, break-down direction, IPW p vs 1/p, IPW cap, outcome leaking into the covariate model, flipped balance sign, inverted layer logic, put-price sign, interpolation weight, opposite-scenario sign, tipping formula, EIV sign, stale-quote filter, tick floor). One survived at first (the permutation "+1"); a test was added and all 17 are now caught.
* `validation/independent_gate3a_check.py` (no package imports; csv/pandas/numpy/scipy): universe counts and reasons; **gap, previous-range, ln RV20 and ln Y1 recomputed from the raw 1-minute bars** for 50 sampled dates (≤ 4.4e-16); the primary ridge-logit coefficient refit with scipy (3.04e-1, agreement to 1.4e-8) and the IPW estimates; Manski bounds and break-down shares; Black-76 re-solved with scipy brentq for two perturbed scenarios over all 6,490 rows (S1 calendar and R1 agree to ≤ 1e-9); the tipping-bias medians — **all agree**.
* Determinism: the whole build was run twice; all 17 CSVs are byte-identical.
* Immutability: 0 tracked files modified; every new file is under Stage 2D paths.
* Full suite: **1214 passed** (1183 + 31), 0 failed.

## 6. Files and reproduction
Code: `optionsengine/research/stage2d/{selection,quote_noise,build_gate3a}.py`; tests: `tests/test_stage2d_gate3a.py`; outputs in this directory: `GATE3A_IMPLEMENTATION_PLAN.md`, this report, 9 `t9_*.csv`, 8 `t10_*.csv`, `run_metadata_3a.json`, `validation/`.
```
python -m optionsengine.research.stage2d.build_gate3a --out research_output/stage2d/gate3        # ~2.5 min
python research_output/stage2d/gate3/validation/independent_gate3a_check.py
python research_output/stage2d/gate1/validation/immutability_check.py b29dc2cff767ba804fcc26451aa863a3ce73eae0
python -m pytest -q
```
