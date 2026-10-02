# Stage 2D — consolidated final report (T12)

Status: **measurement and statistics only.** No trading strategy, entry/exit rule, P&L, signal, sizing, threshold optimisation or Fyers integration exists in Stage 2D. This report is generated from the committed Gate 1–3b tables by `build_gate3c.py`; it adds no statistic. Not committed, not pushed.

> **Headline reading.** The historical sample shows implied volatility above subsequently realised volatility, with intervals that are probably too narrow for the ratio estimands (T11). Whether any of this is tradable is not assessable with this dataset (no bid/ask, costs, margin or execution information), and nothing has been tested on expiries after 2026-09-30.

## 1. Evidence tiers and statuses
* **Tier 0** — descriptive association in this sample. **Tier 1** — statistical evidence: the expiry-block interval excludes the null, dev and holdout lie on the same side of it, every expiry year and every tested specification do too, and T9 detects no outcome-dependent selection (all five conditions are computed and stored in the ledger). **Tier 2** — tradable edge: not assessable here.
* Statuses: SUPPORTED_WITH_CAVEATS, SUPPORTED_METHOD (about the statistical procedures), INCONCLUSIVE, NOT_SUPPORTED, NOT_ASSESSABLE, NOT_CLAIMED. Counts: INCONCLUSIVE 1, NOT_ASSESSABLE 3, NOT_CLAIMED 1, NOT_SUPPORTED 1, SUPPORTED_METHOD 4, SUPPORTED_WITH_CAVEATS 10.

## 2. Claim ledger (summary; full wording, warnings and forbidden phrases in `claim_ledger.md` / `claim_ledger.csv`)
| id | topic | status | tier | key numbers | T11 warning |
|---|---|---|---|---|---|
| C01 | IV exceeds realised volatility (calendar basis) | SUPPORTED_WITH_CAVEATS | 1 | S1 calendar 1.805 vol pts, 95% expiry-block interval 1.54-2.16; dev 1.889, holdout 1.686; all expiry years positive: True; positive in 30 of 30 specifications (range 1.67-2.01) | spread-median synthetic coverage 93.7%-94.7%; not transferable to real data |
| C02 | Spread size depends on the annualization basis | SUPPORTED_WITH_CAVEATS | 0 | S1 session 2.651 (2.26-3.06) vs calendar 1.805; session-basis dev-minus-holdout difference -0.80 (p = 0.052, block-length dependent) | spread-median synthetic coverage 93.7%-94.7%; not transferable to real data |
| C03 | Typical total-variance ratio exceeds 1 | SUPPORTED_WITH_CAVEATS | 1 | R1 1.291 (1.215-1.369); R2 1.371 (1.301-1.437); 79.2% of expiries have median ratio above 1 (exact sign test p = 4.3e-22); intervals exclude 1 in 31 of 31 specifications | RATIO INTERVAL UNDER-COVERS: 88.7%-91.7% synthetic coverage (nominal 95%); worse under strong dependence |
| C04 | Ratio of summed variances (R3) | NOT_SUPPORTED | 0 | R3 1.116 (0.978-1.268) contains 1; interval contains 1 in 29 of 31 specifications; removing the 5 most influential expiries moves it 1.116 -> 1.244 (random-removal median 1.114) | RATIO INTERVAL UNDER-COVERS: 88.7%-91.7% synthetic coverage (nominal 95%); worse under strong dependence |
| C05 | Row-level inference is invalid | SUPPORTED_METHOD | NA | the expiry-block interval is 4.8x (S1 calendar) and 4.8x (R2) as wide as the row-iid interval; ICC by expiry 0.58, design effect 14.8; synthetic coverage of row-iid 24%-45% (B0) vs nominal 95% | - |
| C06 | Date-block (Stage 2C) intervals | SUPPORTED_METHOD | NA | expiry-block interval 1.22x as wide as the date-block interval for S1 calendar; synthetic mean coverage of date_block10 89.7% (B0) | - |
| C07 | Development vs holdout | INCONCLUSIVE | 0 | family A (4 contrasts): all Holm p = 1.00; S1 calendar holdout-minus-dev -0.20 (-0.85 to 0.49) | RATIO INTERVAL UNDER-COVERS: 88.7%-91.7% synthetic coverage (nominal 95%); worse under strong dependence |
| C08 | Heterogeneity across strata | SUPPORTED_WITH_CAVEATS | 0 | 5 of 12 pre-declared heterogeneity tests survive Holm adjustment; expiry-year heterogeneity not rejected | RATIO INTERVAL UNDER-COVERS: 88.7%-91.7% synthetic coverage (nominal 95%); worse under strong dependence |
| C09 | Robustness to conventions and filters | SUPPORTED_WITH_CAVEATS | 0 | S1 calendar range 1.67-2.01 (baseline 1.80); R1 range 1.272-1.920 incl. an intraday-only subset; R3 interval contains 1 in 29 of 31 specifications | RATIO INTERVAL UNDER-COVERS: 88.7%-91.7% synthetic coverage (nominal 95%); worse under strong dependence |
| C10 | Selection of observations | SUPPORTED_WITH_CAVEATS | 0 | 188 of 6678 non-expiry-day groups missing (2.8%); Manski bounds for the median calendar spread [1.733, 1.881]; break-down share 33.6%; outcome coefficient +0.30 (permutation p = 0.30, 53 flagged); IPW shifts S1 calendar by -0.005 | RATIO INTERVAL UNDER-COVERS: 88.7%-91.7% synthetic coverage (nominal 95%); worse under strong dependence |
| C11 | Population is the ≤ 14 DTE universe | SUPPORTED_WITH_CAVEATS | 0 | 14-30 DTE groups: forward-OK 56.4% and strict-ATM 46.4% vs 99.0% for ≤ 14 DTE (composition diagnostic only; no outcomes compared) | - |
| C12 | Systematic last-trade price bias (bound) | NOT_ASSESSABLE | NA | bias needed to explain the whole median spread: Rs 20.5 (IQR 14.3-25.3), 14.2% of the median bracketing price; the ratio would equal 1 if every IV were multiplied by 0.880, a relative reduction of 12.0% of the IV level (not 12 volatility points) | - |
| C13 | Call-vs-put disagreement proxy | SUPPORTED_WITH_CAVEATS | 0 | median call-minus-put IV -0.001 vol pt over 2495 strikes in 298 snapshots; robust sd 0.057 | - |
| C14 | Pipeline recovers a known truth (Tier A) | SUPPORTED_METHOD | NA | max relative error of V 9.8e-13; max IV error 0.0060 vol pt; max estimand difference 0.0011; generator-level ratio-of-sums check 1.0195 +/- 0.0075 vs c = 1 (a ratio of random sums is not exactly c) | - |
| C15 | Coverage of the percentile expiry-block bootstrap (Tier B) | SUPPORTED_METHOD | NA | B0 calibrated, b = 5: spread-median estimands 93.7%-94.7%; ratio estimands 88.7%-91.7%; strong dependence (B1) best method 82%-87%; row-iid 24%-45% | - |
| C16 | Mincer-Zarnowitz slope (descriptive) | SUPPORTED_WITH_CAVEATS | 0 | slope 0.80 (0.68-0.92), intercept 1.45 | spread-median synthetic coverage 93.7%-94.7%; not transferable to real data |
| C17 | Expiry-day stratum | SUPPORTED_WITH_CAVEATS | 0 | F1 calendar spread 8.07 (6.77-9.22) vs 2.45 on other days | spread-median synthetic coverage 93.7%-94.7%; not transferable to real data |
| C18 | Variance risk premium (not claimed) | NOT_CLAIMED | NA | n/a | - |
| C19 | Tradability and profitability (not assessable) | NOT_ASSESSABLE | 2 | n/a | - |
| C20 | Prospective validity | NOT_ASSESSABLE | NA | no prospective result exists | - |

## 3. The T11 finding (preserved, not tuned away)
Tier A (small full-pipeline known-truth recovery) is a successful pipeline-recovery validation (claim C14). Tier B (calibrated synthetic coverage/dependence study, 300 histories of 260 expiries per scenario) is evidence that the current percentile-bootstrap uncertainty procedure does **not** achieve nominal 95% coverage uniformly:
* spread-median estimands reached 93.7%–94.7% in the calibrated scenario with expiry blocks of length 5;
* ratio estimands reached only 88.7%–91.7% there, so **every ratio interval in Stage 2D carries an explicit under-coverage warning**;
* the strong-dependence stress scenario was materially worse (best method 82%–87%);
* row-level i.i.d. inference covered only 24%–45% and is invalid for this dependent, overlapping structure;
* these percentages are properties of a calibrated simulation and must not be extrapolated directly to the real dataset. The simulation is less clustered (ICC 0.49 vs 0.575 observed) but more serially dependent (ACF(1) 0.275 vs 0.128), has less negative skew and more rows per expiry; 18 of 27 calibration moments fall inside the simulated 5–95% range.

Mean coverage over the nine estimands (nominal 95%):

| scenario | date_block10 | expiry_block2 | expiry_block5 | expiry_block8 | expiry_iid | row_iid |
|---|---|---|---|---|---|---|
| B0_calibrated | 0.897 | 0.901 | 0.920 | 0.916 | 0.843 | 0.334 |
| B1_strong_dependence | 0.717 | 0.720 | 0.829 | 0.849 | 0.609 | 0.158 |
| B2_no_overlap_iid | 0.915 | 0.923 | 0.923 | 0.920 | 0.928 | 0.547 |
| B3_extreme_tails | 0.897 | 0.906 | 0.921 | 0.924 | 0.861 | 0.282 |
| B4_overlap_only | 0.899 | 0.914 | 0.923 | 0.921 | 0.874 | 0.331 |

## 4. What each gate contributed
* **Gate 1 (T1–T4):** expiry as the independence unit, block bootstrap for overlapping weekly windows, dependence diagnostics, ratio inference, influence analysis.
* **Gate 2 (T5–T8):** dev-vs-holdout contrasts (holdout previously viewed), pre-registered heterogeneity tests with Holm, a convention/filter specification curve, and the forking-paths ledger.
* **Gate 3a (T9–T10):** selection audit of rejected observations (information known at t only), Manski bounds and IPW; bid/ask-free quote-noise scenarios, labelled assumptions and bounds.
* **Gate 3b (T11):** pipeline recovery on synthetic worlds and the calibrated coverage study above.
* **Gate 3c (T12):** this ledger, the language lint and the frozen prospective protocol.

## 5. What is not claimed
No variance risk premium, no mispricing, no tradability, no profitability, no out-of-sample confirmation (the holdout was displayed in Stage 2C and Gate 1), no prospective validation, and no statement about maturities beyond 14 days (claims C11, C18–C20).

## 6. Limitations carried by every claim
Last-trade prices only (no bid/ask); a same-direction bias of roughly the size in C12 would explain the whole spread and cannot be tested here; percentile intervals from 260 expiries; the ≤ 14 DTE universe; common pipeline assumptions shared by every specification; the synthetic coverage study's calibration gaps; Tier B coverage SE about 1.3 points.

## 7. Prospective test
A frozen protocol (`PROSPECTIVE_PROTOCOL.md`, hash in `run_metadata_3c.json`) applies the unchanged pipeline to expiries after 2026-09-30, requires at least 52 expiries, reports b = 5 and b = 8, and fixes the wording of the outcome. No prospective result exists.

## 8. Language lint (`lint_report.csv`)
Ledger, final report and protocol: 0 flags. Committed Gate 1–3b documents (frozen, not edited): 2 flagged sentences, 16 negated/limited uses. Each flagged sentence is listed with its file and line.

## 9. Reproduction
```
python -m optionsengine.research.stage2d.build_gate3c --out research_output/stage2d/gate3
python research_output/stage2d/gate3/validation/independent_gate3c_check.py
python -m pytest -q
```
