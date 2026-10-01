# Stage 2A report — historical expiry-wise IV surface

Code commit `ec691f5` (`optionsengine/surface.py`, `optionsengine/research/*`); all numbers below come from runs at that commit
(`run_metadata.json` in every output folder records it, clean tree). Source data is read-only; the full CSVs
(`research_output/stage2a/full`, 762k point rows) are not committed — regenerate with the commands in
`optionsengine/research/README.md` (about 3 minutes). Committed here: `sample/`, `validation/`.

**Scope reminder:** 2A only. No realized volatility, no IV-vs-RV, no strategy. Nothing here is a trading signal.

## 1. What was built
Per (snapshot, expiry): fresh quote gate → Phase 1 parity-forward gate → OTM-only IVs → reliability flags → ATM IV (interpolated
at x = ln(K/F) = 0) → 25Δ RR/BF (T > 1 day). 263 expiry files, 10:00 / 13:00 / 15:00 IST snapshots, ≤ 45 calendar days to
expiry, r = 6.5 % (assumption). Development = days before 2025-01-01, holdout = 2025-01-01 onward (fixed before results were seen).

## 2. Coverage (all attempted snapshot-expiries: 21,723)
| outcome | dev | holdout | all |
|---|---|---|---|
| forward **OK** (primary) | 8,760 | 5,141 | 13,901 |
| forward **LOW_CONFIDENCE** (labelled, no metrics) | 1,405 | 556 | 1,961 |
| forward UNAVAILABLE (< 3 fresh pairs) | 3,939 | 1,278 | 5,217 |
| no option bars at snapshot (contract not yet listed/traded) | 366 | 12 | 378 |
| no spot bar at snapshot (see §5) | 248 | 18 | 266 |

By year (snapshots / primary / with ATM): 2021 1,626 / 708 / 674; 2022 5,019 / 2,509 / 2,292; 2023 3,963 / 2,803 / 2,565;
2024 4,110 / 2,740 / 2,457; 2025 4,119 / 3,052 / 2,743; 2026 (to Sep) 2,886 / 2,089 / 1,883.

Forward gate by time to expiry (all snapshots): ≤ 14 days OK ≥ 98.9 % (1d: 99.9 %, 1-3d 100 %, 3-7d 99.9 %, 7-14d 98.9 %);
**14-30d OK 57.0 %** (LOW_CONFIDENCE 16.5 %, UNAVAILABLE 26.5 %); **> 30d OK 30.9 %** (UNAVAILABLE 59.4 %).

Smile metrics among primary smiles (OK forward): ATM IV available 100 % for ≤ 14 days in both splits; 14-30d 83.6 % dev / 80.1 %
holdout; > 30d 71.4 % / 77.9 %. 25Δ RR/BF (T > 1 d): 1-3d 100 %, 3-7d 99.9 % / 100 %, 7-14d 99.0 % / 97.9 %, 14-30d 58.8 % / 52.6 %,
> 30d 32.3 % / 35.4 %.

## 3. Exclusions, convergence, reliability (OTM points)
| | dev | holdout |
|---|---|---|
| OTM points | 481,705 | 280,243 |
| **used** in metrics | 381,149 (79.1 %) | 230,045 (82.1 %) |
| excluded: stale (> 5 min since last trade) | 49,984 | 29,942 |
| excluded: forward LOW_CONFIDENCE | 40,968 | 16,035 |
| excluded: resolution-limited (0.05 tick) | 9,593 | 4,204 |
| excluded: IV solver failed | 11 | 0 |
| excluded: never traded | 0 | 17 |
| ITM-side contracts not used (OTM-only rule) | 337,527 | 214,291 |
| listed strikes with no OTM quote | 2,173 | 2,513 |

For fresh OTM points under OK forwards: IV **converged 100.0 %** (dev 390,742 / 390,753; holdout 234,249 / 234,249);
**reliable** 97.5 % (dev) / 98.2 % (holdout); resolution-limited 2.5 % / 1.8 % of converged; ill-conditioned 0. Reliable share by
|ln K/F| × time to expiry is 100 % for every cell ≥ 1 day; below one day it falls with distance from the money (dev: 98 % at 0.5-1 %,
79 % at 1-2 %, 48 % at 4-7 %, 5 % beyond 12 %). Full matrices: `sample/report/reliable_matrix_*.csv`.

### Expiry-day caution (kept visibly flagged)
Share of converged OTM points that are **resolution-limited**: on non-expiry days 0.0 % at every snapshot; on expiry day
**1.9 % at 10:00, 21.9 % at 13:00, 76.2 % at 15:00** (reliable 98.1 % / 78.1 % / 23.8 %). Consistent with Phase 1 (72 % at 15:15).
Those points stay in `points.csv` with `resolution_limited = True`, are excluded from the metrics, and `atm_loose_iv` (the same
interpolation allowing them) carries `atm_loose_unreliable_inputs`. ATM IV on expiry day exists whenever the two ATM-adjacent
strikes are themselves reliable (ATM vega is large); RR/BF are never produced within one day. None of these expiry-day numbers
should be read as precise IV.

## 4. Validation (historical data, not synthetic)
1. **Independent recomputation** (`validation/independent_check.py`, scipy brentq on a separately written Black-76, T from dates):
   900 points (600 used + 300 resolution-limited): max |IV difference| **1.3e-9 vol points**, delta identical, tick-band
   (iv_low/iv_high) max difference 1.7e-9, T identical. 400 smiles: ATM / RR25 / BF25 re-derived from points.csv with numpy match
   **exactly** (313 had a reproducible 25Δ bracket; library values absent where the independent code also found none). Forward re-derived
   from the raw parquet for 40 OK smiles: median-of-6 minus library forward within **−0.34 … +0.46 pts** (difference = outlier removal).
2. **Hand-verifiable example** (committed in `sample/`, smile `2025-03-06T13:00_2025-03-13`, F = 22,479.018): ATM brackets are the 22,450 put
   (x = −0.0012917, IV 13.0044 %) and 22,500 call (x = +0.00093296, IV 12.8684 %): w = 0.58064 → ATM IV = 12.9254 % = file value. 25Δ
   call between 22,700 (Δ 0.28642, 12.2506 %) and 22,750 (Δ 0.24234, 12.1444 %) → 12.1629 %; 25Δ put between 22,150 (Δ 0.22679, 14.2929 %)
   and 22,200 (Δ 0.25959, 14.1049 %) → 14.1599 %; RR25 = −1.9970 vol pts; BF25 = +0.2359 vol pts. All reproduce the file.
3. **Look-ahead test on real data** (`validation/lookahead_check.py`): rebuilding 429 snapshots (12 random expiries) from only bars with
   ts ≤ t0 reproduces the full-file result exactly — **0 mismatches**.
4. **Comparison with the existing `iv_surface_research.py` outputs** (09:30 snapshots, 1,898 matched day-expiries): forward median |diff| 0.02 pt
   (p95 0.15); ATM IV median |diff| **0.044 vol pt** (p95 0.21, p99 0.47; ≤ 2 days to expiry median 0.076); 25Δ skew correlation **0.998**
   (median |diff| 0.058 vol pt). Differences are by construction: legacy ATM = mean(CE, PE) at the 50-pt strike nearest F, mine is interpolated at F
   from reliable OTM points only; legacy forward uses un-gated parity from any price (stale included); legacy 25Δ uses the nearest observed delta
   within 0.07, mine interpolates between adjacent strikes.
5. **Different regimes** (cut-offs = ATM-IV terciles measured on development smiles, 3-30 day expiries, then applied unchanged to holdout:
   low ≤ 11.8 %, mid ≤ 15.0 %, high above): median ATM IV low/mid/high — dev 10.4 / 13.4 / 17.7 %, holdout 10.1 / 13.2 / 17.4 %; median RR25 −1.4/−1.8/−2.6 vol pts (dev),
   −0.9/−1.5/−2.1 (holdout); RR/BF availability 91/86/83 % (dev), 88/83/77 % (holdout). The highest ATM IVs (50-72 %) occur on 2022-02-24, 2022-06-30, 2024-06-04 (results day) and 2024-12-05 expiry days / the days
   around them; extreme values are real observations in the data, not clipped.
6. **Sanity of levels:** yearly median ATM IV (3-30 d) 2021 15.3 %, 2022 17.5 %, 2023 10.4 %, 2024 13.1 %, 2025 11.4 %, 2026 12.4 %; RR25 negative in 98.0 %
   of 9,882 smiles; BF25 negative in 5.0 % (min −3.6 vol pts).

### Threshold sensitivity (forward-gate limit; the approved 5 bps default was not changed)
| max dispersion | OK | LOW_CONF | UNAVAILABLE | ATM available (of OK) | ATM IV change for smiles OK under both |
|---|---|---|---|---|---|
| 3 bps | 56.9 % | 18.4 % | 24.8 % | 93.3 % | 0.000 |
| **5 bps (default)** | 66.0 % | 9.3 % | 24.8 % | 90.7 % | — |
| 8 bps | 71.5 % | 3.8 % | 24.8 % | 88.9 % | 0.000 |
| 12 bps | 74.0 % | 1.3 % | 24.8 % | 88.0 % | 0.000 |

The limit only decides **which** smiles are admitted; it never changes the forward or any IV of an admitted smile (identical medians and p99 = 0).
The limit was calibrated in Phase 1 on the same history (and is T-independent, so it mostly bites beyond 14 days). Freshness sensitivity
(max age 2 / 5 / 15 min): OK 65.0 / 66.0 / 65.7 %, LOW_CONFIDENCE 5.3 / 9.3 / 15.5 %, ATM available of OK 87.1 / 90.7 / 94.7 %, RR/BF available
(T > 1 d, OK) 69.7 / 75.3 / 81.9 % — staler quotes buy coverage and cost forward quality.

## 5. Data-quality limitations
* **No bid/ask anywhere:** "last trade price" is the only price; quote-quality screening is therefore by trade age only.
* **Missing sessions:** 266 snapshots have no spot bar (168 before spot history begins on 2021-09-16; 98 on 7 special sessions — Saturday sessions 2024-03-02,
  2024-05-18, Muhurat evenings 2021-11-04, 2022-10-24, 2023-11-12, 2024-11-01, 2025-10-21). These are recorded, never filled. 378 snapshots have no
  option bars at t0: early in a contract's life bars start at its first trade (not a data gap).
* **Strike coverage is path-dependent (see §6, issue 1).**
* Spot bars are the NIFTY index, options trade off the forward; the forward is parity-implied (r = 6.5 % assumed; ±1 pp moves IV ~0.004 vol pt).
* Sparse snapshots (3 per day). 2021 coverage is thinner (primary 44 % vs ≥ 66 % later).

## 6. Unresolved issues that could materially distort research
1. **Sample selection from data construction (new finding).** `history_downloader.py` fixed each expiry's strike window from that expiry-week's
   realized low/high. For ≤ 14 days to expiry coverage is ~complete and shows no dependence on the future path (Spearman ≈ 0 … −0.12), but beyond 14 days
   availability falls with the FUTURE realized range (14-30d forward-OK 75.6 % in the calmest quartile of the spot range to expiry vs 41.2 % in the most
   volatile; ATM 63.6 % → 33.1 %; RR/BF 50.9 % → 16.2 %; `validation/selection_bias_output.txt`). Any IV-vs-RV analysis using > 14-day surfaces would be conditioned
   on the future. **Recommendation: Stage 2C restricted to ≤ 14 calendar days to expiry** (or explicitly modelled).
2. **Forward-gate threshold calibrated on the same history** (user-flagged): sensitivity above shows it affects admission only, but out-of-sample generalisation
   to other regimes/instruments is untested.
3. **Expiry-day IVs** are tick-limited; they must stay flagged in any downstream table (Stage 2C must carry `resolution_limited`/`reliable`).
4. **r = 6.5 % is an assumption**, `T` uses the repo's F&O-close convention (not an exchange circular).
5. 25Δ metrics interpolate linearly in delta between adjacent strikes (≤ 200 pts); in wide-gap strike grids they are missing rather than approximated.

## 7. Reproduce
```
pip install -r requirements-research.txt
python -m optionsengine.research.build_surface --root data/hist1m --out research_output/stage2a/full
python -m optionsengine.research.report --dir research_output/stage2a/full --sens research_output/stage2a/sens_3bps:3,research_output/stage2a/sens_8bps:8,research_output/stage2a/sens_12bps:12
python research_output/stage2a/validation/independent_check.py research_output/stage2a/full
python research_output/stage2a/validation/lookahead_check.py 12
python -m pytest -q
```
