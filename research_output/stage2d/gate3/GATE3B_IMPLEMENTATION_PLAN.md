# Stage 2D — Gate 3b (T11 Tier A + Tier B): implementation plan and frozen design

Written after Gate 3a was committed (`b2a7584`; parents `b29dc2c`, `f2a9a2f`; none pushed) and **before any Tier A/Tier B result was computed** (the process code and its calibration, which use only the observed Stage 2C summary moments, exist; no coverage, size or power number does).
Scope: T11 only. Measurement/validation only: no strategy, P&L, signal or Fyers link; Stage 2A/2B/2C and Gates 1–3a are read, never edited; all outputs are new files under Stage 2D paths. T12 is not started.

## Tier A — small full-pipeline known-truth recovery
Three small synthetic worlds (38 weeks, 36 weekly expiries, ≈950 snapshot rows each; c = 1.00 / 1.30 / 0.80) are written in the on-disk format the real stack reads (1-minute index file, one option file per expiry, a participant-OI date directory used only as a trading calendar). The **real, unmodified** Stage 2A (surface), 2B/2C (RV, targets) and annualization-audit code is run on them and compared with the generator's truth:
* deterministic vol path (cycle + regime step), minute returns with a U-shaped intraday variance profile plus rare jumps, overnight/weekend gap variance, fixed holidays; realised window variance computed from the generated returns, **not** from the pipeline;
* oracle expectation F = E[V | information at t] in closed form; implied total variance W = c·F; IV_true = √(W/T) (ACT/365 to the 15:30 expiry close);
* option prices: Black-76 (r = 6.5%) on a 50-point strike grid (±5%), smile IV(K) = IV_true(1 + skew·x + curv·x²), forward F = S·e^{rT}, 0.05 tick rounding, quote age 1 minute.
Recovery criteria (stated in advance): realised variance reproduced to ≤ 1e-9 relative; row-level IV error ≤ 0.05 vol pt (tick rounding and 50-point interpolation); spread and ratio errors consistent with that; the Gate 1 estimands computed from the pipeline rows within 0.05 of the same estimands computed from the truth rows; the session-vs-calendar identity (RV_cal − RV_sess = spread_sess − spread_cal) reproduced; R3 (ratio of sums) within ±3 bootstrap standard errors of c; the generator's closed-form expectation F validated by Monte Carlo.

## Tier B — larger synthetic study (the main statistical coverage evidence)
**Process** (`synth_process.py`, daily level, known truth): K-state Markov volatility regimes with exact stationary law and rare uniform regime shifts (incl. a crisis state), lognormal variance noise, jumps with heavy-tailed sizes, 4-segment intraday variance profile (nested 10:00/13:00/15:00 remainders), overnight/weekend variance, random holidays, **weekly expiries with ≤ 14-day windows that overlap** (cross-expiry dependence arises from shared vol states and shared future sessions), a persistent market-wide mispricing factor and per-row IV noise. Implied variance W = c·E[V | state]·exp(mispricing + noise − mean). Population values of all nine Gate 1 estimands = the estimands on a 40,000-expiry simulation of the same scenario.
**Calibration** (done before any inference result, tuned only on the observed Stage 2C moments of the 6,490-row sample): median IV, median RV, medians and tails of the spread (both bases) and of the ratio, share of ratio > 1, ln-ratio sd/skew, within-expiry ICC, lag-1 autocorrelation of the per-expiry spread, rows per expiry. The table of simulated vs observed moments is part of the output; residual differences are stated, not hidden.

**Scenarios (R = 300 independent histories of 260 expiries each):**
| Id | Purpose |
|---|---|
| B0 calibrated | headline: coverage by method, dev-vs-holdout under a true null and a true shift (holdout implied variance ×1.15), Wald size and power |
| B1 strong dependence | vol regimes and mispricing far more persistent: stress test of the dependence handling; dev-vs-holdout null |
| B2 no overlap, i.i.d. | i.i.d. vol states, no persistence, windows ≤ 5.5 days (weekly windows do not overlap): the case where expiry-iid should work |
| B3 extreme tails | frequent large jumps, fatter-tailed variance noise |
| B4 overlap only | i.i.d. states and no mispricing persistence, ≤ 14-day windows: dependence arises **only** from the overlapping windows |

**Methods under test** (real Gate 1/2 code, 95% percentile intervals, 500 bootstrap draws): row-iid (ignores clustering), date-block b = 10 (the Stage 2C method), expiry-iid, expiry moving-block b = 2, 5, 8.
**Reported for every scenario × method × estimand:** coverage with Wilson 95% interval, mean interval width, mean bias; flags `in_target_band_93_97` (point coverage in [0.93, 0.97]), `consistent_with_band` (Wilson interval intersects the band) and `undercovers` (Wilson upper bound < 0.93). Coverage standard error at R = 300 is about 1.3 points, so a method is described as meeting the target only when it is `consistent_with_band`; failures are reported as failures.
**Dev-vs-holdout (Gate 2 `split_contrast`)**: coverage of the true difference and the rejection rate (p < 0.05) under a true null (type I) and a true shift (power), for b = 2, 5, 8.
**Heterogeneity Wald test (Gate 2 bootstrap Wald)**: false-rejection rate at α = 0.05 on placebo strata (levels assigned at random per row and per date) and power for a planted ×1.25 effect in one DTE stratum; target "near 5%" = Wilson interval contains 0.05.
**Dependence evidence**: per-scenario mean ICC and lag-1 autocorrelation of per-expiry spreads, and the coverage ordering row-iid < expiry-iid < block methods as dependence increases.

## Code, outputs, tests
New code: `optionsengine/research/stage2d/{synth_process,coverage_study,synth_pipeline,build_gate3b}.py`; tests `tests/test_stage2d_gate3b.py`; outputs `research_output/stage2d/gate3/` (`t11a_*.csv`, `t11b_*.csv`, `run_metadata_3b.json`, `REPORT_3B.md`, `validation/`).
Tests: process invariants (stationary law, nested remainders, Monte-Carlo unbiasedness of F, population R3 ≈ c, determinism, calendar/expiry structure), Tier A generator expectation by Monte Carlo and a small real-pipeline recovery, coverage-study machinery (Wilson interval, aggregation hand values, deterministic, builder refusal outside `stage2d`), mutation checks, independent recomputation (pure-Python simulation of a Tier A world and of the Tier B moments), immutability of all earlier artifacts, full pytest.
