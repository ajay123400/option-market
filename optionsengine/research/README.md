# Stage 2A — historical expiry-wise IV surface

Read-only research layer on the recorded NIFTY 1-minute data (`data/hist1m`). Source files are never written.
Needs `pip install -r requirements-research.txt`. Pure logic lives in `optionsengine/surface.py` (stdlib only).

## Run
```
python -m optionsengine.research.build_surface --root data/hist1m --out research_output/stage2a/full
python -m optionsengine.research.report --dir research_output/stage2a/full \
    --sens research_output/stage2a/sens_3bps:3,research_output/stage2a/sens_8bps:8,research_output/stage2a/sens_12bps:12
python research_output/stage2a/validation/independent_check.py research_output/stage2a/full   # scipy re-computation
python research_output/stage2a/validation/lookahead_check.py 8                                # real-data look-ahead test
python -m pytest -q                                                                           # full suite
```
Sensitivity runs: `--forward-bps N` (forward-gate limit; default 5 is unchanged unless you pass it), `--max-age-min N`.

## Conventions (all explicit; nothing is guessed)
| item | convention |
|---|---|
| snapshots | bar-START 10:00, 13:00, 15:00 IST (`--times`); observation time = bar start + 60 s; only bars with ts <= t0 are read |
| expiries / dates | every expiry file, every trading day with ≤ 45 calendar days to expiry (`--max-dte`) |
| spot | the exact 1-minute spot bar at t0; if absent the snapshot is recorded as `no_spot_bar_at_snapshot` (never filled) |
| quotes | last traded price = that minute's bar close, rounded to 2 dp (float32 → 0.05 tick grid) |
| fresh | last bar with volume > 0 is at most 5 minutes old (`age_min` is an upper bound); zero-volume copied-forward bars are excluded |
| T | ACT/365 to the expiry close from `sessions.NSE_FNO_CLOSE_SCHEDULE` (15:30 IST before 2026-08-03, 15:40 from then) |
| r | 6.5 % — an **assumption** (repo constant), written to `run_metadata.json`; no dividend input (q is carry-implied from F) |
| forward | Phase 1 parity gate on fresh call+put pairs; `ok` → primary, `low_confidence` → labelled, never primary, no metrics; `unavailable` → no smile |
| OTM / moneyness | call if K ≥ F else put; x = ln(K / F) |
| reliable | `IVResult.reliable` = converged, well-conditioned and not resolution-limited at the 0.05 tick |
| ATM IV | linear interpolation in x at 0 between the two **adjacent** listed strikes bracketing F, both reliable, ≤ 100 pts apart; no extrapolation. `atm_loose_iv` allows unreliable inputs and is labelled (`atm_loose_unreliable_inputs`) |
| 25Δ RR/BF | forward delta N(d1)/N(−d1) at each point's own IV; IV at |Δ| = 0.25 by linear interpolation in delta between **adjacent** reliable strikes ≤ 200 pts apart on each wing; only if T > 1 day, forward OK and ATM available. RR = IV25c − IV25p; BF = (IV25c + IV25p)/2 − ATM |
| split | development: day < 2025-01-01; holdout: day ≥ 2025-01-01 (fixed before any result was inspected) |

## Outputs
* `smiles.csv` — one row per attempted (snapshot, expiry), including those with no smile (`group_reason`:
  `no_spot_bar_at_snapshot`, `no_option_bars_at_snapshot`, `forward_unavailable`). Columns: `smile_id, day, time, expiry,
  split, expiry_day, spot, T_days, n_quotes, forward_status, primary, forward_used, fwd_*, carry_yield, n_points, n_used,
  n_iv_converged, n_reliable, n_itm_side_excluded, n_*` (exclusion counts), `atm_iv, atm_k_low/high, atm_reason, atm_loose_iv,
  rr25, bf25, iv25_call, iv25_put, rr_bf_reason`.
* `points.csv` — one row per OTM-side listed strike: `price, age_min, log_moneyness, iv, iv_status, reliable,
  resolution_limited, ill_conditioned, iv_low, iv_high, delta_fwd, used, exclusion`. Every row has exactly one
  `exclusion` (empty only when `used`).
* `report/*.csv` — coverage, exclusion, convergence/reliability, expiry-day flags, regimes, sensitivity.
* `run_metadata.json` — command, git commit, configuration, input inventory.

## Reading the numbers
* A contract **not yet listed/traded** at t0 simply has no bar (early in a contract's life bars start at its first trade); this is
  not a data gap and shows up as fewer strikes / `forward_unavailable`.
* Expiry-day IVs are visibly flagged: most OTM/ITM points near the close are `resolution_limited` (price tick). Do not read
  them as precise IV measurements; `atm_loose_iv` exists only so thin expiry-day data is not silently dropped.
* The forward-gate limit (5 bps) was calibrated in Phase 1 on the same history this layer analyses. `report --sens` shows how
  results move with 3/8/12 bps; the default is not changed by any of it.
