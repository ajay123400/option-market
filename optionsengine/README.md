# optionsengine — Phase 1

Pricing, implied volatility, Greeks and a normalized data schema for European
index options (built for NIFTY). Standard library only; tests need `pytest`.

| Module | Responsibility |
|---|---|
| `bsm.py` | Black-Scholes-Merton price (continuous dividend yield `q`), bounds, carry-implied yield |
| `sensitivities.py` | Delta, gamma, vega, theta, rho (closed form) and their units |
| `implied_vol.py` | Safeguarded Newton/bisection IV solver, status + diagnostics |
| `quality.py` | Quote-quality flags (zero bid, crossed, stale, wide spread …) and price selection |
| `schema.py` | RAW `OptionContract`, `MarketAssumptions` (no calculated fields) |
| `sessions.py` | Expiry close-time schedule (15:30 IST before 2026-08-03, 15:40 IST from then) |
| `forward.py` | Put-call-parity forward estimator with a dispersion/outlier quality gate |
| `analytics.py` | `analyze_contract(raw, assumptions)` → separate `OptionAnalytics` |

## Conventions
* Rates, yields, vol are decimals p.a. `T` is in years, **ACT/365 calendar time**.
* Prices are per unit of underlying, not per lot.
* **Greek units:** `delta`/`gamma` per 1 underlying point; `vega` per **1 vol
  percentage point**; `theta` per **calendar day**; `rho` per **1 percentage
  point** of rate. `Greeks.raw()` gives vega per 1.00, theta per year, rho per 1.00.
* **Market IV** (`IVResult.kind == "market_implied"`) is solved from a quote; a
  **theoretical** value comes only from `bsm_price`. They are never mixed.

## IV solver
Checks no-arbitrage bounds first, brackets sigma in [1e-4, 5.0], then a
safeguarded Newton (bisection fallback). Converged iff
`|BSM(sigma) - price| <= min(price_tol, price_rel_tol*price)`. Anything else
(`BELOW_LOWER_BOUND`, `ABOVE_UPPER_BOUND`, `AT_BOUND`, `OUTSIDE_VOL_RANGE`,
`NOT_CONVERGED`, `EXPIRED`, `INVALID_INPUT`, `PRICE_NOT_POSITIVE`) returns
`iv=None`. `diagnostics.ill_conditioned` marks roots whose vol uncertainty
(`tol/vega`) exceeds `max_vol_uncertainty` — don't fit smiles to those.

## Expiry close time (`sessions.py`)
T is measured to the contract's expiry instant, and near expiry it is the dominant input (a 10-minute
error moves expiry-day ATM IV by ~x1.29 on a 25-minute horizon). No dataset in this repo carries an expiry
time, so the engine uses an explicit, overridable **schedule keyed on the expiry date**:

| expiry date | close (IST) | source |
|---|---|---|
| before 2026-08-03 | 15:30 | NSE F&O regular close |
| on/after 2026-08-03 | 15:40 | `market_calendar.py:27`, `simulator.py:32`; recorded 1-min option bars end at 15:29 before / 15:39 from that date |

Provider/contract metadata always wins: put a real expiry date-time into `OptionContract.expiry`. The
schedule is a convention for date-only expiries, not an exchange circular; to change it append a
`CloseTimeRule` to a custom `ExpiryCloseSchedule` and pass `schedule=`. Observation date does not matter -
a 2026-07-31 snapshot of a contract expiring 2026-08-04 uses the 15:40 close.

## Price resolution (tick size) and IV uncertainty
Convergence and resolution are **separate**. A converged IV can still be poorly determined because exchange
prices are quantised (tick 0.05): the true price is only known to +/- tick/2. `SolverConfig.price_resolution`
(default 0.05; `None`/0 disables) re-solves at `price -/+ res/2` and reports
`diagnostics.iv_interval`, `resolution_uncertainty` and `resolution_limited` (> `max_resolution_uncertainty`,
default 1 vol point). The IV is still returned and status stays `CONVERGED` - nothing is rejected; use
`IVResult.reliable` / `OptionAnalytics.iv_reliable` when you need converged + well-conditioned + resolved.
Near expiry vega shrinks like sqrt(T) so a half-tick maps to an ever wider IV band. Example (S=24,500, 15 min to
expiry, vol 14 %): ATM call 7.35 -> IV +/-0.05 vol point; call 200 pts ITM quotes 200.05 -> band 0 %-52 %;
245-pt OTM call quotes 0.00 -> no IV. Rounding a price onto the grid can also push an ITM put slightly below its
discounted intrinsic (`BELOW_LOWER_BOUND`). A bid/ask *mid* sits on a finer grid but its real uncertainty is
the spread - use `quality.py`, not a smaller `price_resolution`.

## Parity-forward gate (`forward.py`)
`estimate_parity_forward(observations, r, T, reference_price)` -> `ForwardEstimate(status, forward, ...)`.
Estimator: each strike with a fresh call and put gives `F_i = K + e^{rT}(C - P)`; take the 6 strikes nearest
the reference price; with >= 4 candidates drop outliers (|F_i - median| > max(3.5 * 1.4826 * MAD, 1 bp of
reference)); require >= 3 inliers and <= 50 % outliers; require inlier dispersion (max - min) <= 5 bps of the
reference (~12 index points at 24,500); forward = median of inliers. Callers must pass only fresh quotes - the
estimator cannot see staleness.

| status | meaning | `forward` |
|---|---|---|
| `OK` | consistent | median of inliers |
| `LOW_CONFIDENCE` | enough data but inconsistent (dispersion/outliers) | **None** (`candidate_forward` for diagnostics only) |
| `UNAVAILABLE` | < 3 valid pairs | **None** |

All thresholds are policy parameters (`ForwardConfig`), calibrated on this repo's NIFTY history (EOD
dispersion median 2.2 bps / p90 10.7 bps; error vs monthly futures grows smoothly with dispersion, so 5 bps is
a risk choice, not a cliff). The same forward error costs more IV near expiry (IV error ~ delta*dF/vega):
tighten `max_dispersion_bps` for short-dated analytics.

## Assumptions and limitations
* European exercise, constant vol / r / q, lognormal returns, no costs or jumps.
* There is **no default r or q**; you must pass both with a stated source.
  NIFTY options price off the *forward*: derive `q` with `implied_carry_yield(S, F, T, r)`
  rather than guessing a dividend yield. The legacy `greeks.py` (Black-76) is untouched.
* Greeks raise `DegenerateInputError` at `T == 0` or `sigma == 0` instead of returning a number.
* Quality thresholds (60 s quote age, 50 % relative spread …) are conservative placeholders to be tuned on real data.
* `expiry` must be tz-aware. Prefer the provider's real expiry date-time; if you only have a date use
  `expiry_at_close(date)` (see "Expiry close time") and set `expiry_time_assumed=True`.

## Run
    pip install pytest
    python -m pytest -q          # from the repo root
    python examples/phase1_demo.py
