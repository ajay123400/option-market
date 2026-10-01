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

## Assumptions and limitations
* European exercise, constant vol / r / q, lognormal returns, no costs or jumps.
* There is **no default r or q**; you must pass both with a stated source.
  NIFTY options price off the *forward*: derive `q` with `implied_carry_yield(S, F, T, r)`
  rather than guessing a dividend yield. The legacy `greeks.py` (Black-76) is untouched.
* Greeks raise `DegenerateInputError` at `T == 0` or `sigma == 0` instead of returning a number.
* Quality thresholds (60 s quote age, 50 % relative spread …) are conservative placeholders to be tuned on real data.
* `expiry` must be tz-aware; if you fill the 15:30 IST close yourself set `expiry_time_assumed=True`.

## Run
    pip install pytest
    python -m pytest -q          # from the repo root
    python examples/phase1_demo.py
