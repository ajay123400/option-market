# micro_analysis — Phase A Step 3: descriptive analysis of recorded bid/ask data

Offline and read-only: it reads the recorder's daily SQLite databases and writes CSV tables. Nothing in the app or the recorder imports it, and it never talks to Fyers.
**Everything it produces is a descriptive statistic of recorded quotes. None of it is a trading rule, a signal, or evidence of tradability.** The executable-IV tables are arithmetic scenarios, not forecasts.

```
python -m micro_analysis.run E:\nifty_microstructure\micro_20261005.sqlite [more days ...] --out E:\nifty_microstructure\analysis_20261005 [--rate 0.065]
```

## What is computed (and how it ties to the research)
* **Which rows count.** Option rows from the websocket only, with a positive bid and ask, not crossed, quote age (capture − exchange feed time − estimated clock skew) at most 300 s, and traded today (`volume > 0`).
  Everything else is dropped and counted by reason in `exclusions.csv` (`not_websocket_row`, `no_bid_or_ask`, `crossed`, `stale_quote`, `never_traded_today`; first match wins).
* **Time to expiry.** ACT/365 from the observation instant — the capture time minus the estimated skew (our clock runs about 1–1.5 s ahead) — to the provider-supplied expiry instant. r = 6.5% (as Stage 2A).
* **Forward.** Per (cycle, expiry), from the MID prices of call/put pairs through the Stage 2A parity-forward gate (`optionsengine.forward`: nearest strikes, MAD outliers, dispersion limit). No forward → no IVs, but liquidity metrics are still produced.
* **Implied volatilities.** The research solver (`optionsengine.implied_vol`, BSM with the carry implied by the forward) on four prices of the OTM option at each strike: **mid, bid, ask and last trade**.
  A point is usable only if converged and reliable (not ill-conditioned, not tick-resolution-limited) — the Stage 2A `used` criterion. The last-trade IV additionally requires a trade no older than 300 s (Stage 2A used 5 minutes).
* **ATM IV.** Linear interpolation in ln(K/F) between the adjacent strikes bracketing the forward (never extrapolated, at most 100 points apart): identical to Stage 2A (a test checks equality with `build_smile` on the same prices).
* **Spreads.** `half_spread_rs = (ask − bid)/2`; `spread_pct`; `iv_spread_vol_pts = 100·(IV_ask − IV_bid)` where both exist; `half_spread_vol_pts_vega = half_spread_rs / vega` (Black-76 vega in ₹ per vol point at the strike's mid IV).
  The IV-based and vega-based half-spreads agree near the money (tested).
* **Last-trade bias.** `ltp − mid` in ₹ and in half-spreads (signed; positive = last trade above the mid), `ltp_at` (ask / bid / inside), and `IV(ltp) − IV(mid)` in vol points, for rows whose last trade is fresh (≤ 300 s).
* **Executable-IV scenarios (ATM).** If a seller crosses a fraction f ∈ {0, 0.5, 1} of the half-spread, the ATM IV obtained is `IV_mid − f·(IV_mid − IV_bid)`; the buyer's is `IV_mid + f·(IV_ask − IV_mid)`.
  Reported against the mid and against the last-trade ATM IV (the quantity the historical study used).
* **Buckets.** Days to expiry as the research (`<=1d`, `1-3d`, `3-7d`, `7-14d`, and `>14d` which is outside the research universe), moneyness by strike offset (ATM, 1-2, 3-6, 7-12), time of day (open < 10:30, midday < 13:00, afternoon < 14:30, close).

## Output tables
`row_metrics.csv` (every analysed option row), `atm_by_cycle_expiry.csv`, `primary_instants.csv` (the 10:01 / 13:01 / 15:01 cycles = the historical observation instants), `liquidity_summary.csv`, `last_trade_bias.csv`,
`last_trade_bias_by_day.csv` (near the money, day by day: shows whether an average is stable or driven by one day), `cost_scenarios.csv`, `exclusions.csv`, `run_metadata.json` (input SHA-256s, configuration, versions).

## Cautions
* A few days of data say little: always look at `n_days`, `last_trade_bias_by_day.csv`, and the spread of the quantiles, never only a mean.
* Mid, bid and ask IVs share one forward (from the mids); bid/ask prices alone would give a biased forward.
* Quotes are the displayed best bid/ask, not executable size or cost: top-of-book sizes at ATM can be a few lots. Brokerage, taxes, hedging and margin are not included.
* The websocket quote time is the exchange feed time in whole seconds; it orders updates only to the second.

## Validation
`tests/test_micro_analysis.py` (synthetic databases with known vols, spreads and last-trade biases; consistency with the Stage 2A builder; summaries by hand; CLI determinism), `micro_analysis/validation/mutation_check.py`,
`micro_analysis/validation/independent_check.py` (scipy/Black-76 re-solve of the stored prices and an independent recomputation of the tables).
