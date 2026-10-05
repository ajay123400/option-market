"""Phase A Step 3: descriptive analysis of the recorder's bid/ask data (offline, read-only on the databases).

Turns recorded rows into: mid-price / bid / ask / last-trade implied volatilities (same Black-76 solver, parity-forward gate and ATM construction as the Stage 2A research code),
half-spreads in rupees and in vol points, last-trade-versus-mid bias, liquidity by days-to-expiry / moneyness / time of day, and executable-cost SCENARIOS.

Everything here is a descriptive statistic of recorded quotes. Nothing in this package is a trading rule, signal or evidence of tradability. It never writes to the recorder's data directory
unless you point --out there, and nothing in the app or the recorder imports it.
"""
