"""Reproducible Phase 1 demo (synthetic data -- no network, no Fyers).
Run from the repo root:  python examples/phase1_demo.py
"""
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optionsengine import (MarketAssumptions, OptionContract, analyze_contract, bsm_price, expiry_at_close,
                           implied_carry_yield, time_to_expiry_years)

IST = timezone(timedelta(hours=5, minutes=30))
now = datetime(2026, 10, 1, 11, 0, tzinfo=IST)
expiry = expiry_at_close(date(2026, 10, 8))   # 15:40 IST: NSE F&O close since 2026-08-03 (see optionsengine/sessions.py)
S, F, r = 24500.0, 24562.0, 0.065            # illustrative numbers, NOT live data
T = time_to_expiry_years(now, expiry)
q = implied_carry_yield(S, F, T, r)          # forward-consistent yield, not a guessed dividend
assume = MarketAssumptions(r, q, rate_source="illustrative", dividend_source="implied from futures 24562")

print(f"T = {T:.6f} y   q(carry-implied) = {q:.4%}")
for strike, kind, sigma in [(24500, "call", 0.14), (24800, "call", 0.13), (24200, "put", 0.16)]:
    mid = bsm_price(S, strike, T, r, q, sigma, kind)
    raw = OptionContract("NIFTY", S, now, expiry, float(strike), kind, "demo", bid=mid * 0.995, ask=mid * 1.005,
                         last=mid, quote_timestamp=now, volume=1000, open_interest=10000, lot_size=75,
                         contract_symbol=f"NIFTY-{strike}-{kind}", expiry_time_assumed=True)
    a = analyze_contract(raw, assume)
    g = a.greeks
    print(f"{strike} {kind:4s} mid={a.price_used:9.4f} IV={a.iv:.4%} status={a.status} "
          f"delta={g.delta:+.4f} gamma={g.gamma:.6f} vega/1pt={g.vega:.3f} theta/day={g.theta:.3f} rho/1pp={g.rho:+.3f}")

bad = OptionContract("NIFTY", S, now, expiry, 25500.0, "call", "demo", bid=0.0, ask=0.35, last=0.3)
a = analyze_contract(bad, assume)
print(f"25500 call zero-bid quote -> status={a.status}, iv={a.iv}, flags={[f.value for f in a.quote.flags]}")
