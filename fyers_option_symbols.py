"""fyers_option_symbols.py -- NIFTY option chain symbol lookup, built from
Fyers' own public NSE F&O symbol master (same public-CSV pattern as
StockDashboard's fyers_symbols.py, but for NSE_FO instead of NSE_CM).

Columns of interest in the master CSV (0-indexed):
  8  -> expiry as a unix timestamp -- NOT column 7, which is some other
        "YYYY-MM-DD" date (looks like a listing/update date, confirmed live:
        for a contract described as "NIFTY 08 Sep 26 ...", column 7 read
        "2026-09-04" while column 8 decoded to 2026-09-08, matching the
        description) -- converted to an IST calendar date for grouping.
  9  -> Fyers trading symbol, e.g. "NSE:NIFTY2690818850CE"
  13 -> underlying name, e.g. "NIFTY"
  15 -> strike price (float)
  16 -> option type, "CE" / "PE" (or "XX" for futures)
"""
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

IST = timezone(timedelta(hours=5, minutes=30))

SYMBOL_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
_cache = None          # {underlying: {expiry_date_str: {(strike, type): fyers_symbol}}}
_cache_lock = threading.Lock()
_cached_at = 0
_CACHE_TTL_SEC = 6 * 3600  # F&O master changes rarely intraday


def _build_map():
    resp = requests.get(SYMBOL_MASTER_URL, timeout=30)
    resp.raise_for_status()
    mapping = {}
    for line in resp.text.splitlines():
        parts = line.split(",")
        if len(parts) < 17:
            continue
        ts_s, fyers_symbol, underlying, strike_s, opt_type = parts[8], parts[9], parts[13], parts[15], parts[16]
        if opt_type not in ("CE", "PE"):
            continue
        try:
            strike = float(strike_s)
            expiry = datetime.fromtimestamp(int(ts_s), tz=IST).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            continue
        mapping.setdefault(underlying, {}).setdefault(expiry, {})[(strike, opt_type)] = fyers_symbol
    return mapping


def _get_map():
    global _cache, _cached_at
    with _cache_lock:
        if _cache is None or (time.monotonic() - _cached_at) > _CACHE_TTL_SEC:
            _cache = _build_map()
            _cached_at = time.monotonic()
        return _cache


def list_expiries(underlying="NIFTY"):
    """Sorted list of expiry date strings ('YYYY-MM-DD') currently listed."""
    m = _get_map().get(underlying, {})
    return sorted(m.keys())


def option_symbol(underlying, expiry, strike, opt_type):
    """expiry as 'YYYY-MM-DD'. Returns the Fyers symbol or None if not listed."""
    m = _get_map().get(underlying, {}).get(expiry, {})
    return m.get((float(strike), opt_type))


def chain_symbols(underlying, expiry, strikes):
    """Returns {strike: {'CE': fyers_symbol_or_None, 'PE': fyers_symbol_or_None}}
    for the given list of strikes."""
    m = _get_map().get(underlying, {}).get(expiry, {})
    out = {}
    for s in strikes:
        out[s] = {
            "CE": m.get((float(s), "CE")),
            "PE": m.get((float(s), "PE")),
        }
    return out


def available_strikes(underlying, expiry):
    """All strikes listed for this underlying+expiry, sorted."""
    m = _get_map().get(underlying, {}).get(expiry, {})
    return sorted({strike for (strike, _typ) in m.keys()})


if __name__ == "__main__":
    exps = list_expiries("NIFTY")
    print(f"{len(exps)} NIFTY expiries found. Nearest few: {exps[:5]}")
    nearest = exps[0]
    strikes = available_strikes("NIFTY", nearest)
    print(f"{len(strikes)} strikes for {nearest}. Sample: {strikes[:5]} ... {strikes[-5:]}")
    sym = option_symbol("NIFTY", nearest, strikes[len(strikes)//2], "CE")
    print(f"Example symbol: {sym}")
