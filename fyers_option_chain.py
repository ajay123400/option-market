"""fyers_option_chain.py -- wraps Fyers' /data/options-chain-v3 endpoint,
which returns LTP + OI + OI-change + volume for every strike of a chosen
expiry in ONE call (this is the only Fyers endpoint that carries OI at all
-- confirmed live that neither the WebSocket tick feed nor /data/quotes
include it). Used for both the live Option Chain page and the strategy
engine's leadership tracking (OI-increase confirmation).
"""
import threading
import time

import requests

import fyers_auth
import greeks as greeks_mod

CHAIN_URL = "https://api-t1.fyers.in/data/options-chain-v3"
INDEX_SYMBOL = "NSE:NIFTY50-INDEX"

# Every open page (Option Chain, Strategy Builder, Journal) polls
# independently, each landing here -- without this, 3 tabs open at once
# triple the real Fyers call rate for data that's identical within the
# same couple of seconds anyway. A short TTL cache collapses near-
# simultaneous requests for the same (symbol, strikecount, expiry) into
# one real network call; still short enough that the chain stays "live".
_CACHE_TTL_SEC = 2.0
_cache = {}
_cache_lock = threading.Lock()


class FyersOptionChainError(RuntimeError):
    pass


def _get(params):
    key = (params.get("symbol"), params.get("strikecount"), params.get("timestamp"))
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and (now - hit[0]) < _CACHE_TTL_SEC:
            return hit[1]

    headers = {"Authorization": fyers_auth.get_auth_header()}
    resp = requests.get(CHAIN_URL, headers=headers, params=params, timeout=15)
    body = resp.json()
    if body.get("code") != 200 or body.get("s") != "ok":
        raise FyersOptionChainError(f"options-chain-v3 failed: {body}")

    with _cache_lock:
        _cache[key] = (now, body["data"])
    return body["data"]


QUOTES_URL = "https://api-t1.fyers.in/data/quotes"
_QUOTES_BATCH = 50  # Fyers' max symbols per /data/quotes call


def get_quotes(symbols):
    """{symbol: {'ltp', 'bid', 'ask', 'tt'}} for any list of symbols,
    straight from /data/quotes -- independent of the option-chain window.
    Used for legs whose strike has drifted outside the chain's ATM +/-N
    window (a far hedge, or any leg after a big move): the chain lookup
    alone left those with no price, so SL/Target weren't watched and Close
    All silently skipped them. Shares the 2s cache. `tt` is the exchange
    timestamp (epoch seconds) of the last trade/quote, for staleness checks."""
    out, missing = {}, []
    now = time.monotonic()
    with _cache_lock:
        for s in dict.fromkeys(symbols):
            hit = _cache.get(("quote", s))
            if hit is not None and (now - hit[0]) < _CACHE_TTL_SEC:
                out[s] = hit[1]
            else:
                missing.append(s)
    headers = {"Authorization": fyers_auth.get_auth_header()}
    for i in range(0, len(missing), _QUOTES_BATCH):
        batch = missing[i:i + _QUOTES_BATCH]
        resp = requests.get(QUOTES_URL, headers=headers, params={"symbols": ",".join(batch)}, timeout=15)
        body = resp.json()
        if body.get("s") != "ok":
            raise FyersOptionChainError(f"quotes failed: {body}")
        with _cache_lock:
            for d in body.get("d", []):
                v = d.get("v") or {}
                if d.get("s") == "error" or v.get("lp") is None:
                    continue
                q = {"ltp": v.get("lp"), "bid": v.get("bid"), "ask": v.get("ask"), "tt": v.get("tt"),
                     "ch": v.get("ch"), "chp": v.get("chp")}
                out[d["n"]] = q
                _cache[("quote", d["n"])] = (now, q)
    return out


MARGIN_URL = "https://api-t1.fyers.in/api/v3/multiorder/margin"


def get_margin(legs, lot_size):
    """Exchange margin (SPAN + exposure, with hedge benefit) Fyers would
    block for this basket. legs: [{'symbol', 'side': BUY/SELL, 'qty': lots}].
    BUY legs are sent FIRST -- verified live that Fyers only credits the
    hedge benefit to a short when its long hedge precedes it in the list
    (short 23600CE alone: ~Rs1.57L; long 23800CE then short: ~Rs43k).
    Returns {'margin_total', 'margin_avail'}."""
    ordered = sorted(legs, key=lambda l: 0 if l["side"] == "BUY" else 1)
    data = [{"symbol": l["symbol"], "qty": int(l["qty"]) * lot_size, "side": 1 if l["side"] == "BUY" else -1,
             "type": 2, "productType": "MARGIN", "limitPrice": 0.0, "stopLoss": 0.0} for l in ordered]
    resp = requests.post(MARGIN_URL, headers={"Authorization": fyers_auth.get_auth_header()},
                         json={"data": data}, timeout=15)
    body = resp.json()
    if body.get("s") != "ok":
        raise FyersOptionChainError(f"margin failed: {body}")
    d = body.get("data", {})
    return {"margin_total": d.get("margin_total"), "margin_avail": d.get("margin_avail")}


def chain_stats(chain):
    """Summary analytics for the chain window being shown: PCR by OI and by
    OI change, max pain, ATM straddle (= the market's priced-in move to
    expiry) and the call/put OI walls near the market (resistance/support).
    All computed over the strikes in `chain` only -- a +/-10 strike window
    gives a max pain / PCR for that window, not the whole series."""
    rows = chain.get("strikes") or []
    spot = chain.get("spot")
    if not rows or not spot:
        return None
    ce_oi = sum((r["ce"] or {}).get("oi") or 0 for r in rows)
    pe_oi = sum((r["pe"] or {}).get("oi") or 0 for r in rows)
    ce_chg = sum((r["ce"] or {}).get("oich") or 0 for r in rows)
    pe_chg = sum((r["pe"] or {}).get("oich") or 0 for r in rows)
    strikes = [r["strike"] for r in rows]

    def writers_payout(S):
        return sum(max(0.0, S - r["strike"]) * ((r["ce"] or {}).get("oi") or 0)
                   + max(0.0, r["strike"] - S) * ((r["pe"] or {}).get("oi") or 0) for r in rows)
    max_pain = min(strikes, key=writers_payout)
    atm = min(rows, key=lambda r: abs(r["strike"] - spot))
    c, p = (atm["ce"] or {}).get("ltp"), (atm["pe"] or {}).get("ltp")
    straddle = round(c + p, 2) if c and p else None
    import oi_walls
    w = oi_walls.walls({r["strike"]: (r["ce"] or {}).get("oi") for r in rows},
                       {r["strike"]: (r["pe"] or {}).get("oi") for r in rows}, spot, straddle=straddle)
    return {
        "pcr_oi": round(pe_oi / ce_oi, 2) if ce_oi else None,
        "pcr_chg_oi": round(pe_chg / ce_chg, 2) if ce_chg > 0 and pe_chg > 0 else None,
        "max_pain": max_pain,
        "atm_strike": atm["strike"],
        "atm_straddle": straddle,
        "expected_move_pct": round(straddle / spot * 100, 2) if straddle else None,
        # support / resistance = OI walls near the market (see oi_walls.py);
        # upgraded to delta-weighted once ATM IV is known (app._chain_stats_with_iv)
        "resistance": w["resistance"], "support": w["support"], "walls": w,
        "max_ce_oi_strike": w["resistance"], "max_pe_oi_strike": w["support"],
    }


def list_expiries(symbol=INDEX_SYMBOL):
    """Returns [{'date': '08-09-2026', 'expiry': '1788862200', 'expiry_flag': 'W'}, ...]"""
    data = _get({"symbol": symbol, "strikecount": 1, "timestamp": ""})
    return data.get("expiryData", [])


def get_chain(symbol=INDEX_SYMBOL, strikecount=10, expiry_timestamp="", include_greeks=False):
    """Returns a dict:
    {
      'spot': float,
      'expiries': [...],
      'strikes': [
        {'strike': 24000, 'pe': {...fields...}, 'ce': {...fields...}},
        ...
      ]
    }
    Each of pe/ce (when present) has: symbol, ltp, ltpch, ltpchp, oi, oich,
    oichp, prev_oi, volume, bid, ask, and -- only when include_greeks=True --
    iv, delta, gamma, theta, vega (each None if they couldn't be computed
    for that contract, e.g. missing/stale LTP or an already-expired series;
    see greeks.py for the Black-Scholes assumptions/caveats behind these).
    include_greeks defaults off since it's extra CPU work (an implied-vol
    solve per contract) that most callers -- the strategy engine's
    leadership tracking, the 2s-cache warm-up -- don't need.
    """
    data = _get({"symbol": symbol, "strikecount": strikecount, "timestamp": expiry_timestamp})
    chain = data.get("optionsChain", [])
    expiry_data = data.get("expiryData", [])
    spot = None
    by_strike = {}
    for row in chain:
        if row.get("strike_price", -1) == -1:
            spot = row.get("ltp")
            continue
        strike = row["strike_price"]
        side = "ce" if row.get("option_type") == "CE" else "pe"
        by_strike.setdefault(strike, {"strike": strike, "pe": None, "ce": None})
        by_strike[strike][side] = {
            "symbol": row.get("symbol"),
            "ltp": row.get("ltp"),
            "ltpch": row.get("ltpch"),
            "ltpchp": row.get("ltpchp"),
            "oi": row.get("oi"),
            "oich": row.get("oich"),
            "oichp": row.get("oichp"),
            "prev_oi": row.get("prev_oi"),
            "volume": row.get("volume"),
            "bid": row.get("bid"),
            "ask": row.get("ask"),
        }
    strikes = [by_strike[k] for k in sorted(by_strike.keys())]

    # Resolve which expiry this chain actually is: an explicit timestamp
    # was requested, or Fyers picked the nearest one for us (timestamp="")
    # -- expiryData[0] is that nearest expiry either way (same convention
    # the frontend's own dropdown relies on). Always computed (it's free,
    # just reading data already in this response) so callers doing their
    # own time-to-expiry math -- e.g. the Strategy Builder's T+0 payoff
    # curve -- don't have to re-derive it themselves.
    resolved_expiry_ts = float(expiry_timestamp) if expiry_timestamp else (
        float(expiry_data[0]["expiry"]) if expiry_data else None
    )

    if include_greeks and spot:
        T = (resolved_expiry_ts - time.time()) / (365 * 86400) if resolved_expiry_ts else None

        # Use the put-call-parity-implied forward, not the raw index spot,
        # as S for every Greeks calc -- see synthetic_forward()'s docstring
        # for why (NIFTY options price off the futures, which sit ~40-80pt
        # above the cash index; feeding raw spot in skews CE vs PE IV badly
        # at the very same strike).
        near_atm = sorted(strikes, key=lambda r: abs(r["strike"] - spot))[:6]
        quotes = [(r["strike"], r["ce"]["ltp"] if r["ce"] else None, r["pe"]["ltp"] if r["pe"] else None)
                  for r in near_atm]
        forward = greeks_mod.synthetic_forward(quotes) or spot

        for row in strikes:
            for side, is_call in (("ce", True), ("pe", False)):
                leg = row[side]
                if not leg:
                    continue
                g = greeks_mod.option_greeks_fwd(leg.get("ltp"), forward, row["strike"], T,
                                              greeks_mod.RISK_FREE_RATE, is_call) if T and T > 0 else None
                leg["iv"] = g["iv"] if g else None
                leg["delta"] = g["delta"] if g else None
                leg["gamma"] = g["gamma"] if g else None
                leg["theta"] = g["theta"] if g else None
                leg["vega"] = g["vega"] if g else None

    return {
        "spot": spot,
        "expiries": expiry_data,
        "strikes": strikes,
        "resolved_expiry_ts": resolved_expiry_ts,
    }


if __name__ == "__main__":
    exps = list_expiries()
    print(f"{len(exps)} expiries. Nearest: {exps[0] if exps else None}")
    chain = get_chain(strikecount=10)
    print(f"spot={chain['spot']}, {len(chain['strikes'])} strikes")
    for row in chain["strikes"][:3]:
        print(row)
