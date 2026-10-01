"""market_view.py -- the user's range (from range_strategy's engine state,
plus a LIVE recompute from the option chain every poll) and a combined
market-direction read built from the chain, for the Option Chain page.

Everything here is decision SUPPORT: signals with their reasons, and what
the user's own range rules would do next. It is not a trade instruction.
"""
import json
import os

import market_calendar as mc
import paths
import range_strategy

STEP = range_strategy.CONFIG["strike_step"]


def _load_state():
    try:
        with open(range_strategy.STATE_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _shift(a, b):
    """Direction from range a to range b, by the move of its midpoint."""
    if not a or not b:
        return None
    mid_a, mid_b = (a["upper"] + a["lower"]) / 2, (b["upper"] + b["lower"]) / 2
    d = round(mid_b - mid_a, 1)
    return {"points": d, "direction": "UP" if d > 0 else "DOWN" if d < 0 else "FLAT",
            "from": [round(a["lower"]), round(a["upper"])], "to": [round(b["lower"]), round(b["upper"])],
            "since": a.get("time")}


def range_view(chain, viewed_expiry_ts):
    """The official (candle-based) range, its shift history, and a LIVE
    range from the chain's current day volumes. None outside a cycle."""
    st = _load_state()
    if not st or not st.get("range"):
        return None
    today = mc.now_ist().date().isoformat()
    exp_match = False
    for e in chain.get("expiries") or []:
        d, m, y = e["date"].split("-")
        if f"{y}-{m}-{d}" == st["expiry"]:
            exp_match = viewed_expiry_ts is not None and float(e["expiry"]) == float(viewed_expiry_ts)
    hist = st.get("range_history") or []
    official = st["range"]
    out = {
        "cycle_start": st["cycle_start"], "expiry": st["expiry"], "for_viewed_expiry": exp_match,
        "official": official, "last_candle": st.get("last_candle"),
        "history": hist[-12:],
        "last_shift": _shift(hist[-2], hist[-1]) if len(hist) >= 2 else None,
        "cycle_shift": _shift(hist[0], hist[-1]) if len(hist) >= 2 else None,
        "strategy": {"status": st["status"], "exit_reason": st.get("exit_reason"),
                     "legs": st.get("legs") or {}},
        "live": None,
    }
    # LIVE leaders from the chain being shown (day's cumulative volume on
    # 100-pt strikes). If they differ from the candle-based leaders, the
    # range is recomputed now with today's first-candle highs.
    fh = st.get("today_first_high") or {}
    if exp_match and st.get("today") == today and fh.get("CE") and fh.get("PE"):
        rows = [r for r in chain.get("strikes") or [] if int(r["strike"]) % STEP == 0]
        if rows:
            ce_lead = max(rows, key=lambda r: (r["ce"] or {}).get("volume") or 0)["strike"]
            pe_lead = max(rows, key=lambda r: (r["pe"] or {}).get("volume") or 0)["strike"]
            ce_hi, pe_hi = fh["CE"].get(str(int(ce_lead))), fh["PE"].get(str(int(pe_lead)))
            if ce_hi is not None and pe_hi is not None:
                live = {"ce_leader": int(ce_lead), "pe_leader": int(pe_lead),
                        "upper": round(ce_lead + pe_hi, 2), "lower": round(pe_lead - ce_hi, 2),
                        "ce_first_high": ce_hi, "pe_first_high": pe_hi}
                live["changed"] = (live["ce_leader"] != official["ce_leader"] or live["pe_leader"] != official["pe_leader"])
                live["shift"] = _shift(official, live) if live["changed"] else None
                out["live"] = live
    return out


def _sig(name, score, why):
    return {"name": name, "score": score, "why": why}


def market_view(chain, stats, rv, index_chp=None):
    """Combines direction signals into Bullish / Bearish / Neutral, with the
    reason for each, plus method-specific suggestions."""
    spot = chain.get("spot")
    if not spot or not stats:
        return None
    sigs = []
    cur = (rv or {}).get("live") if (rv or {}).get("live", {}) and rv["live"].get("changed") else (rv or {}).get("official")
    shift = ((rv or {}).get("live") or {}).get("shift") or (rv or {}).get("last_shift")
    if shift and shift["direction"] != "FLAT":
        s = 2 if shift["direction"] == "UP" else -2
        sigs.append(_sig("Range shift (your method)", s,
                         f"range moved {shift['direction'].lower()} {abs(shift['points']):.0f} pts: "
                         f"{shift['from'][0]}-{shift['from'][1]} -> {shift['to'][0]}-{shift['to'][1]}"))
    pcr = stats.get("pcr_oi")
    if pcr is not None:
        s = 1 if pcr >= 1.2 else -1 if pcr <= 0.8 else 0
        sigs.append(_sig("PCR (OI)", s, f"{pcr} -- " + ("put writers dominate (support below)" if s > 0 else
                                                        "call writers dominate (resistance above)" if s < 0 else "balanced")))
    pcrc = stats.get("pcr_chg_oi")
    if pcrc is not None:
        s = 1 if pcrc >= 1.2 else -1 if pcrc <= 0.8 else 0
        sigs.append(_sig("PCR (today's OI change)", s, f"{pcrc} -- " + ("fresh put writing today" if s > 0 else
                                                                        "fresh call writing today" if s < 0 else "balanced fresh positions")))
    mp = stats.get("max_pain")
    if mp is not None:
        d = mp - spot
        s = 1 if d >= 75 else -1 if d <= -75 else 0
        sigs.append(_sig("Max pain", s, f"{mp} is {abs(d):.0f} pts {'above' if d > 0 else 'below'} spot"
                         + (" (pull toward it grows into expiry)" if s else "")))
    sup, res = stats.get("support"), stats.get("resistance")
    if sup and res:
        # walls near the market (delta-weighted OI inside the expected move):
        # the closer, the more it caps / props price
        to_res, to_sup = res - spot, spot - sup
        s = 1 if to_sup < to_res * 0.5 else -1 if to_res < to_sup * 0.5 else 0
        w = stats.get("walls") or {}
        sigs.append(_sig("OI walls", s, f"support {sup} ({to_sup:.0f} pts below) / resistance {res} ({to_res:.0f} pts above)"
                         + (f" -- walls within ±{w['window']} pts, {w.get('method')}" if w.get("window") else "")))
    if index_chp is not None:
        s = 1 if index_chp >= 0.3 else -1 if index_chp <= -0.3 else 0
        sigs.append(_sig("NIFTY today", s, f"{index_chp:+.2f}%"))

    def _verdict(sc, strong=3):
        return ("BULLISH" if sc >= strong else "MILD BULLISH" if sc >= 1 else
                "BEARISH" if sc <= -strong else "MILD BEARISH" if sc <= -1 else "NEUTRAL")

    # Two separate reads, shown separately on the page: what the OPTION DATA
    # says on its own, and what YOUR RANGE's shift says -- then combined.
    opt_sigs = [x for x in sigs if not x["name"].startswith("Range shift")]
    range_sig = next((x for x in sigs if x["name"].startswith("Range shift")), None)
    option_score = sum(x["score"] for x in opt_sigs)
    score = sum(x["score"] for x in sigs)
    verdict = _verdict(score)

    tips = []
    legs = ((rv or {}).get("strategy") or {}).get("legs") or {}
    if cur:
        width = cur["upper"] - cur["lower"]
        straddle = stats.get("atm_straddle")
        if straddle and width < straddle * 1.5:
            tips.append(f"Range is narrow ({width:.0f} pts) vs the market's expected move to expiry "
                        f"(ATM straddle ±{straddle:.0f}) -- higher chance of a breach; consider smaller size.")
        if spot > cur["upper"]:
            tips.append(f"NIFTY {spot:.0f} is ABOVE the range upper {cur['upper']:.0f} -- call side under pressure.")
        elif spot < cur["lower"]:
            tips.append(f"NIFTY {spot:.0f} is BELOW the range lower {cur['lower']:.0f} -- put side under pressure.")
        want_ce, want_pe = round(cur["upper"] / STEP) * STEP, round(cur["lower"] / STEP) * STEP
        for side, want in (("CE", want_ce), ("PE", want_pe)):
            leg = legs.get(side)
            if not leg:
                continue
            k = leg["strike"]
            dist = (k - spot) if side == "CE" else (spot - k)
            if dist < 0:
                tips.append(f"Sold {k} {side} is IN THE MONEY ({abs(dist):.0f} pts) -- your rule rolls it to {want} on the next leader change.")
            elif dist < 75:
                tips.append(f"Sold {k} {side} is only {dist:.0f} pts from NIFTY -- watch it closely.")
            if want != k:
                toward = want < k if side == "CE" else want > k
                tips.append(f"Range now points to {want} {side} (you hold {k}) -- by your rule: "
                            + ("roll toward it (more premium)." if toward else
                               "move away only if the leg goes in the money."))
    if verdict.endswith("BULLISH"):
        tips.append(("Leaning bullish" if verdict.startswith("MILD") else "Bullish read") + ": the PUT side is the safer side; risk is on the CALL. "
                    "Don't pull the call closer into strength; a put roll up (if the range shifts up) collects more.")
    elif verdict.endswith("BEARISH"):
        tips.append(("Leaning bearish" if verdict.startswith("MILD") else "Bearish read") + ": the CALL side is the safer side; risk is on the PUT. "
                    "Don't pull the put closer into weakness; a call roll down (if the range shifts down) collects more.")
    else:
        tips.append("Neutral read: range-bound conditions -- the setting your method earns in.")
    ivr = stats.get("iv_rank")
    if ivr is not None:
        if ivr < 20:
            tips.append(f"IV Rank {ivr} is low -- premiums are thin for the risk; a smaller size or a wider range is reasonable.")
        elif ivr > 60:
            tips.append(f"IV Rank {ivr} is high -- premiums are rich, but high IV often comes with big moves.")
    return {"verdict": verdict, "score": score, "signals": sigs, "suggestions": tips,
            "option_verdict": _verdict(option_score), "option_score": option_score, "option_signals": opt_sigs,
            "range_signal": range_sig,
            "note": "Signals from your range method and the option chain -- decision support, not a trade instruction."}


_first_high_cache = {}


def _first_candle_high(symbol):
    """High of the first 5-min candle (09:15) of the latest session with data
    for one option contract. Cached per symbol + day."""
    import pandas as pd
    import history_downloader as hd
    today = mc.now_ist().date()
    key = (symbol, today.isoformat())
    if key in _first_high_cache:
        return _first_high_cache[key]
    b = hd._get(hd.HIST_API, {"symbol": symbol, "resolution": "5", "date_format": "1",
                              "range_from": (today - pd.Timedelta(days=6)).isoformat(),
                              "range_to": today.isoformat(), "cont_flag": "1"}, tries=2)
    c = b.get("candles") or []
    out = None
    if c:
        t = pd.to_datetime([x[0] for x in c], unit="s", utc=True).tz_convert("Asia/Kolkata")
        last_day = t[-1].date()
        first = [x for x, tt in zip(c, t) if tt.date() == last_day]
        now = mc.now_ist()
        if last_day < today or (now.hour, now.minute) >= (9, 20):   # first candle complete
            out = {"high": float(first[0][2]), "day": last_day.isoformat()}
    if out:
        _first_high_cache[key] = out
    return out


def expiry_range(chain):
    """The user's range rules applied to ANY expiry's chain: 100-pt volume
    leaders (the session's cumulative volume) + their first 5-min candle
    highs. Used for expiries other than the one range_strategy tracks (e.g.
    next week's), where there is no engine state. None before 09:20."""
    rows = [r for r in chain.get("strikes") or [] if int(r["strike"]) % STEP == 0 and r.get("ce") and r.get("pe")]
    if not rows:
        return None
    ce = max(rows, key=lambda r: r["ce"].get("volume") or 0)
    pe = max(rows, key=lambda r: r["pe"].get("volume") or 0)
    if not (ce["ce"].get("volume") and pe["pe"].get("volume")):
        return None
    ce_hi, pe_hi = _first_candle_high(ce["ce"]["symbol"]), _first_candle_high(pe["pe"]["symbol"])
    if not ce_hi or not pe_hi:
        return None
    ce_k, pe_k = int(ce["strike"]), int(pe["strike"])
    return {"ce_leader": ce_k, "pe_leader": pe_k,
            "upper": round(ce_k + pe_hi["high"], 2), "lower": round(pe_k - ce_hi["high"], 2),
            "ce_first_high": ce_hi["high"], "pe_first_high": pe_hi["high"],
            "day": ce_hi["day"], "source": "this expiry's own volume leaders"}


def range_for_chain(chain, viewed_expiry_ts):
    """range_view() when the chain IS the expiry the range engine tracks;
    otherwise the same rules on this expiry's own volume leaders
    (expiry_range), shaped like range_view's output with own=True."""
    rv = range_view(chain, viewed_expiry_ts)
    if rv and rv.get("for_viewed_expiry"):
        return rv
    own = expiry_range(chain)
    if not own:
        return rv
    return {"own": True, "for_viewed_expiry": True, "expiry": None, "cycle_start": None,
            "official": own, "live": None, "history": [], "last_shift": None, "cycle_shift": None,
            "strategy": None, "engine_expiry": (rv or {}).get("expiry")}
