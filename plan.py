"""plan.py -- "Aaj ka Plan": ONE place that turns the backtested rules into
today's decision, so nothing else in the app has to be read to know what to
do. Information only -- nothing is ever ordered.

Rules (all from the 5-year backtests; see results/*/report.html):
  IV rule     ATM IV of the nearest weekly expiry with 1-4 trading days left
              >= the 60th percentile of the past 500 days at that time of day
              (sell_rules.py). >= the 70th percentile = "strong".
  Setup A     IV rule green -> sell that expiry (straddle ATM / strangle
              0.15 delta), hold to expiry.
  Setup B     IV rule green at 09:30 AND the current expiry has 1 trading day
              left -> sell NEXT week's expiry, exit on the first check where
              the IV rule reads normal again (else hold to expiry).
  Otherwise   no new trade.
  Put skew    steep (25-delta put IV - call IV >= its 60th percentile) -> the
              put side did better.

The plan's own trades are followed virtually (results/plan/book.json): entered
at the first update after the setup appears (from 09:30), marked every 5
minutes, exited by their rule, settled at expiry -- a live scorecard to hold
next to the backtest.
"""
import json
import os
import threading
import time
import traceback
from datetime import datetime, timedelta

import charges
import fyers_option_chain as chain_mod
import manual_trades
import market_calendar as mc
import paths
import sell_rules as SR

DIR = os.path.join(paths.BASE_DIR, "results", "plan")
BOOK = os.path.join(DIR, "book.json")
SLIP = 0.5
STRATS = {"straddle": ("Short straddle|ATM", "Short straddle ATM"), "strangle": ("Short strangle|Δ0.15", "Short strangle 0.15Δ")}
# out-of-sample backtest numbers per 1 lot (reports: iv_surface / ashish_rules / next_expiry_exit)
TEST = {
    "A": {"straddle": {"avg": 4753, "win": 67, "p5": -20275, "dd": -109890, "strong_avg": 6025},
          "strangle": {"avg": 2378, "win": 86, "p5": -11083, "dd": -67361, "strong_avg": 3243},
          "normal_day_straddle": -11},
    "B": {"straddle": {"avg": 4880, "win": 74, "p5": -17660, "dd": -42308},
          "strangle": {"avg": 2863, "win": 85, "p5": -4381, "dd": -9361},
          "normal_day_straddle": -546},
}
_lock = threading.Lock()
_cache = {"t": 0, "v": None}


# ---------------------------------------------------------------------------
def _load_book():
    try:
        with open(BOOK) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"trades": {}}


def _save_book(b):
    os.makedirs(DIR, exist_ok=True)
    paths.atomic_write_json(BOOK, b)


def _expiries(now):
    out = []
    for e in chain_mod.list_expiries():
        d = datetime.strptime(e["date"], "%d-%m-%Y").date()
        if d < now.date():
            continue
        out.append({"date": d.isoformat(), "ts": str(int(float(e["expiry"]))), "dte": SR._dte(now.date(), d), "flag": e.get("expiry_flag")})
    return out


def _p70(bucket):
    h = SR.history()
    if h is None or not bucket:
        return None
    today = mc.now_ist().date().isoformat()
    v = h[(h.slot == bucket) & (h.day < today)].atm_iv.tail(SR.LOOKBACK)
    return round(float(v.quantile(0.7)), 2) if len(v) >= SR.MIN_ROWS else None


def _legkey(legs):
    return " ".join(f"{l['side'][0]}{l['strike']}{l['type']}" for l in legs)


def _trades_for(chain):
    """Straddle + strangle for this chain, priced like the Strategy Ideas table."""
    import strategy_ideas as SI
    try:
        res = SI.build(chain, chain_mod.chain_stats(chain), None, None)
    except Exception as e:
        print(f"[plan] cannot price trades: {e}")
        return {}
    out = {}
    for key, (tpl, label) in STRATS.items():
        it = next((i for i in res["ideas"] if i.get("tpl") == tpl), None)
        if not it:
            continue
        legs = [{k: l[k] for k in ("side", "type", "strike", "price", "symbol")} for l in it["legs"]]
        out[key] = {"label": label, "name": it["name"], "legs": legs, "net_premium": round(it["net_premium"], 2),
                    "breakevens": [round(b) for b in it.get("breakevens") or []], "legkey": _legkey(legs),
                    "record": (it.get("track") or {}).get("similar")}
    return out


def _nifty_close(day, today):
    """NIFTY's close on `day`: from the 1-min history, else Fyers daily
    candles, else (for today) the live quote."""
    try:
        import pandas as pd
        import simulator as S
        sp = S._load_spot()
        x = sp[pd.Index(sp.index.date).astype(str) == day]
        if len(x):
            return float(x.close.iloc[-1])
    except Exception:
        pass
    try:
        import historical_recorder
        c = historical_recorder._fetch_history_candles(chain_mod.INDEX_SYMBOL, day, day, "D")
        if c:
            return float(c[-1][4])
    except Exception:
        pass
    if day == today:
        q = chain_mod.get_quotes([chain_mod.INDEX_SYMBOL]).get(chain_mod.INDEX_SYMBOL) or {}
        return float(q["ltp"]) if q.get("ltp") else None
    return None


def _quote(chain, strike, typ):
    for r in chain.get("strikes") or []:
        if int(r["strike"]) == int(strike):
            q = r.get("ce" if typ == "CE" else "pe") or {}
            return q
    return {}


def _mark(chain, t):
    """(P&L now if closed at ask/bid, per-leg buy-back prices) for a short trade."""
    tot, ch, px = 0.0, 0.0, []
    for l in t["legs"]:
        q = _quote(chain, l["strike"], l["type"])
        p = q.get("ask") or q.get("ltp")
        if not p:
            return None, None
        p = float(p) + (0 if q.get("ask") else SLIP)
        px.append(p)
        tot += (l["price"] - p) * t["lot"]
        ch += charges.order_charges("BUY", p, t["lot"])
    return round(tot - ch - t["entry_charges"], 2), px


# ---------------------------------------------------------------------------
def compute(now=None):
    """Today's decision + the rule readings behind it."""
    now = now or mc.now_ist()
    mkt = mc.market_state().get("state")
    exps = _expiries(now)
    e1 = exps[0] if exps else None
    rule = next((e for e in exps if 1 <= e["dte"] <= 4), None)
    out = {"time": now.strftime("%Y-%m-%d %H:%M"), "market": mkt, "expiries": exps[:3], "current": e1, "rule_expiry": rule,
           "setups": [], "decision": None}
    if not rule:
        out["decision"] = {"kind": "none", "text": "No weekly expiry with 1–4 trading days left — the IV rule does not apply today."}
        return out
    chain = chain_mod.get_chain(strikecount=20, expiry_timestamp=rule["ts"])
    st = SR.status(chain)
    st["p70_now"] = _p70(st.get("bucket"))
    st["strong"] = bool(st.get("iv_ok") and st.get("p70_now") and st["atm_iv"] >= st["p70_now"])
    out["rule"] = st
    if not st.get("live"):
        ld = st.get("last_day") or {}
        out["decision"] = {"kind": "wait", "text": "Market closed — the plan is decided from 09:30 while the market is open."
                           if mkt != "OPEN" else "Before 09:30 — the plan is decided at 09:30.", "last_day": ld}
        return out
    green = st["sell_day"]
    today = now.date().isoformat()
    done = [t for t in _load_book()["trades"].values() if t["day"] == today and t.get("status") != "void"]
    if done:
        first = min(done, key=lambda t: t["time"])
        out["decision"] = {"kind": "sell", "text": f"Today's trade was entered at {first['time']} (setup {first['setup']}) — see the open trades below",
                           "strong": st["strong"], "skew_steep": st.get("skew_steep")}
        return out
    if green and not st.get("at_check"):
        # the tested rule reads IV only at 09:30 / 12:00 / 14:30 -- a crossing in between is not a signal yet
        out["decision"] = {"kind": "none", "text": f"WATCH — IV {st['atm_iv']:.1f}% is above the threshold now, but not at a check time",
                           "why": f"The rule was tested on the 09:30 / 12:00 / 14:30 readings only. Next check: {st.get('next_check') or 'none left today'}.",
                           "note": "If IV is still above that check's threshold then, the plan will show the trade."}
        return out
    b_ok = green and e1 and e1["dte"] == 1 and st["bucket"] == "09:30"
    if green:
        if b_ok:
            nxt = next((e for e in exps if e["date"] > e1["date"]), None)
            if nxt:
                ch2 = chain_mod.get_chain(strikecount=20, expiry_timestamp=nxt["ts"])
                out["setups"].append({"id": "B", "title": f"Sell NEXT week's expiry ({nxt['date']})", "expiry": nxt,
                                      "why": f"green IV at 09:30 and the {e1['date']} expiry has 1 trading day left",
                                      "exit": "Exit on the first check (09:30 / 12:00 / 14:30) where the IV rule reads normal; otherwise hold to expiry.",
                                      "exit_rule": "ivnormal", "trades": _trades_for(ch2), "test": TEST["B"], "primary": True})
        out["setups"].append({"id": "A", "title": f"Sell this week's expiry ({rule['date']}, {rule['dte']} day{'s' if rule['dte'] != 1 else ''} left)",
                              "expiry": rule, "why": "green IV" + (" — STRONG (≥ 70th percentile)" if st["strong"] else ""),
                              "exit": "Hold to expiry (in the tests, targets, 2× stops and rolls all lowered the result on trades of up to 4 days).",
                              "exit_rule": "hold", "trades": _trades_for(chain), "test": TEST["A"], "primary": not b_ok})
        out["decision"] = {"kind": "sell", "text": "SELL — " + ("next week's expiry (setup B) — or this week's (A)" if b_ok else "this week's expiry (setup A)"),
                           "strong": st["strong"], "skew_steep": st.get("skew_steep")}
    else:
        why = f"IV {st['atm_iv']:.1f}% is below today's threshold {st['iv_thr_now']:.1f}% ({st['bucket']} bucket)" if st.get("atm_iv") else "IV unavailable"
        out["decision"] = {"kind": "none", "text": "NO NEW TRADE — normal IV day", "why": why,
                           "note": f"In the tests, straddles on normal days made ~₹{TEST['A']['normal_day_straddle']} a trade; "
                                   f"selling next week on normal days lost (~₹{TEST['B']['normal_day_straddle']})."}
    return out


# ---------------------------------------------------------------------------
def update(now=None):
    """Record today's plan trades once, follow the open ones. Called every
    5 minutes by the background thread (and by /api/plan)."""
    now = now or mc.now_ist()
    with _lock:
        p = compute(now)
        book = _load_book()
        today = now.date().isoformat()
        lot = manual_trades.LOT_SIZE
        # 1) record the day's setups (once per setup x strategy)
        for s in p["setups"]:
            for key, t in s["trades"].items():
                tid = f"{today}|{s['id']}|{key}"
                if tid in book["trades"]:
                    continue
                ch = sum(charges.order_charges("SELL", l["price"], lot) for l in t["legs"])
                book["trades"][tid] = {"id": tid, "day": today, "time": now.strftime("%H:%M"), "setup": s["id"], "strategy": key,
                                       "name": t["name"], "expiry": s["expiry"]["date"], "expiry_ts": s["expiry"]["ts"],
                                       "exit_rule": s["exit_rule"], "legs": t["legs"], "lot": lot, "entry_charges": round(ch, 2),
                                       "credit": round(t["net_premium"], 2), "iv_at_entry": p["rule"]["atm_iv"],
                                       "status": "open", "pnl": 0.0, "history": []}
        # 2) follow open trades
        rule = p.get("rule") or {}
        chains = {}
        for tid, t in book["trades"].items():
            if t["status"] != "open":
                continue
            if t["expiry"] < today or (t["expiry"] == today and p["market"] != "OPEN" and now.strftime("%H:%M") >= "15:40"):
                # settle at NIFTY's close on expiry day
                spot = _nifty_close(t["expiry"], today)
                if not spot:
                    continue
                val = sum(l["price"] - (max(0.0, spot - l["strike"]) if l["type"] == "CE" else max(0.0, l["strike"] - spot)) for l in t["legs"])
                t.update(status="expired", pnl=round(val * t["lot"] - t["entry_charges"], 2), closed=f"{t['expiry']} close",
                         close_reason=f"expired, NIFTY {spot:,.0f}")
                continue
            if t["expiry"] not in chains:
                try:
                    chains[t["expiry"]] = chain_mod.get_chain(strikecount=30, expiry_timestamp=t["expiry_ts"])
                except Exception:
                    continue
            pnl, px = _mark(chains[t["expiry"]], t)
            if pnl is None:
                continue
            t["pnl"], t["marked"] = pnl, now.strftime("%Y-%m-%d %H:%M")
            # setup B exit: IV rule back to normal at a check (not on the entry day's own reading)
            if t["exit_rule"] == "ivnormal" and rule.get("live") and rule.get("window") == "ok" and t["day"] < today \
                    and rule.get("atm_iv") is not None and not rule.get("iv_ok"):
                exit_ch = sum(charges.order_charges("BUY", x, t["lot"]) for x in px)
                t.update(status="closed", pnl=round(pnl, 2), closed=now.strftime("%Y-%m-%d %H:%M"),
                         close_reason=f"IV normal ({rule['atm_iv']:.1f}% < {rule['iv_thr_now']:.1f}%)", exit_charges=round(exit_ch, 2))
        _save_book(book)
        p["book"] = book
        return p


def payload():
    """For /api/plan: the decision, the setups, open trades with today's action, the scorecard."""
    if time.time() - _cache["t"] < 30 and _cache["v"]:
        return _cache["v"]
    p = update()
    trades = sorted(p["book"]["trades"].values(), key=lambda t: (t["day"], t["setup"], t["strategy"]), reverse=True)
    today = mc.now_ist().date().isoformat()
    rule = p.get("rule") or {}
    for t in trades:
        if t["status"] != "open":
            continue
        if t["expiry"] == today:
            t["action"] = "HOLD — expires today"
        elif t["exit_rule"] == "ivnormal":
            t["action"] = ("EXIT — IV back to normal" if rule.get("live") and rule.get("window") == "ok" and not rule.get("iv_ok") and t["day"] < today
                           else "HOLD — exit when the IV rule reads normal at a check")
        else:
            t["action"] = "HOLD to expiry"
    closed = [t for t in trades if t["status"] not in ("open", "void")]
    score = {}
    for key in ("A", "B"):
        c = [t for t in closed if t["setup"] == key]
        if c:
            score[key] = {"n": len(c), "total": round(sum(t["pnl"] for t in c)), "win": round(sum(t["pnl"] > 0 for t in c) / len(c) * 100),
                          "avg": round(sum(t["pnl"] for t in c) / len(c))}
    out = {**{k: v for k, v in p.items() if k != "book"}, "open": [t for t in trades if t["status"] == "open"],
           "closed": closed[:40], "score": score, "test": TEST, "lot": manual_trades.LOT_SIZE}
    _cache.update(t=time.time(), v=out)
    return out


# ---------------------------------------------------------------------------
_thread = None


def _loop():
    while True:
        try:
            now = mc.now_ist()
            if mc.is_trading_day(now.date()) and "09:30" <= now.strftime("%H:%M") <= "15:50":
                update(now)
        except Exception:
            traceback.print_exc()
        time.sleep(300)


def start_background():
    global _thread
    if _thread is not None and _thread.is_alive():
        return _thread
    _thread = threading.Thread(target=_loop, daemon=True, name="plan")
    _thread.start()
    return _thread
