"""snapshot_ideas.py -- strategy ideas attached to every Range Diary snapshot
(09:20 day start and every leader change), and a follow-up of the ideas given
at earlier snapshots: what to do with each now.

At each snapshot, from the live option chain of the cycle's expiry:

  environment  IV level (where today's ATM IV sits among past entries of the
               same days-to-expiry), days to expiry, the market view
  top 3        1. best premium-selling idea WITH unlimited risk (naked)
               2. best DEFINED-risk idea
               3. best idea FOR THE VIEW (bullish / bearish), or the next best
               ranked by the similar-IV 5-year record (template_backtest):
               average P&L per rupee of the worst-5% trade; "good" = positive
               average and 55%+ wins; never an idea that bets against the view
  follow-up    every earlier idea still open is re-priced (exit at bid/ask)
               and checked:
                 BOOK   credit +50% of premium (debit +100%), or expiry day
                        after 14:30 with a profit
                 EXIT   loss of 2x the credit (debit -50%)
                 ADJUST a short strike is tested (NIFTY within 50 pts or
                        |delta| >= 0.40): roll that side away / close that
                        side / exit, whichever has the best expected P&L
                        from here after switching costs (with a tail-risk
                        penalty); on a range shift the UNtested side may be
                        rolled closer (your method: range down -> call down,
                        range up -> put up)
                 HOLD   otherwise
               The ideas are followed "virtually" (as if taken): a BOOK/EXIT
               closes it, an ADJUST replaces its legs, booked P&L carries on.

Nothing is traded -- suggestions only. Stored per expiry in
results/range_ideas/<expiry>.json; shown on the Range Diary and sent to Telegram.
"""
import json
import os
from datetime import datetime

import numpy as np

import charges
import paths
import strategy_ideas as SI

DIR = os.path.join(paths.BASE_DIR, "results", "range_ideas")
TEST_PTS, TEST_DELTA = 50, 0.40
TAIL_WEIGHT = 0.02       # how much the worst-5% loss counts when a strike is tested
SHIFT_MIN = 25           # range-mid move (pts) that counts as a range shift


# ---------------------------------------------------------------------------
# store
def _path(expiry):
    return os.path.join(DIR, f"{expiry}.json")


def load(expiry):
    try:
        with open(_path(expiry), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"rows": {}, "book": {}}


def _save(expiry, data):
    os.makedirs(DIR, exist_ok=True)
    paths.atomic_write_json(_path(expiry), data)


def row_key(row):
    return f"{row['time']}|{row['kind']}"


# ---------------------------------------------------------------------------
# environment + top 3
def environment(res):
    env = res["env"]
    pcts = [i["track"]["similar"]["iv_pct"] for i in res["ideas"] if (i.get("track") or {}).get("similar")]
    iv_pct = float(np.median(pcts)) if pcts else None
    level = None if iv_pct is None else ("High IV" if iv_pct >= 67 else "Low IV" if iv_pct <= 33 else "Normal IV")
    dte = env.get("track_dte")
    return {"atm_iv": env["atm_iv"], "iv_pct": iv_pct, "iv_level": level, "dte": dte,
            "dte_label": "expiry day" if dte == 0 else f"{dte} day{'s' if dte != 1 else ''} to expiry",
            "view": env["verdict"], "view_dir": env["view_direction"], "spot": env["spot"], "slot": env.get("track_slot")}


def _sim(i):
    t = i.get("track") or {}
    s = t.get("similar")
    return s if s and s.get("n", 0) >= 30 else None


def _score(i):
    s = _sim(i)
    return None if not s else s["avg_hold"] / max(500.0, abs(s["p5_hold"]))


def _good(i):
    s = _sim(i)
    return bool(s and s["avg_hold"] > 0 and s["win_hold"] >= 55)


def _against(i, vdir):
    return vdir in ("bullish", "bearish") and i["direction"] in ("bullish", "bearish") and i["direction"] != vdir


def pick_top3(res, env):
    vdir = env["view_dir"]
    pool = [i for i in res["ideas"] if _score(i) is not None and not _against(i, vdir)]
    pool.sort(key=_score, reverse=True)
    fam = lambda i: i["name"].split(" · ")[0]
    picks, used = [], set()

    def take(pred, slot, why):
        for strict in (True, False):
            for i in pool:
                if fam(i) in used or not pred(i) or (strict and not _good(i)):
                    continue
                used.add(fam(i))
                picks.append({"idea": i, "slot": slot, "why": why + ("" if strict else " (weak odds now)")})
                return

    take(lambda i: i["unlimited_loss"], "Best premium selling (naked)", "highest similar-IV record among naked sells")
    take(lambda i: not i["unlimited_loss"], "Best defined risk", "highest similar-IV record with capped loss")
    if vdir in ("bullish", "bearish"):
        take(lambda i: i["direction"] == vdir, f"Best for the {vdir} view", f"suits the {env['view'].lower()} view")
    if len(picks) < 3:
        take(lambda i: True, "Next best", "next highest similar-IV record")
    return picks


def _idea_record(p, idx, row, env):
    i, s = p["idea"], _sim(p["idea"])
    return {"id": f"{row['time'][5:16].replace('T', ' ')}#{idx}", "created": row["time"], "slot": p["slot"], "why": p["why"],
            "name": i["name"], "tpl": i["tpl"], "direction": i["direction"], "credit": i["credit"],
            "legs": [{k: l[k] for k in ("side", "type", "strike", "price", "symbol")} for l in i["legs"]],
            "net_premium": i["net_premium"], "max_profit": i["max_profit"], "max_loss": i["max_loss"],
            "unlimited_loss": i["unlimited_loss"], "breakevens": i["breakevens"],
            "record": None if not s else {k: s[k] for k in ("n", "win_hold", "avg_hold", "worst_hold", "p5_hold", "iv_lo", "iv_hi")},
            "dte0": env.get("dte"), "status": "open", "booked": 0.0, "last_pnl": 0.0, "history": []}


# ---------------------------------------------------------------------------
# follow-up of earlier ideas
def _exit_px(ctx, leg):
    v = ctx["q"].get((int(leg["strike"]), leg["type"]))
    if not v:
        return None
    return v["ask"] if leg["side"] == "SELL" else v["bid"]


def _pnl_now(ctx, idea):
    """Rupees if every open leg were closed now at bid/ask, plus what earlier
    adjustments already booked, minus exit charges."""
    lot, tot, chg = ctx["lot"], idea["booked"], 0.0
    for l in idea["legs"]:
        x = _exit_px(ctx, l)
        if x is None:
            return None
        sgn = 1 if l["side"] == "SELL" else -1
        tot += sgn * (l["price"] - x) * lot
        chg += charges.order_charges("BUY" if l["side"] == "SELL" else "SELL", x, lot)
    return round(tot - chg, 2)


def _tested(ctx, idea):
    spot, out = ctx["spot"], []
    for l in idea["legs"]:
        if l["side"] != "SELL":
            continue
        k = int(l["strike"])
        dist = (k - spot) if l["type"] == "CE" else (spot - k)
        v = ctx["q"].get((k, l["type"]))
        d = abs(v["g"]["delta"]) if v and v.get("g") else 0
        if dist < TEST_PTS or d >= TEST_DELTA:
            out.append((l["type"], k, round(dist), round(d, 2)))
    return out


def _switch_cost(ctx, remove, add):
    """Half-spread + charges of closing `remove` legs and opening `add` legs (1 lot each)."""
    lot, c = ctx["lot"], 0.0
    for side, typ, k in list(remove) + list(add):
        v = ctx["q"].get((k, typ))
        if not v:
            return None
        c += (v["ask"] - v["bid"]) / 2 * lot + charges.order_charges(side, v["mid"], lot)
    return c


def _eval(ctx, legs):
    if not legs:
        return {"ev": 0.0, "cvar": 0.0, "max_loss": 0.0}
    e = SI.evaluate_legs(ctx, legs, mid=True)
    if not e:
        return None
    return {"ev": e["ev_hist"] if e["ev_hist"] is not None else e["ev_model"], "cvar": e.get("cvar5") or 0.0,
            "max_loss": e["max_loss"]}


def _options(ctx, idea, tested, shift, rng):
    """Candidate actions -> [{action, detail, legs, ev, cost, cvar, score}]."""
    cur = [(l["side"], l["type"], int(l["strike"])) for l in idea["legs"]]
    out = []

    def add(action, detail, new_legs, removed, added):
        ev = _eval(ctx, new_legs)
        cost = _switch_cost(ctx, [("BUY" if s == "SELL" else "SELL", t, k) for s, t, k in removed], added)
        if ev is None or cost is None:
            return
        out.append({"action": action, "detail": detail, "legs": new_legs, "ev": round(ev["ev"], 0), "cost": round(cost, 0),
                    "cvar": round(ev["cvar"], 0), "max_loss": ev["max_loss"],
                    "score": ev["ev"] - cost + TAIL_WEIGHT * ev["cvar"]})

    add("HOLD", "keep as is", cur, [], [])
    add("EXIT", "close everything now", [], cur, [])
    for typ, k, dist, d in tested:
        side_legs = [x for x in cur if x[1] == typ]                    # this side: the short + its wing(s)
        other = [x for x in cur if x[1] != typ]
        new_short = SI._near_delta(ctx["q"], typ, 0.20, ctx["fwd"])
        if new_short and new_short != k:
            moved = [(s, t, kk + (new_short - k)) for s, t, kk in side_legs]
            add("ADJUST", f"roll {typ} side {k}→{new_short} (tested: {dist} pts away, Δ{d})", other + moved, side_legs, moved)
        if other:
            add("ADJUST", f"close the {typ} side only (tested)", other, side_legs, [])
    if shift and rng and not tested:
        # your method: range down -> bring the call closer; range up -> bring the put closer
        typ = "CE" if shift < 0 else "PE"
        shorts = [x for x in cur if x[1] == typ and x[0] == "SELL"]
        if shorts:
            k = shorts[0][2]
            want = int(round((rng["upper"] if typ == "CE" else rng["lower"]) / 50) * 50)
            if (typ == "CE" and want < k) or (typ == "PE" and want > k):
                side_legs = [x for x in cur if x[1] == typ]
                moved = [(s, t, kk + (want - k)) for s, t, kk in side_legs]
                add("ADJUST", f"range shifted {'down' if shift < 0 else 'up'}: roll {typ} {k}→{want} for more premium",
                    [x for x in cur if x[1] != typ] + moved, side_legs, moved)
    return out


def review(ctx, idea, env, shift, rng, now, sell=None):
    """What to do with an earlier idea at a later snapshot (user's rule,
    2026-09-29: ideas only at Day Start, adjustments at changes). Follows the
    backtests (results/ashish_rules, next_expiry_exit): on trades of up to 4
    days, holding to expiry beat profit targets, 2x stops and delta rolls --
    so the advice is HOLD, with the adjustment options listed for reference.
    The one tested exit: a trade opened 5+ trading days before expiry is
    closed once the IV rule reads "normal" (IV back below its threshold)."""
    pnl = _pnl_now(ctx, idea)
    if pnl is None:
        return {"id": idea["id"], "name": idea["name"], "pnl": None, "action": "HOLD", "reason": "no live quote for a leg"}
    base = abs(idea["net_premium"]) or 1
    r = {"id": idea["id"], "name": idea["name"], "pnl": pnl, "pct": round(pnl / base * 100, 0), "options": []}
    tested = _tested(ctx, idea)
    opts = _options(ctx, idea, tested, shift, rng) if (tested or shift) else []
    r["options"] = [{k: o[k] for k in ("action", "detail", "ev", "cost", "cvar", "max_loss")} for o in opts if o["action"] != "HOLD"]
    if (idea.get("dte0") or 0) >= 5 and sell and sell.get("window") == "ok" and sell.get("atm_iv") is not None             and sell.get("iv_thr_now") is not None and not sell.get("iv_ok"):
        r.update(action="EXIT", reason=f"IV back to normal ({sell['atm_iv']:.1f}% < {sell['iv_thr_now']:.1f}%) -- the tested exit for "
                                       "trades opened a week out")
        return r
    why = ("tested: " + ", ".join(f"{t} {k} ({d} pts, Δ{dl})" for t, k, d, dl in tested)) if tested else (
        f"range shifted {shift:+.0f} pts" if shift else "no strike tested, no range shift")
    r.update(action="HOLD", reason=f"{why} -- hold to expiry (in the 5-yr tests holding beat targets, stops and rolls on "
                                   "trades of up to 4 days); adjustment options below are for reference")
    return r


def _apply(ctx, idea, r, when):
    """Follow the suggestion virtually."""
    idea["last_pnl"] = r.get("pnl")
    idea["history"].append({"time": when, "action": r["action"], "pnl": r.get("pnl"), "reason": r.get("reason")})
    if r["action"] in ("BOOK", "EXIT") and r.get("pnl") is not None:
        idea["status"] = "booked" if r["action"] == "BOOK" else "exited"
        idea["result"] = r["pnl"]
    elif r["action"] == "ADJUST" and r.get("new_legs"):
        lot = ctx["lot"]
        old = {(l["side"], l["type"], int(l["strike"])): l for l in idea["legs"]}
        new = [tuple(x) for x in r["new_legs"]]
        for key, l in old.items():                     # close what's no longer in the position
            if key not in new:
                x = _exit_px(ctx, l)
                sgn = 1 if l["side"] == "SELL" else -1
                idea["booked"] += sgn * (l["price"] - x) * lot - charges.order_charges("BUY" if sgn > 0 else "SELL", x, lot)
        legs = []
        for s, t, k in new:
            if (s, t, k) in old:
                legs.append(old[(s, t, k)])
            else:
                v = ctx["q"][(k, t)]
                px = v["bid"] if s == "SELL" else v["ask"]
                idea["booked"] -= charges.order_charges(s, px, lot)
                legs.append({"side": s, "type": t, "strike": k, "price": round(px, 2), "symbol": v["symbol"]})
        idea["legs"] = legs
        idea["booked"] = round(idea["booked"], 2)


# ---------------------------------------------------------------------------
# snapshot entry point
def _stats(chain):
    import fyers_option_chain as chain_mod
    import greeks as g
    import time as _t
    stats = chain_mod.chain_stats(chain) or {}
    try:
        import eod_analysis
        rows, spot, exp_ts = chain["strikes"], chain["spot"], chain.get("resolved_expiry_ts")
        near = sorted(rows, key=lambda r: abs(r["strike"] - spot))[:6]
        fwd = g.synthetic_forward([(r["strike"], (r["ce"] or {}).get("ltp"), (r["pe"] or {}).get("ltp")) for r in near]) or spot
        T = (exp_ts - _t.time()) / (365 * 86400) if exp_ts else 0
        atm = min(rows, key=lambda r: abs(r["strike"] - fwd))
        ivs = [g.implied_vol_fwd((atm[k] or {}).get("ltp"), fwd, atm["strike"], T, g.RISK_FREE_RATE, k == "ce") for k in ("ce", "pe")] if T > 0 else []
        ivs = [v for v in ivs if v]
        if ivs:
            stats["atm_iv"] = round(sum(ivs) / len(ivs) * 100, 2)
            stats.update(eod_analysis.iv_rank(stats["atm_iv"]))
    except Exception:
        pass
    return stats


def on_snapshot(expiry, expiry_ts, row, now=None):
    """Called by the Range Strategy loop for a NEW live diary row. Returns the
    Telegram text (or None)."""
    import fyers_option_chain as chain_mod
    import market_view
    key = row_key(row)
    data = load(expiry)
    if key in data["rows"]:
        return None
    now = now or datetime.now()
    chain = chain_mod.get_chain(strikecount=30, expiry_timestamp=str(int(float(expiry_ts))))
    stats = _stats(chain)
    rv = view = None
    try:
        rv = market_view.range_view(chain, chain.get("resolved_expiry_ts"))
        view = market_view.market_view(chain, stats, rv, None)
    except Exception:
        pass
    rng = row.get("range") or {}
    res = SI.build(chain, stats, rng if rng.get("upper") else None, view)
    ctx = res["ctx"]
    env = environment(res)
    sell = None
    try:
        import sell_rules
        sell = sell_rules.status(chain)
        if sell.get("window") == "ok" and sell.get("atm_iv") is not None:
            env["iv_level"] = "Sell day (IV rule)" if sell.get("sell_day") else "Normal IV (IV rule)"
        elif sell.get("window") == "expiry_day":
            env["iv_level"] = "Expiry day (IV rule n/a)"
        elif sell.get("window") == "ok":
            env["iv_level"] = "IV rule: no live reading (market closed / before 09:30)"
        else:
            env["iv_level"] = f"{sell.get('dte')} days left (IV rule is for 1-4)"
    except Exception:
        pass
    # range shift since the previous snapshot of this cycle
    prev = [v for k, v in data["rows"].items() if k < key]
    shift = None
    if prev and rng.get("upper"):
        pr = prev[-1].get("range") or {}
        if pr.get("upper"):
            mv = (rng["upper"] + rng["lower"]) / 2 - (pr["upper"] + pr["lower"]) / 2
            shift = mv if abs(mv) >= SHIFT_MIN else None
    reviews = []
    for iid, idea in data["book"].items():
        if idea["status"] != "open":
            continue
        r = review(ctx, idea, env, shift, rng if rng.get("upper") else None, now, sell)
        _apply(ctx, idea, r, row["time"])
        reviews.append(r)
    # new ideas only at the day's first snapshot; later snapshots only review them
    picks = pick_top3(res, env) if row["kind"] == "DAY_START" else []
    ideas = [_idea_record(p, n + 1, row, env) for n, p in enumerate(picks)]
    if env.get("dte") == 0:
        for i in ideas:
            i["why"] += " · EXPIRY DAY: in the 5-yr test, entries on expiry day earned ~nothing on average"
    for i in ideas:
        data["book"][i["id"]] = i
    data["rows"][key] = {"time": row["time"], "kind": row["kind"], "env": env,
                         "range": {k: rng.get(k) for k in ("upper", "lower")} if rng.get("upper") else None,
                         "shift": None if shift is None else round(shift, 1),
                         "ideas": ideas, "reviews": reviews}
    _save(expiry, data)
    return fmt_telegram(env, ideas, reviews)


def on_day_end(expiry, expiry_ts, row, now=None):
    """DAY_END row: mark every open idea at the close; on expiry day settle
    them at NIFTY's close (intrinsic). No new ideas. Returns None."""
    import fyers_option_chain as chain_mod
    key = row_key(row)
    data = load(expiry)
    if key in data["rows"]:
        return None
    open_ideas = {k: v for k, v in data["book"].items() if v["status"] == "open"}
    reviews = []
    if open_ideas:
        import manual_trades
        chain = chain_mod.get_chain(strikecount=30, expiry_timestamp=str(int(float(expiry_ts))))
        spot = chain.get("spot")
        settle = row["time"][:10] == expiry
        # settling needs only NIFTY's close; marking needs the live context
        # (which refuses an expiry that is already over)
        ctx = None if settle else SI.build(chain, _stats(chain), None, None)["ctx"]
        lot = ctx["lot"] if ctx else manual_trades.LOT_SIZE
        for iid, idea in open_ideas.items():
            if settle and spot:
                val = sum((1 if l["side"] == "SELL" else -1) * (l["price"] - (max(0.0, spot - l["strike"]) if l["type"] == "CE"
                                                                             else max(0.0, l["strike"] - spot))) for l in idea["legs"])
                pnl = round(idea["booked"] + val * lot, 2)
                idea.update(status="expired", result=pnl, last_pnl=pnl)
                action, reason = "SETTLED", f"expired: NIFTY closed {spot:,.0f}"
            else:
                pnl = _pnl_now(ctx, idea)
                idea["last_pnl"] = pnl
                action, reason = "DAY END", "marked at the close"
            idea["history"].append({"time": row["time"], "action": action, "pnl": pnl, "reason": reason})
            reviews.append({"id": iid, "name": idea["name"], "pnl": pnl,
                            "pct": None if pnl is None else round(pnl / (abs(idea["net_premium"]) or 1) * 100, 0),
                            "action": action, "reason": reason, "options": []})
    data["rows"][key] = {"time": row["time"], "kind": row["kind"], "env": None, "range": None, "shift": None,
                         "ideas": [], "reviews": reviews}
    _save(expiry, data)
    return None


def _rs(v):
    return "—" if v is None else ("-" if v < 0 else "") + "₹" + f"{abs(v):,.0f}"


def fmt_telegram(env, ideas, reviews):
    if not ideas and not reviews:
        return None
    lines = [f"💡 Top 3 ideas (1 lot) — {env.get('iv_level') or 'IV ?'} (ATM IV {env['atm_iv']}%) · {env['dte_label']} · view {env['view']}"] if ideas         else [f"🔧 Adjustments — {env.get('iv_level') or ''} · {env['dte_label']}"]
    for n, i in enumerate(ideas, 1):
        legs = " + ".join(f"{'S' if l['side'] == 'SELL' else 'B'} {l['strike']}{l['type']} @{l['price']}" for l in i["legs"])
        rec = i.get("record")
        rec_t = f" · 5y at similar IV: {rec['win_hold']:.0f}% win, avg {_rs(rec['avg_hold'])}, worst {_rs(rec['worst_hold'])}" if rec else ""
        risk = "unlimited risk" if i["unlimited_loss"] else f"max loss {_rs(i['max_loss'])}"
        lines.append(f"{n}) {i['slot']}: {i['name']}\n   {legs} · {'credit' if i['credit'] else 'debit'} {_rs(abs(i['net_premium']))} · {risk}{rec_t}")
    if reviews:
        if ideas:
            lines.append("🔧 Earlier ideas:")
        icon = {"BOOK": "✅", "EXIT": "⛔", "ADJUST": "🔁", "HOLD": "⏸"}
        for r in reviews:
            pct = f" ({r['pct']:+.0f}%)" if r.get("pct") is not None else ""
            lines.append(f"• {icon.get(r['action'], '')} {r['action']} {r['name']} [{r['id']}]: {_rs(r.get('pnl'))}{pct} — {r.get('reason', '')}")
    return "\n".join(lines)
