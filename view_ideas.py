"""view_ideas.py -- "My View": the user's own market view in, the best
DEFINED-RISK structures for it out.

The user states a view (direction + target, a range, or a big move), by when,
and the most they are willing to lose. Every candidate structure is priced
at the real bid/ask of the chain and judged on the view's own scenarios:

  right       NIFTY at the target on the chosen date (range: the worst point
              inside the range at expiry; big move: the worse of +/- the move)
  half right  NIFTY half-way to the target
  flat        NIFTY unchanged
  wrong       NIFTY moves the same distance the other way (range: half a
              range-width outside it; big move: NIFTY unchanged)

Naked long options and anything with unlimited loss are left out (the user's
rule). Before expiry, positions are valued with Black-76 at each leg's
current IV (sticky strike) -- an approximation; at expiry it is exact.

It cannot say whether the view is right -- the 5-year tests found no
indicator that predicts NIFTY's direction -- so the "wrong" columns matter as
much as the "right" ones.
"""
import math
from datetime import datetime

import numpy as np

import charges
import greeks as g
import market_calendar as mc
import strategy_ideas as SI

STEP = 50


def _k(x):
    return int(round(x / STEP) * STEP)


def _span(a, b):
    lo, hi = min(a, b), max(a, b)
    return list(range(_k(lo), _k(hi) + 1, STEP))


def candidates(view, spot, q):
    """[(name, tpl, [(side, type, strike), ...])] for the view."""
    out = []
    d = view["dir"]
    atm = _k(spot)
    if d in ("up", "down"):
        T = float(view["target"])
        up = d == "up"
        t, s = ("CE", 1) if up else ("PE", -1)
        # debit vertical: buy near the money, sell toward / past the target
        for k1 in _span(atm - 150 * s, T):
            for w in (50, 100, 150, 200, 300, 400):
                k2 = k1 + s * w
                if (up and k2 > T + 200) or (not up and k2 < T - 200):
                    continue
                out.append((f"{'Bull call' if up else 'Bear put'} spread {k1}/{k2}", "Debit spread",
                            [("BUY", t, k1), ("SELL", t, k2)]))
        # credit vertical on the other side: sell between spot and target's far side
        t2 = "PE" if up else "CE"
        for k2 in _span(atm - 300 * s, T):
            for w in (50, 100, 200, 300):
                k1 = k2 - s * w
                out.append((f"{'Bull put' if up else 'Bear call'} spread {k2}/{k1}", "Credit spread",
                            [("SELL", t2, k2), ("BUY", t2, k1)]))
        # butterflies centred on / near the target, and broken wings
        for c in (_k(T) - STEP, _k(T), _k(T) + STEP):
            for w in (50, 100, 150, 200, 300):
                out.append((f"{t} butterfly {c - w}/{c}/{c + w}", "Butterfly", [("BUY", t, c - w), ("SELL", t, c), ("SELL", t, c), ("BUY", t, c + w)]))
            for w1 in (100, 150, 200):
                for extra in (50, 100):
                    near, far = (c - s * w1), (c + s * (w1 + extra))
                    out.append((f"{t} broken-wing butterfly {near}/{c}/{far}", "Broken-wing butterfly",
                                [("BUY", t, near), ("SELL", t, c), ("SELL", t, c), ("BUY", t, far)]))
    elif d == "range":
        lo, hi = sorted((float(view["lo"]), float(view["hi"])))
        mid = _k((lo + hi) / 2)
        for ck in (_k(hi) - STEP, _k(hi), _k(hi) + STEP):
            for pk in (_k(lo) - STEP, _k(lo), _k(lo) + STEP):
                if ck <= pk:
                    continue
                for w in (100, 200, 300):
                    out.append((f"Iron condor {pk}/{ck} wings {w}", "Iron condor",
                                [("SELL", "CE", ck), ("BUY", "CE", ck + w), ("SELL", "PE", pk), ("BUY", "PE", pk - w)]))
        for w in (200, 300, 400):
            out.append((f"Iron butterfly {mid} wings {w}", "Iron butterfly",
                        [("SELL", "CE", mid), ("BUY", "CE", mid + w), ("SELL", "PE", mid), ("BUY", "PE", mid - w)]))
        for t in ("CE", "PE"):
            for w in (100, 150, 200, 300):
                out.append((f"{t} butterfly {mid - w}/{mid}/{mid + w}", "Butterfly", [("BUY", t, mid - w), ("SELL", t, mid), ("SELL", t, mid), ("BUY", t, mid + w)]))
        for w in (100, 200, 300):
            out.append((f"Bear call spread {_k(hi)}/{_k(hi) + w}", "Credit spread", [("SELL", "CE", _k(hi)), ("BUY", "CE", _k(hi) + w)]))
            out.append((f"Bull put spread {_k(lo)}/{_k(lo) - w}", "Credit spread", [("SELL", "PE", _k(lo)), ("BUY", "PE", _k(lo) - w)]))
    elif d == "move":
        M = float(view["move"])
        for a in (0, 50, 100, 150):
            for extra in (0, 100, 200):
                w = _k(M) + extra
                if w <= 0:
                    continue
                c1, p1 = atm + a, atm - a
                out.append((f"Reverse iron condor {p1}/{c1} (capped at ±{w})", "Reverse iron condor",
                            [("BUY", "CE", c1), ("SELL", "CE", c1 + w), ("BUY", "PE", p1), ("SELL", "PE", p1 - w)]))
    # drop unknown strikes, duplicates
    seen, res = set(), []
    for name, tpl, legs in out:
        key = tuple(sorted(legs))
        if key in seen or not all((k, t) in q for _, t, k in legs):
            continue
        seen.add(key)
        res.append((name, tpl, legs))
    return res


def _value(legs_px, S, T_rem, basis, q):
    """Position value per unit (what closing it would pay/cost) at NIFTY=S."""
    F = S + basis
    v = 0.0
    for side, typ, k, px in legs_px:
        if T_rem <= 0:
            val = max(0.0, S - k) if typ == "CE" else max(0.0, k - S)
        else:
            iv = (q.get((k, typ)) or {}).get("iv") or 0.15
            val = g.b76_price(F, k, T_rem, g.RISK_FREE_RATE, iv, typ == "CE")
        v += (-val if side == "SELL" else val)
    return v


def build(chain, view, now=None):
    now = now or mc.now_ist()
    ctx = SI.context(chain)
    q, fwd, T, lot = ctx["q"], ctx["fwd"], ctx["T"], ctx["lot"]
    spot = float(chain["spot"])
    basis = fwd - spot
    exp_ts = ctx["exp_ts"]
    # the view's date: T left at that date's close (0 = the expiry itself)
    by = view.get("by")
    if by:
        by_close = datetime.strptime(by + " 15:30", "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo).timestamp()
        T_by = max(0.0, (exp_ts - by_close) / (365 * 86400))
    else:
        T_by = 0.0
    max_loss = float(view["max_loss"]) if view.get("max_loss") else None
    em = fwd * ctx["sigma"] * math.sqrt(T)
    d = view["dir"]
    if d in ("up", "down"):
        tgt = float(view["target"])
        if (d == "up" and tgt <= spot) or (d == "down" and tgt >= spot):
            raise ValueError("target must be above NIFTY for an up view and below it for a down view")
        right, half, flat, wrong = tgt, spot + (tgt - spot) / 2, spot, spot - (tgt - spot)
        # a view is never exact: "right" = the worst P&L within +/- z of the target
        zone = max(25.0, 0.2 * abs(tgt - spot))
    rows = []
    for name, tpl, legs in candidates(view, spot, q):
        e = SI._evaluate({"name": name, "category": tpl, "direction": d, "tpl": tpl, "legs": legs}, q, fwd, T, lot,
                         ctx["grid"], ctx["dens_model"], ctx["dens_hist"], ctx["sigma"], mid=False)
        if not e or e["unlimited_loss"] or e["max_loss"] is None:
            continue
        if max_loss and e["max_loss"] < -max_loss:
            continue
        lp = [(l["side"], l["type"], l["strike"], l["price"]) for l in e["legs"]]
        credit = sum((l[3] if l[0] == "SELL" else -l[3]) for l in lp)
        chg = e["charges"]

        def pnl(S, Tr):
            return round((credit + _value(lp, S, Tr, basis, q)) * lot - chg)
        if d in ("up", "down"):
            sc = {"right": min(pnl(right - zone, T_by), pnl(right, T_by), pnl(right + zone, T_by)), "right_exp": pnl(right, 0), "half": pnl(half, T_by), "flat": pnl(flat, T_by), "wrong": pnl(wrong, T_by)}
        elif d == "range":
            lo, hi = sorted((float(view["lo"]), float(view["hi"])))
            inside = [pnl(x, 0) for x in np.linspace(lo, hi, 9)]
            w = hi - lo
            sc = {"right": min(inside), "right_exp": max(inside), "half": pnl((lo + hi) / 2, T_by), "flat": pnl(spot, T_by),
                  "wrong": min(pnl(lo - w / 2, 0), pnl(hi + w / 2, 0))}
        else:
            M = float(view["move"])
            sc = {"right": min(pnl(spot + M, T_by), pnl(spot - M, T_by)), "right_exp": min(pnl(spot + M, 0), pnl(spot - M, 0)),
                  "half": min(pnl(spot + M / 2, T_by), pnl(spot - M / 2, T_by)), "flat": pnl(spot, T_by), "wrong": pnl(spot, 0)}
        ml = e["max_loss"]
        rr = round(sc["right"] / -ml, 2) if ml < 0 and sc["right"] > 0 else None
        # where the trade turns profitable on the way to the target (up/down views)
        be_note = None
        if d in ("up", "down") and e["breakevens"]:
            toward = [b for b in e["breakevens"] if (b > spot if d == "up" else b < spot) and (b <= tgt + 400 if d == "up" else b >= tgt - 400)]
            if pnl(spot, 0) > 0:
                be_note = "already in profit at today's level"
            elif toward:
                b = min(toward) if d == "up" else max(toward)
                be_note = f"profitable from {b:,.0f} ({abs(b - spot):.0f} pts away, {abs(b - spot) / abs(tgt - spot) * 100:.0f}% of the way)"
        wide = [l["strike"] for l in e["legs"] if (lambda v: v and v["mid"] and (v["ask"] - v["bid"]) / v["mid"] > 0.10)(q.get((l["strike"], l["type"])))]
        rows.append({"name": name, "tpl": tpl, "legs": e["legs"], "net_premium": e["net_premium"], "credit": e["credit"],
                     "max_profit": e["max_profit"], "max_loss": ml, "breakevens": e["breakevens"], "be_note": be_note,
                     "pop_model": e["pop_model"], "pop_hist": e["pop_hist"], "greeks": e["greeks"], "curve": e["curve"],
                     "rr": rr, "illiquid": wide, **sc})
    good = [r for r in rows if r["right"] > 0]
    picks = []

    def pick(key, label, why, pool):
        pool = [r for r in pool if r not in [p["row"] for p in picks]]
        if pool:
            picks.append({"label": label, "why": why, "row": max(pool, key=key)})
    best_right = max((r["right"] for r in good), default=0)
    pick(lambda r: (r["rr"] or 0, r["right"]), "Best reward : risk", "most profit if right for each rupee at risk",
         [r for r in good if r["right"] >= 0.25 * best_right])
    pick(lambda r: (r["half"] + 0.5 * r["flat"], r["right"]), "Most forgiving", "still does best if the move is only half done or NIFTY stalls",
         [r for r in good if r["right"] >= 0.25 * best_right])
    pick(lambda r: (r["right"], r["rr"] or 0), "Most if right", "biggest profit if the view plays out", good)
    for p in picks:
        p.update(p.pop("row"))
    good.sort(key=lambda r: (r["rr"] or 0), reverse=True)
    env = {"spot": round(spot, 2), "fwd": round(fwd, 2), "expected_move": round(em), "sigma": round(ctx["sigma"] * 100, 2),
           "days_left": round(T * 365, 2), "lot": lot, "by": by, "n": len(rows), "n_right": len(good)}
    if d in ("up", "down"):
        dist = abs(float(view["target"]) - spot)
        env["target_vs_move"] = round(dist / em, 2) if em else None
        env["zone"] = round(zone)
    try:
        import sell_rules
        st = sell_rules.status(chain)
        env.update(iv_ok=st.get("iv_ok"), iv_live=st.get("live"), atm_iv=st.get("atm_iv"), iv_thr=st.get("iv_thr_now"),
                   window=st.get("window"), skew=st.get("skew25"), skew_steep=st.get("skew_steep"))
    except Exception:
        pass
    return {"env": env, "picks": picks, "table": good[:25]}
