"""positional_bt.py -- 5-year backtest of the user's POSITIONAL weekly range
method on the downloaded 1-min history (data/hist1m/).

Rules (user, 2026-09-26):
  Week      Day 1 = first trading day after the previous expiry; the position
            is held to EXPIRY (no day-before exit) and settles at intrinsic
            against NIFTY's expiry close, unless the stop or target hits first.
  Width     (min_width variants) enter only on a range at least this wide:
            the Day-1 09:20 range only, or (late_entry) the first range event
            -- Day-1 09:20 or any later leader change -- that is wide enough,
            up to the day before expiry (entry_on_expiry: expiry day too).
  Approach  (be* variants) breakeven-based: A = centred straddle moved with the
            range middle; B = current strikes + a breakeven guard that rolls a
            leg AWAY when a range edge comes within `guard` of its breakeven;
            C = balanced strangle re-balanced on every range change.
  Target    target_frac (0.7 default, None = hold to expiry); with
            target_pre_expiry_only the target is not checked on expiry day.
  Narrow    (exit_width variants) close everything as soon as a recalculated
            range is `exit_width` points wide or less.
  Hedge     (hedge variants) buy a CE `hedge_pts` above the sold call and a PE
            `hedge_pts` below the sold put; a roll moves the hedge with its
            leg (hedge_follow=False: bought once at entry and never moved).
            Hedged variants run WITHOUT a stop (the wings cap the loss);
            premium / target are on the NET credit.
  Leaders   Highest-volume 100-pt CE and PE strikes on each DAY's own volume
            (from 09:15), checked on every 5-min candle close from 09:20.
  Range     UPPER = CE leader + PE leader's first-5-min-candle high (that day)
            LOWER = PE leader - CE leader's first-5-min-candle high (that day)
            Recalculated ONLY when a leader changes (a new day with the same
            leaders keeps the running range).
  Entry     Day 1 09:20 -> sell the 50-pt strike nearest UPPER (CE) and
            nearest LOWER (PE), 1 lot each, no hedge.
  Adjust    Only when the range changes, and only TOWARD the market:
              call -> rolls DOWN to the strike nearest the new UPPER if that
                      is below the current call (range shifted down)
              put  -> rolls UP to the strike nearest the new LOWER if that
                      is above the current put (range shifted up)
            A leg never moves away. Limit: the call may come down to the put's
            strike (a straddle) and at most `cross_pts` (100) points below
            it -- no further; the same for the put going up.
  Stop      Total P&L (booked + open) <= -50% of the premium collected at entry.
  Target    NET P&L (after charges so far + estimated exit charges) >= 70% of
            the entry premium.
Fills at the next 1-min bar's open after the 5-min close that triggered them,
`slippage` points against us per order.

Usage:  python positional_bt.py [--example 2026-09-22]
Output: results/positional_bt/<variant>/{weeks.csv, trades.csv, weeks_detail.json, summary.json}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import charges
import paths
import simulator as S
from intraday_range_bt import Day, _nearest

OUT_DIR = os.path.join(paths.BASE_DIR, "results", "positional_bt")
BASE = {"leader_step": 100, "sell_step": 50, "check_min": 5, "sl_frac": 0.5, "target_frac": 0.7,
        "slippage": 0.5, "lots": 1, "cross_pts": 100, "roll_on_expiry_day": True}
VARIANTS = {"user_rules": {**BASE, "label": "No hedge · SL 50% (previous test)"},
            "nohedge_nosl": {**BASE, "sl_frac": None, "label": "No hedge · no SL"}}
for _w in (200, 300, 400, 500):
    VARIANTS[f"hedge{_w}_nosl"] = {**BASE, "sl_frac": None, "hedge_pts": _w,
                                   "label": f"Hedge {_w} pts, moved with rolls · no SL"}
for _w in (200, 300, 400, 500):
    VARIANTS[f"fixhedge{_w}_nosl"] = {**BASE, "sl_frac": None, "hedge_pts": _w, "hedge_follow": False,
                                      "label": f"Fixed hedge {_w} pts (bought at entry, never moved) · no SL"}
for _mw in (350, 400):
    VARIANTS[f"w{_mw}_day1"] = {**BASE, "sl_frac": None, "min_width": _mw,
                                "label": f"Enter only if Day-1 09:20 range >= {_mw} · no hedge · no SL"}
    VARIANTS[f"w{_mw}_late"] = {**BASE, "sl_frac": None, "min_width": _mw, "late_entry": True,
                                "label": f"Enter on first range >= {_mw} (Day 1 or later, till day before expiry) · no hedge · no SL"}
    VARIANTS[f"w{_mw}_late_exp"] = {**BASE, "sl_frac": None, "min_width": _mw, "late_entry": True, "entry_on_expiry": True,
                                    "label": f"Enter on first range >= {_mw} (any day incl. expiry day) · no hedge · no SL"}
for _k in ("nohedge_nosl", "w400_late", "w350_late", "w400_day1"):
    VARIANTS[f"{_k}_notgt"] = {**VARIANTS[_k], "target_frac": None, "label": VARIANTS[_k]["label"] + " · NO target"}
for _k in ("w400_late", "w400_late_notgt", "w350_late", "w350_late_notgt", "nohedge_nosl", "nohedge_nosl_notgt"):
    for _xw in (200, 250):
        VARIANTS[f"{_k}_x{_xw}"] = {**VARIANTS[_k], "exit_width": _xw, "label": VARIANTS[_k]["label"] + f" · exit if range <= {_xw}"}
for _k in ("w400_late", "w350_late"):
    for _t in (0.8, 0.9):
        VARIANTS[f"{_k}_t{int(_t * 100)}"] = {**VARIANTS[_k], "target_frac": _t,
                                              "label": VARIANTS[_k]["label"].replace("· no SL", f"· no SL · target {int(_t * 100)}%")}
    VARIANTS[f"{_k}_t70pre"] = {**VARIANTS[_k], "target_pre_expiry_only": True,
                                "label": VARIANTS[_k]["label"].replace("· no SL", "· no SL · target 70% until the day before expiry, then hold")}
_BEBASE = {**BASE, "sl_frac": None, "target_frac": 0.9}
for _f, _fc, _fl in (("w400", {"min_width": 400, "late_entry": True}, "≥400 late entry"), ("all", {}, "every week")):
    VARIANTS[f"beA50_{_f}"] = {**_BEBASE, **_fc, "approach": "A", "a_shift": 50, "label": f"A · centred straddle, move on 50+ shift · {_fl} · target 90%"}
    VARIANTS[f"beA100_{_f}"] = {**_BEBASE, **_fc, "approach": "A", "a_shift": 100, "label": f"A · centred straddle, move on 100+ shift · {_fl} · target 90%"}
    VARIANTS[f"beB50_{_f}"] = {**_BEBASE, **_fc, "approach": "B", "guard": 50, "label": f"B · breakeven guard 50 · {_fl} · target 90%"}
    VARIANTS[f"beB100_{_f}"] = {**_BEBASE, **_fc, "approach": "B", "guard": 100, "label": f"B · breakeven guard 100 · {_fl} · target 90%"}
    VARIANTS[f"beC_{_f}"] = {**_BEBASE, **_fc, "approach": "C", "label": f"C · balanced strangle · {_fl} · target 90%"}
VARIANTS["w400_late_t90"]["label"] = "Current best: range strikes + toward-rolls · ≥400 late entry · target 90%"
VARIANTS["nohedge_nosl_t90"] = {**BASE, "sl_frac": None, "target_frac": 0.9, "label": "Current rules · every week · target 90%"}
for _m in (2, 3):
    VARIANTS[f"nohedge_dstop{_m}x"] = {**BASE, "sl_frac": float(_m),
                                       "label": f"No hedge · disaster stop at {_m}x premium"}


def _balanced(D, i, U, L, step, slip):
    """CE/PE pair (CE above PE) whose breakevens sit equally far outside the
    range [L, U], as close to it as possible."""
    best = None
    lo, hi = int(L // step) * step - 400, int(U // step) * step + 450
    ce = {k: D.price("CE", k, i) for k in range(lo, hi, step)}
    pe = {k: D.price("PE", k, i) for k in range(lo, hi, step)}
    for kc, c in ce.items():
        if c is None:
            continue
        for kp, p in pe.items():
            if p is None or kp >= kc:
                continue
            cr = c + p - 2 * slip
            ou, od = kc + cr - U, L - (kp - cr)
            if ou < 0 or od < 0:
                continue
            sc = abs(ou - od) * 3 + ou + od
            if best is None or sc < best[0]:
                best = (sc, kc, kp)
    return (best[1], best[2]) if best else None


def run_week(exp, days, frames, spot, cfg):
    """days: trading days (date) of the week in order; frames: {day: Day}.
    Legs: "CE"/"PE" = the sold legs; "CEH"/"PEH" = the bought hedges
    (`hedge_pts` beyond the sold strike, moved together with it on a roll)."""
    lot = S.lot_size_for(exp)
    units = lot * cfg["lots"]
    step, slip = cfg["sell_step"], cfg["slippage"]
    W = cfg.get("hedge_pts")
    sl = cfg.get("sl_frac")
    legs, closed, events = {}, [], []
    st = {"booked": 0.0, "charges": 0.0, "premium": None, "R": None, "L": None, "exit": None}

    def sgn(key):
        return -1 if key in ("CE", "PE") else 1   # sold legs gain when price falls

    def opt(key):
        return key[:2]

    def open_leg(key, k, D, i, why, day):
        p = D.price(opt(key), k, i)
        if p is None:
            return None
        side = "SELL" if sgn(key) < 0 else "BUY"
        e = round(p - slip, 2) if side == "SELL" else round(p + slip, 2)
        legs[key] = {"type": opt(key), "side": side, "strike": k, "entry": e, "entry_time": f"{day} {D.hhmm[i]}", "why": why}
        st["charges"] += charges.order_charges(side, e, units)
        return e

    def close_leg(key, D, i, why, day, price=None):
        leg = legs.pop(key)
        if price is None:
            p = D.price(leg["type"], leg["strike"], i)
            x = round(p + slip, 2) if leg["side"] == "SELL" else round(max(0.0, p - slip), 2)
            st["charges"] += charges.order_charges("BUY" if leg["side"] == "SELL" else "SELL", x, units)
        else:
            x = round(price, 2)  # settled at expiry: no order
        pnl = (leg["entry"] - x) if leg["side"] == "SELL" else (x - leg["entry"])
        leg.update(exit=x, exit_time=f"{day} {D.hhmm[i]}" if D else f"{day} expiry", exit_reason=why, pnl_pts=round(pnl, 2))
        st["booked"] += leg["pnl_pts"]
        closed.append(leg)
        return x

    def hedge_strike(key, k, D, i):
        """Hedge strike W points beyond k; if that one has no quote, the next
        ones further out (up to +200) -- never closer."""
        if not W:
            return None
        d = 1 if key == "CEH" else -1
        for extra in range(0, 250, 50):
            h = k + d * (W + extra)
            if D.price(opt(key), h, i) is not None:
                return h
        return None

    def open_pts(D, i):
        tot = 0.0
        for l in legs.values():
            p = D.price(l["type"], l["strike"], i, "close")
            p = l["entry"] if p is None else p
            tot += (l["entry"] - p) if l["side"] == "SELL" else (p - l["entry"])
        return tot

    exp_day = pd.Timestamp(exp).date()
    for day in days:
        D = frames.get(day)
        if D is None or D.n < 30 or D.hhmm[0] != "09:15":
            continue
        is_exp = day == exp_day
        for i in range(D.n - 1):
            if (D.mins[i] + 1) % cfg["check_min"] or D.mins[i] < 4:
                continue  # only 5-min candle closes, from the 09:15-09:19 candle on
            ld = D.leaders(i)
            if not ld:
                continue
            changed = st["L"] is None or ld != st["L"]
            newR = None
            if changed:
                newR = D.rng(ld)
                if newR:
                    ev = {"time": f"{day} {D.hhmm[i + 1]}", "kind": "RANGE", **newR,
                          "prev": None if st["R"] is None else [st["R"]["lower"], st["R"]["upper"]],
                          "prev_leaders": st["L"], "spot": None if not np.isfinite(D.spot[i]) else round(float(D.spot[i]), 2)}
                    st["L"], st["R"] = ld, newR
                    events.append(ev)
            if st["exit"]:
                continue
            # entry: Day 1 at 09:20 -- or, with a width filter, the first range
            # event (Day-1 09:20 / any later leader change) at least `min_width` wide
            if st["premium"] is None:
                mw = cfg.get("min_width")
                if not mw:
                    if day != days[0] or not st["R"]:
                        if day != days[0]:
                            st["exit"] = "NO ENTRY"
                        continue
                else:
                    late = cfg.get("late_entry")
                    allowed = day == days[0] if not late else (not is_exp or cfg.get("entry_on_expiry"))
                    first_check = day == days[0] and D.mins[i] == 4
                    event = changed and newR is not None and (late or first_check)
                    if not allowed or not event or (st["R"]["upper"] - st["R"]["lower"]) < mw:
                        continue
                ap = cfg.get("approach")
                if ap == "A":
                    ce_k = pe_k = int(round((st["R"]["upper"] + st["R"]["lower"]) / 2 / step) * step)
                elif ap == "C":
                    bal = _balanced(D, i + 1, st["R"]["upper"], st["R"]["lower"], step, slip)
                    if not bal:
                        continue
                    ce_k, pe_k = bal
                else:
                    ce_k, pe_k = _nearest(st["R"]["upper"], step, "CE"), _nearest(st["R"]["lower"], step, "PE")
                if D.price("CE", ce_k, i + 1) is None or D.price("PE", pe_k, i + 1) is None:
                    continue
                ceh, peh = hedge_strike("CEH", ce_k, D, i + 1), hedge_strike("PEH", pe_k, D, i + 1)
                if W and (ceh is None or peh is None):
                    st["exit"] = "NO ENTRY (no hedge quote)"
                    continue
                a = open_leg("CE", ce_k, D, i + 1, "ENTRY", day)
                b = open_leg("PE", pe_k, D, i + 1, "ENTRY", day)
                hc = open_leg("CEH", ceh, D, i + 1, "HEDGE", day) if W else 0.0
                hp = open_leg("PEH", peh, D, i + 1, "HEDGE", day) if W else 0.0
                st["premium"] = a + b - hc - hp
                events.append({"time": f"{day} {D.hhmm[i + 1]}", "kind": "ENTRY", "ce": ce_k, "pe": pe_k,
                               "width": round(st["R"]["upper"] - st["R"]["lower"], 2), "late": day != days[0] or D.mins[i] != 4,
                               "ce_price": a, "pe_price": b, "ceh": ceh, "peh": peh, "ceh_price": hc, "peh_price": hp,
                               "premium": round(st["premium"], 2),
                               "stop": round(-sl * st["premium"], 2) if sl else None,
                               "target": round(cfg["target_frac"] * st["premium"], 2) if cfg.get("target_frac") else None})
                continue
            # risk checks on this close
            total = st["booked"] + open_pts(D, i)
            est_exit = 0.0
            for l in legs.values():
                p = D.price(l["type"], l["strike"], i, "close")
                est_exit += charges.order_charges("BUY" if l["side"] == "SELL" else "SELL", l["entry"] if p is None else p, units)
            net_rs = total * units - st["charges"] - est_exit
            why = ("STOP LOSS" if sl and total <= -sl * st["premium"] else
                   "TARGET" if cfg.get("target_frac") and not (is_exp and cfg.get("target_pre_expiry_only"))
                   and net_rs >= cfg["target_frac"] * st["premium"] * units else None)
            if why:
                for key in list(legs):
                    close_leg(key, D, i + 1, why, day)
                st["exit"] = why
                events.append({"time": f"{day} {D.hhmm[i + 1]}", "kind": why, "total_pts": round(total, 2),
                               "net_rs": round(net_rs, 2)})
                continue
            # narrow-range exit: the recalculated range has shrunk to `exit_width` or less
            if cfg.get("exit_width") and changed and newR and newR["upper"] - newR["lower"] <= cfg["exit_width"]:
                for key in list(legs):
                    close_leg(key, D, i + 1, "NARROW RANGE", day)
                st["exit"] = "NARROW RANGE"
                events.append({"time": f"{day} {D.hhmm[i + 1]}", "kind": "NARROW RANGE", "total_pts": round(total, 2),
                               "net_rs": round(net_rs, 2), "width": round(newR["upper"] - newR["lower"], 2)})
                continue
            # adjustments: only when the range changed, only toward the market
            if not (changed and newR) or (is_exp and not cfg["roll_on_expiry_day"]):
                continue
            rolls = []
            ap = cfg.get("approach")
            if ap in ("A", "B", "C"):
                def roll(key, k, lab):
                    if legs[key]["strike"] == k or D.price(key, k, i + 1) is None:
                        return
                    cur = legs[key]["strike"]
                    bx = close_leg(key, D, i + 1, lab, day)
                    sx = open_leg(key, k, D, i + 1, lab, day)
                    rolls.append(f"{key} {cur} -> {k} (bought back {bx}, sold {sx}) [{lab}]")

                def credit():
                    return sum(l["entry"] for l in legs.values()) + st["booked"]

                U, Lo = newR["upper"], newR["lower"]
                if ap == "A":
                    mid = (U + Lo) / 2
                    if abs(mid - legs["CE"]["strike"]) >= cfg.get("a_shift", 50):
                        nk = int(round(mid / step) * step)
                        if D.price("CE", nk, i + 1) is not None and D.price("PE", nk, i + 1) is not None:
                            roll("CE", nk, "RECENTRE")
                            roll("PE", nk, "RECENTRE")
                elif ap == "C":
                    bal = _balanced(D, i + 1, U, Lo, step, slip)
                    if bal:
                        roll("CE", bal[0], "REBALANCE")
                        roll("PE", bal[1], "REBALANCE")
                else:  # B: usual roll toward the market, then the breakeven guard
                    ce, pe = legs["CE"]["strike"], legs["PE"]["strike"]
                    tc = max(_nearest(U, step, "CE"), pe - cfg["cross_pts"])
                    if tc < ce:
                        roll("CE", tc, "ROLL DOWN")
                    tp = min(_nearest(Lo, step, "PE"), legs["CE"]["strike"] + cfg["cross_pts"])
                    if tp > pe:
                        roll("PE", tp, "ROLL UP")
                    g = cfg.get("guard", 50)
                    for key, d in (("CE", 1), ("PE", -1)):
                        k0 = legs[key]["strike"]
                        be = k0 + d * credit()
                        edge = U if d > 0 else Lo
                        if (edge - (be - d * g)) * d <= 0:
                            continue  # breakeven still at least `g` beyond the edge
                        bb = D.price(key, k0, i + 1)
                        if bb is None:
                            continue
                        k = k0
                        for _ in range(40):
                            k += d * step
                            p_ = D.price(key, k, i + 1)
                            if p_ is None:
                                continue
                            cr = credit() - (bb + slip) + (p_ - slip)
                            if (k + d * cr - edge) * d >= g:
                                roll(key, k, "GUARD AWAY")
                                break
                if rolls:
                    events[-1]["rolls"] = rolls
                    events[-1]["total_pts"] = round(total, 2)
                continue
            ce, pe = legs["CE"]["strike"], legs["PE"]["strike"]
            want_ce = max(_nearest(newR["upper"], step, "CE"), pe - cfg["cross_pts"])
            want_pe = min(_nearest(newR["lower"], step, "PE"), ce + cfg["cross_pts"])
            for key, cur, want, better, lab in (("CE", ce, want_ce, want_ce < ce, "ROLL DOWN"),
                                                ("PE", pe, want_pe, want_pe > pe, "ROLL UP")):
                if not better or D.price(key, want, i + 1) is None:
                    continue
                hk = key + "H"
                follow = W and cfg.get("hedge_follow", True)
                new_h = hedge_strike(hk, want, D, i + 1) if follow else None
                if follow and new_h is None:
                    continue  # can't re-hedge there -> don't roll
                bx = close_leg(key, D, i + 1, lab, day)
                sx = open_leg(key, want, D, i + 1, lab, day)
                txt = f"{key} {cur} -> {want} (bought back {bx}, sold {sx})"
                if follow and legs[hk]["strike"] != new_h:
                    old_h = legs[hk]["strike"]
                    hx = close_leg(hk, D, i + 1, "HEDGE MOVE", day)
                    hb = open_leg(hk, new_h, D, i + 1, "HEDGE MOVE", day)
                    txt += f", hedge {old_h} -> {new_h} (sold {hx}, bought {hb})"
                rolls.append(txt)
            if rolls:
                events[-1]["rolls"] = rolls
                events[-1]["total_pts"] = round(total, 2)
    # hold to expiry: settle open legs at intrinsic vs NIFTY's expiry close
    if legs:
        settle = spot
        for key in list(legs):
            k = legs[key]["strike"]
            intrinsic = max(0.0, settle - k) if legs[key]["type"] == "CE" else max(0.0, k - settle)
            close_leg(key, None, None, "EXPIRED", exp_day, price=intrinsic)
        st["exit"] = "EXPIRY"
        events.append({"time": f"{exp} close", "kind": "EXPIRY", "settle": settle})
    if st["premium"] is None:
        return None
    pts = round(sum(c["pnl_pts"] for c in closed), 2)
    for c in closed:
        c.update(expiry=exp, lot=lot, pnl_rs=round(c["pnl_pts"] * units, 2))
    ent = next(e for e in events if e["kind"] == "ENTRY")
    return {"expiry": exp, "day1": days[0].isoformat(), "days": len(days), "lot": lot,
            "entry_time": ent["time"], "entry_width": ent["width"], "late_entry": ent["late"],
            "premium0": round(st["premium"], 2), "exit": st["exit"],
            "rolls": sum(len(e.get("rolls", [])) for e in events),
            "range_changes": sum(1 for e in events if e["kind"] == "RANGE") - 1,
            "pnl_pts": pts, "pnl_rs": round(pts * units, 2), "charges": round(st["charges"], 2),
            "net_rs": round(pts * units - st["charges"], 2), "events": events}, closed


def _drawdown(x):
    eq = np.cumsum(x)
    return float((eq - np.maximum.accumulate(np.concatenate([[0], eq]))[1:]).min()) if len(eq) else 0.0


def summarize(weeks):
    d = pd.DataFrame(weeks)
    d["year"] = d.expiry.str[:4]
    by = {str(k): {"weeks": int(len(g)), "net_rs": round(float(g.net_rs.sum())), "pts": round(float(g.pnl_pts.sum()), 1),
                   "win_pct": round(float((g.net_rs > 0).mean() * 100), 1)} for k, g in d.groupby("year")}
    w, l = d[d.net_rs > 0], d[d.net_rs <= 0]
    return {"weeks": len(d), "from": d.expiry.min(), "to": d.expiry.max(),
            "net_rs": round(float(d.net_rs.sum())), "gross_rs": round(float(d.pnl_rs.sum())),
            "charges": round(float(d.charges.sum())), "points": round(float(d.pnl_pts.sum()), 1),
            "win_pct": round(len(w) / len(d) * 100, 1), "avg_win": round(float(w.net_rs.mean())) if len(w) else 0,
            "avg_loss": round(float(l.net_rs.mean())) if len(l) else 0,
            "max_drawdown_rs": round(_drawdown(d.net_rs.to_numpy())),
            "exits": d.exit.value_counts().to_dict(), "avg_rolls": round(float(d.rolls.mean()), 2),
            "avg_premium": round(float(d.premium0.mean()), 1), "by_year": by,
            "best": d.loc[d.net_rs.idxmax(), ["expiry", "net_rs"]].to_dict(),
            "worst": d.loc[d.net_rs.idxmin(), ["expiry", "net_rs"]].to_dict()}


def _expiry_close(st, sc, exp):
    e_end = int(pd.Timestamp(f"{exp} 23:59", tz=S.IST).timestamp())
    e_start = int(pd.Timestamp(f"{exp} 09:00", tz=S.IST).timestamp())
    j = np.searchsorted(st, e_end, side="right") - 1
    return float(sc[j]) if j >= 0 and st[j] >= e_start else None


def run(variants=VARIANTS, only=None, verbose=True):
    man = S._manifest()
    exps = sorted(man)
    sp = S._load_spot()
    st, sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
    res = {v: {"weeks": [], "trades": []} for v in variants}
    for n, exp in enumerate(exps):
        if (man[exp] or {}).get("status") != "done" or (only and exp not in only):
            continue
        prev = exps[n - 1] if n else None
        lo = pd.Timestamp(prev).date() if prev else S._week_of(exp)[0]
        settle = _expiry_close(st, sc, exp)
        if settle is None:
            continue
        df = S._load_options(exp)
        mins = df["date"].dt.hour * 60 + df["date"].dt.minute
        df = df[mins <= 15 * 60 + 30]
        days = [d for d in sorted(df["day"].unique()) if lo < d <= pd.Timestamp(exp).date()]
        if not days:
            continue
        frames = {d: Day(g, 100, 5, st, sc) for d, g in df[df["day"].isin(days)].groupby("day")}
        for v, cfg in variants.items():
            out = run_week(exp, days, frames, settle, cfg)
            if out:
                res[v]["weeks"].append(out[0])
                res[v]["trades"].extend(out[1])
        if verbose:
            print(f"\r{exp}", end="", flush=True)
    if verbose:
        print()
    sums = {}
    for v, r in res.items():
        if not r["weeks"] or only:
            continue
        out = os.path.join(OUT_DIR, v)
        os.makedirs(out, exist_ok=True)
        wk = sorted(r["weeks"], key=lambda x: x["expiry"])
        pd.DataFrame([{k: x for k, x in w.items() if k != "events"} for w in wk]).to_csv(os.path.join(out, "weeks.csv"), index=False)
        pd.DataFrame(r["trades"]).to_csv(os.path.join(out, "trades.csv"), index=False)
        s = summarize([{k: x for k, x in w.items() if k != "events"} for w in wk])
        s["config"] = variants[v]
        for name, obj in (("weeks_detail.json", wk), ("summary.json", s)):
            with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, default=str, indent=None if name == "weeks_detail.json" else 1)
        sums[v] = s
    return res, sums


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv
    if "--example" in a:
        exp = a[a.index("--example") + 1]
        res, _ = run(only={exp}, verbose=False)
        w = res["user_rules"]["weeks"][0]
        print({k: v for k, v in w.items() if k != "events"})
        for e in w["events"]:
            print(" ", e)
        for t in res["user_rules"]["trades"]:
            print("   ", t["type"], t["strike"], t["entry"], t["entry_time"], "->", t["exit"], t["exit_time"], t["exit_reason"], t["pnl_pts"])
    else:
        for v, s in run()[1].items():
            print(v, {k: s[k] for k in ("weeks", "net_rs", "gross_rs", "charges", "points", "win_pct", "max_drawdown_rs", "exits")})
