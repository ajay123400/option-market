"""intraday_range_bt.py -- intraday option-selling backtests on the downloaded
1-min history (data/hist1m/). Several rule-sets ("variants") run in ONE pass
over the data, so they're compared on exactly the same days and prices.

Common ground
  Leaders   The highest-volume 100-pt CE strike and PE strike (day's
            cumulative volume), first fixed at 09:20 after the first 5-min
            candle (09:15-09:19 bars), then re-checked every minute.
  Range     UPPER = CE leader + PE leader's first-5-min-candle high
            LOWER = PE leader - CE leader's first-5-min-candle high
  Fills     A decision on bar t's close fills at bar t+1's open, `slippage`
            points against us on every order. 1 lot, current-week expiry
            (expiry day included). P&L in points and rupees (the expiry's own
            lot size) net of charges (charges.py).
  Stop      Combined, per trade: exit every open leg when the trade's loss
            (booked + open) reaches `sl_frac` x the premium collected at its
            entry (e.g. 30%: collected 200 -> stopped when worth 260).
  Exit      15:15.

Variant keys
  entry     "range"  : sell the 50-pt strikes nearest UPPER (CE) / LOWER (PE)
            "leader" : sell the leader strikes themselves (CE leader + PE leader)
  roll      "range"  : range up -> book PE, sell PE nearest new LOWER;
                       range down -> book CE, sell CE nearest new UPPER
            None     : hold the entry strikes
  max_rolls    cap on rolls per day (default: no cap)
  roll_min_shift  roll only when the range mid moved at least this many points
  range_sl_min exit everything when NIFTY CLOSES a 5/15-min candle outside the
               current range (UPPER/LOWER, updated on leader changes); the
               premium stop stays on as an emergency stop
  expiry_only  True -> trade only on the expiry day itself
  weekdays  optional list (e.g. ["Tue"]) -- trade only on these days
  reentries after a stop, enter again (fresh strikes by the same `entry`
            rule, fresh stop) up to this many times, not after `last_entry`

Usage:  python intraday_range_bt.py
Output: results/intraday_range_bt/<variant>/{trades.csv, days.csv, days_detail.json, summary.json}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import charges
import paths
import simulator as S

OUT_DIR = os.path.join(paths.BASE_DIR, "results", "intraday_range_bt")
BASE = {"leader_step": 100, "sell_step": 50, "first_candle_min": 5, "exit_time": "15:15",
        "last_entry": "15:00", "slippage": 0.5, "lots": 1}
VARIANTS = {
    "leader_sl30_re1": {**BASE, "label": "Sell leader strikes · SL 30% · 1 re-entry",
                        "entry": "leader", "roll": None, "sl_frac": 0.30, "reentries": 1},
    "leader_sl30": {**BASE, "label": "Sell leader strikes · SL 30% · no re-entry",
                    "entry": "leader", "roll": None, "sl_frac": 0.30, "reentries": 0},
    "leader_sl50_tue": {**BASE, "label": "Sell leader strikes · SL 50% · no re-entry · Tuesdays only",
                        "entry": "leader", "roll": None, "sl_frac": 0.50, "reentries": 0, "weekdays": ["Tue"]},
    "leader_sl40_tue": {**BASE, "label": "Sell leader strikes · SL 40% · no re-entry · Tuesdays only",
                        "entry": "leader", "roll": None, "sl_frac": 0.40, "reentries": 0, "weekdays": ["Tue"]},
    "leader_sl60_tue": {**BASE, "label": "Sell leader strikes · SL 60% · no re-entry · Tuesdays only",
                        "entry": "leader", "roll": None, "sl_frac": 0.60, "reentries": 0, "weekdays": ["Tue"]},
    "leader_sl50": {**BASE, "label": "Sell leader strikes · SL 50% · no re-entry · all days",
                    "entry": "leader", "roll": None, "sl_frac": 0.50, "reentries": 0},
    "leader_sl40": {**BASE, "label": "Sell leader strikes · SL 40% · no re-entry · all days",
                    "entry": "leader", "roll": None, "sl_frac": 0.40, "reentries": 0},
    "leader_sl60": {**BASE, "label": "Sell leader strikes · SL 60% · no re-entry · all days",
                    "entry": "leader", "roll": None, "sl_frac": 0.60, "reentries": 0},
    "leader_sl30_exp": {**BASE, "label": "Sell leader strikes · SL 30% · no re-entry · expiry day only",
                        "entry": "leader", "roll": None, "sl_frac": 0.3, "reentries": 0, "expiry_only": True},
    "leader_sl40_exp": {**BASE, "label": "Sell leader strikes · SL 40% · no re-entry · expiry day only",
                        "entry": "leader", "roll": None, "sl_frac": 0.4, "reentries": 0, "expiry_only": True},
    "leader_sl50_exp": {**BASE, "label": "Sell leader strikes · SL 50% · no re-entry · expiry day only",
                        "entry": "leader", "roll": None, "sl_frac": 0.5, "reentries": 0, "expiry_only": True},
    "leader_sl60_exp": {**BASE, "label": "Sell leader strikes · SL 60% · no re-entry · expiry day only",
                        "entry": "leader", "roll": None, "sl_frac": 0.6, "reentries": 0, "expiry_only": True},
    "range_sl50_roll": {**BASE, "label": "Range strikes + rolls · SL 50% (previous test)",
                        "entry": "range", "roll": "range", "sl_frac": 0.50, "reentries": 0},
}


ORIG = {**BASE, "entry": "range", "roll": "range", "sl_frac": 0.50, "reentries": 0}
_ROLLS = {"full": ({}, "rolls"), "none": ({"roll": None}, "no rolls"), "max1": ({"max_rolls": 1}, "max 1 roll"),
          "min50": ({"roll_min_shift": 50}, "roll only if range moves 50+")}
_SLS = {"prem": ({}, "SL 50% premium"), "rsl5": ({"range_sl_min": 5}, "range SL on 5-min close"),
        "rsl15": ({"range_sl_min": 15}, "range SL on 15-min close")}
IMPROVE = {"imp_all_orig": {**ORIG, "label": "Original: range strikes, rolls, SL 50% · all days", "group": "all days"}}
for _k, (_c, _l) in list(_ROLLS.items())[1:]:
    IMPROVE[f"imp_all_{_k}"] = {**ORIG, **_c, "label": f"Original + {_l} · all days", "group": "all days"}
for _k, (_c, _l) in list(_SLS.items())[1:]:
    IMPROVE[f"imp_all_{_k}"] = {**ORIG, **_c, "label": f"Original + {_l} · all days", "group": "all days"}
for _rk, (_rc, _rl) in _ROLLS.items():
    for _sk, (_sc, _sl) in _SLS.items():
        IMPROVE[f"imp_exp_{_rk}_{_sk}"] = {**ORIG, **_rc, **_sc, "expiry_only": True, "group": "expiry day",
                                           "label": f"Expiry day · {_rl} · {_sl}"}


def _nearest(x, step, side):
    """Nearest `step` strike to x; an exact tie goes further OTM (CE up, PE down)."""
    lo = np.floor(x / step) * step
    hi = lo + step
    if abs(x - lo) < abs(hi - x):
        return int(lo)
    if abs(hi - x) < abs(x - lo):
        return int(hi)
    return int(hi) if side == "CE" else int(lo)


class Day:
    """One day's 1-min bars as arrays: per-contract close / fill / high, and
    the leader (highest cumulative volume, `leader_step` strikes) per minute."""

    def __init__(self, d, leader_step=100, first_candle_min=5, spot_ts=None, spot_close=None):
        d = d.sort_values("ts")
        ts = np.sort(d["ts"].unique())
        self.n = len(ts)
        self.hhmm = [pd.Timestamp(t, unit="s", tz="UTC").tz_convert(S.IST).strftime("%H:%M") for t in ts]
        self.mins = [int(h[:2]) * 60 + int(h[3:]) - 555 for h in self.hhmm]  # minutes since 09:15
        # NIFTY spot close at each bar (last print within 10 min, else NaN)
        self.spot = np.full(self.n, np.nan)
        if spot_ts is not None:
            j = np.searchsorted(spot_ts, ts, side="right") - 1
            ok = (j >= 0) & (ts - spot_ts[np.maximum(j, 0)] < 600)
            self.spot[ok] = spot_close[j[ok]]
        self.fc = first_candle_min
        self.arr = {}
        lead = {"CE": ([], []), "PE": ([], [])}
        for (typ, k), g in d.groupby(["type", "strike"], sort=False):
            idx = np.searchsorted(ts, g["ts"].to_numpy())
            close = np.full(self.n, np.nan); opn = np.full(self.n, np.nan); high = np.full(self.n, np.nan)
            vol = np.zeros(self.n)
            close[idx] = g["close"].to_numpy(); opn[idx] = g["open"].to_numpy()
            high[idx] = g["high"].to_numpy(); vol[idx] = g["volume"].to_numpy()
            c_ff = pd.Series(close).ffill().to_numpy()
            k = int(k)
            self.arr[(typ, k)] = {"close": c_ff, "fill": np.where(np.isnan(opn), c_ff, opn), "high": high}
            if k % leader_step == 0:
                lead[typ][0].append(k)
                lead[typ][1].append(np.cumsum(vol))
        self.leader = {}
        for typ, (ks, cv) in lead.items():
            if not ks:
                self.leader[typ] = [None] * self.n
                continue
            m = np.vstack(cv)
            best = m.argmax(axis=0)
            self.leader[typ] = [ks[b] if m[b, i] > 0 else None for i, b in enumerate(best)]

    def leaders(self, i):
        c, p = self.leader["CE"][i], self.leader["PE"][i]
        return (c, p) if c is not None and p is not None else None

    def fc_high(self, typ, k):
        a = self.arr.get((typ, k))
        if a is None:
            return None
        h = a["high"][:self.fc]
        return float(np.nanmax(h)) if np.isfinite(h).any() else None

    def rng(self, ld):
        ce_k, pe_k = ld
        ph, ch = self.fc_high("PE", pe_k), self.fc_high("CE", ce_k)
        if ph is None or ch is None:
            return None
        return {"upper": round(ce_k + ph, 2), "lower": round(pe_k - ch, 2), "ce_leader": ce_k, "pe_leader": pe_k,
                "ce_high": round(ch, 2), "pe_high": round(ph, 2)}

    def price(self, typ, k, i, what="fill"):
        a = self.arr.get((typ, k))
        if a is None:
            return None
        v = a[what][i]
        return float(v) if np.isfinite(v) else None


def run_day(day, expiry, D, cfg):
    if cfg.get("expiry_only") and day.isoformat() != expiry:
        return None, None
    if cfg.get("weekdays") and day.strftime("%a") not in cfg["weekdays"]:
        return None, None  # not a trading day for this variant (not counted as skipped)
    hhmm = D.hhmm
    if D.n < 30 or hhmm[0] != "09:15":
        return None, "no 09:15 data"
    i0 = D.fc - 1  # the 09:19 bar closes the first 5-min candle -> decide at 09:20
    try:
        i_exit = next(i for i, h in enumerate(hhmm) if h >= cfg["exit_time"])
    except StopIteration:
        return None, "no data at exit time"
    slip, lot = cfg["slippage"], S.lot_size_for(expiry)
    units = lot * cfg["lots"]
    closed, events = [], []

    def strikes_for(i):
        ld = D.leaders(i)
        if not ld:
            return None, None
        R = D.rng(ld)
        if cfg["entry"] == "leader":
            return (ld[0], ld[1]), R
        if not R:
            return None, None
        return (_nearest(R["upper"], cfg["sell_step"], "CE"), _nearest(R["lower"], cfg["sell_step"], "PE")), R

    trade_no, entries_left, dec = 0, 1 + cfg["reentries"], i0
    rolls, stops, last_stopped, stop_kind = 0, 0, False, "STOP LOSS"
    first_premium = first_range = None
    while entries_left > 0:
        e = dec + 1
        if e >= i_exit or (trade_no > 0 and hhmm[e] > cfg["last_entry"]):
            break
        ks, R = strikes_for(dec)
        if not ks:
            if trade_no == 0:
                return None, "no leaders / first-candle trade at 09:20"
            break
        prices = {t: D.price(t, k, e) for t, k in zip(("CE", "PE"), ks)}
        if None in prices.values():
            if trade_no == 0:
                return None, f"no price for {ks[0]} CE / {ks[1]} PE at 09:20"
            break
        trade_no += 1
        entries_left -= 1
        legs = {t: {"trade": trade_no, "type": t, "strike": k, "entry": round(prices[t] - slip, 2),
                    "entry_time": hhmm[e]} for t, k in zip(("CE", "PE"), ks)}
        booked = 0.0
        premium = legs["CE"]["entry"] + legs["PE"]["entry"]
        if first_premium is None:
            first_premium, first_range = premium, R
        events.append({"time": hhmm[e], "kind": "ENTRY" if trade_no == 1 else "RE-ENTRY", "trade": trade_no,
                       **(R or {}), "ce": ks[0], "pe": ks[1], "premium": round(premium, 2)})

        def buy_back(t, i, why):
            nonlocal booked
            leg = legs.pop(t)
            p = D.price(t, leg["strike"], i)
            leg.update(exit=round(p + slip, 2), exit_time=hhmm[i], exit_reason=why)
            leg["pnl_pts"] = round(leg["entry"] - leg["exit"], 2)
            booked += leg["pnl_pts"]
            closed.append(leg)

        cur_R = R
        stopped = False
        rsl = cfg.get("range_sl_min")
        track = cfg["roll"] == "range" or bool(rsl)
        for i in range(e, i_exit):
            open_val = sum(D.price(t, l["strike"], i, "close") or l["entry"] for t, l in legs.items())
            pnl = booked + sum(l["entry"] for l in legs.values()) - open_val
            why = None
            if pnl <= -cfg["sl_frac"] * premium:
                why = "STOP LOSS"
            elif rsl and cur_R and (D.mins[i] + 1) % rsl == 0 and np.isfinite(D.spot[i]) \
                    and not (cur_R["lower"] <= D.spot[i] <= cur_R["upper"]):
                why = "RANGE SL"
            if why:
                for t in list(legs):
                    buy_back(t, i + 1, why)
                events.append({"time": hhmm[i + 1], "kind": "STOP", "trade": trade_no, "pnl_pts": round(pnl, 2),
                               "why": why, "spot": None if not np.isfinite(D.spot[i]) else round(float(D.spot[i]), 2)})
                stopped, stops, dec = True, stops + 1, i + 1
                stop_kind = why
                break
            if not track or i + 1 >= i_exit:
                continue
            new = D.leaders(i)
            if not new or not cur_R or new == (cur_R["ce_leader"], cur_R["pe_leader"]):
                continue
            R2 = D.rng(new)
            if not R2:
                continue
            shift = (R2["upper"] + R2["lower"]) / 2 - (cur_R["upper"] + cur_R["lower"]) / 2
            ev = {"time": hhmm[i + 1], "kind": "LEADER", "trade": trade_no, **R2, "shift": round(shift, 2),
                  "prev_leaders": [cur_R["ce_leader"], cur_R["pe_leader"]]}
            cur_R = R2
            side = "PE" if shift > 0 else "CE" if shift < 0 else None
            can_roll = (cfg["roll"] == "range" and rolls < cfg.get("max_rolls", 10 ** 9)
                        and abs(shift) >= cfg.get("roll_min_shift", 0))
            if side and can_roll:
                k = _nearest(R2["lower"] if side == "PE" else R2["upper"], cfg["sell_step"], side)
                if k != legs[side]["strike"] and D.price(side, k, i + 1) is not None:
                    buy_back(side, i + 1, "ROLL UP" if side == "PE" else "ROLL DOWN")
                    legs[side] = {"trade": trade_no, "type": side, "strike": k,
                                  "entry": round(D.price(side, k, i + 1) - slip, 2), "entry_time": hhmm[i + 1]}
                    ev["roll"] = f"{side} -> {k}"
                    rolls += 1
            events.append(ev)
        last_stopped = stopped
        if not stopped:
            for t in list(legs):
                buy_back(t, i_exit, "TIME 15:15")
            break
    if not stops:
        exit_reason = "TIME 15:15"
    elif last_stopped:
        exit_reason = stop_kind if stops == 1 else f"{stops} STOPS"
    else:
        exit_reason = "STOP, RE-ENTRY -> TIME"

    pnl_pts = round(sum(c["pnl_pts"] for c in closed), 2)
    chg = sum(charges.round_trip("SELL", c["entry"], c["exit"], units) for c in closed)
    for c in closed:
        c.update(day=day.isoformat(), expiry=expiry, lot=lot, pnl_rs=round(c["pnl_pts"] * units, 2))
    row = {"day": day.isoformat(), "weekday": day.strftime("%a"), "expiry": expiry,
           "expiry_day": day.isoformat() == expiry, "lot": lot, "premium0": round(first_premium, 2),
           "range_920": [first_range["lower"], first_range["upper"]] if first_range else None,
           "trades": trade_no, "stops": stops, "rolls": rolls,
           "leader_changes": sum(1 for x in events if x["kind"] == "LEADER"),
           "exit": exit_reason, "pnl_pts": pnl_pts, "pnl_rs": round(pnl_pts * units, 2),
           "charges": round(chg, 2), "net_rs": round(pnl_pts * units - chg, 2), "events": events}
    return (row, closed), None


def _drawdown(series):
    eq = np.cumsum(series)
    peak = np.maximum.accumulate(np.concatenate([[0], eq]))[1:]
    return float((eq - peak).min()) if len(eq) else 0.0


def summarize(days):
    d = pd.DataFrame(days)
    if d.empty:
        return {}
    wins, losses = d[d.net_rs > 0], d[d.net_rs <= 0]
    by = lambda col: {str(k): {"days": int(len(g)), "net_rs": round(float(g.net_rs.sum()), 0),
                               "win_pct": round(float((g.net_rs > 0).mean() * 100), 1),
                               "pts": round(float(g.pnl_pts.sum()), 1)} for k, g in d.groupby(col)}
    d["year"] = d.day.str[:4]
    d["dte"] = np.where(d.expiry_day, "expiry day", "other days")
    return {
        "days": int(len(d)), "from": d.day.min(), "to": d.day.max(),
        "net_rs": round(float(d.net_rs.sum()), 0), "gross_rs": round(float(d.pnl_rs.sum()), 0),
        "charges": round(float(d.charges.sum()), 0), "points": round(float(d.pnl_pts.sum()), 1),
        "win_pct": round(len(wins) / len(d) * 100, 1),
        "avg_win": round(float(wins.net_rs.mean()), 0) if len(wins) else 0,
        "avg_loss": round(float(losses.net_rs.mean()), 0) if len(losses) else 0,
        "avg_premium": round(float(d.premium0.mean()), 1),
        "best_day": d.loc[d.net_rs.idxmax(), ["day", "net_rs"]].to_dict(),
        "worst_day": d.loc[d.net_rs.idxmin(), ["day", "net_rs"]].to_dict(),
        "max_drawdown_rs": round(_drawdown(d.net_rs.to_numpy()), 0),
        "stop_loss_days": int((d.stops > 0).sum()),
        "reentry_days": int((d.trades > 1).sum()),
        "avg_rolls": round(float(d.rolls.mean()), 2),
        "by_year": by("year"), "by_weekday": by("weekday"), "by_expiry_day": by("dte"),
    }


def run(variants=VARIANTS, date_from=None, date_to=None, verbose=True):
    man = S._manifest()
    exps = sorted(man)                       # includes failed expiries (their days are skipped)
    res = {v: {"days": [], "trades": [], "skipped": []} for v in variants}
    lstep = {c["leader_step"] for c in variants.values()}
    fcm = {c["first_candle_min"] for c in variants.values()}
    assert len(lstep) == 1 and len(fcm) == 1, "variants must share leader_step / first_candle_min"
    lstep, fcm = lstep.pop(), fcm.pop()
    sp = S._load_spot()
    spot_ts, spot_close = sp["ts"].to_numpy(), sp["close"].to_numpy()
    for n, exp in enumerate(exps):
        if (man[exp] or {}).get("status") != "done":
            continue
        prev = exps[n - 1] if n else None
        df = S._load_options(exp)
        mins = df["date"].dt.hour * 60 + df["date"].dt.minute
        df = df[mins <= 15 * 60 + 30]
        # days after the previous expiry (for the very first expiry: after its
        # downloaded week's start, which is the previous expiry day)
        lo = pd.Timestamp(prev).date() if prev else S._week_of(exp)[0]
        for day, g in df.groupby("day", sort=True):
            iso = day.isoformat()
            if (lo and day <= lo) or iso > exp:
                continue                     # only this expiry's own week
            if (date_from and iso < date_from) or (date_to and iso > date_to):
                continue
            if all(c.get("expiry_only") and iso != exp for c in variants.values()):
                continue  # no variant trades this day
            D = Day(g, lstep, fcm, spot_ts, spot_close)
            for v, cfg in variants.items():
                out, err = run_day(day, exp, D, cfg)
                if out is None and err is None:
                    continue
                if err:
                    res[v]["skipped"].append({"day": iso, "expiry": exp, "reason": err})
                    continue
                res[v]["days"].append(out[0])
                res[v]["trades"].extend(out[1])
        if verbose:
            print(f"\r{exp}: {len(next(iter(res.values()))['days'])} days", end="", flush=True)
    if verbose:
        print()
    summaries = {}
    for v, r in res.items():
        days = sorted(r["days"], key=lambda x: x["day"])
        summ = summarize([{k: val for k, val in x.items() if k != "events"} for x in days])
        summ.update(skipped=len(r["skipped"]), config=variants[v], key=v)
        out = os.path.join(OUT_DIR, v)
        os.makedirs(out, exist_ok=True)
        pd.DataFrame(r["trades"]).to_csv(os.path.join(out, "trades.csv"), index=False)
        pd.DataFrame([{k: val for k, val in x.items() if k != "events"} for x in days]).to_csv(os.path.join(out, "days.csv"), index=False)
        for name, obj in (("days_detail.json", days), ("skipped.json", r["skipped"]), ("summary.json", summ)):
            with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, default=str, indent=1 if name != "days_detail.json" else None)
        summaries[v] = summ
    return summaries


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv
    f = a[a.index("--from") + 1] if "--from" in a else None
    t = a[a.index("--to") + 1] if "--to" in a else None
    vs = IMPROVE if "--set" in a and a[a.index("--set") + 1] == "improve" else VARIANTS
    for v, s in run(vs, date_from=f, date_to=t).items():
        print(v, {k: s[k] for k in ("days", "net_rs", "gross_rs", "charges", "points", "win_pct",
                                     "max_drawdown_rs", "stop_loss_days", "reentry_days")})
