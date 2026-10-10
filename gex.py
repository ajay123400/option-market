"""gex.py -- dealer gamma exposure (GEX) for NIFTY, INFORMATION ONLY.

The backtest (gex_research.py, results/gex/report.html) found no trading
edge in GEX beyond the IV rule, so nothing here feeds the plan; the /gex
page only shows the picture, the way dashboards like TradingAlgo's do.

  net GEX       sum over strikes of gamma x (call OI - put OI) x S^2 x 1%,
                in Rs crore per 1% NIFTY move (dealers assumed long the
                calls / short the puts customers trade)
  total gamma   the same with call OI + put OI
  flip          the NIFTY level where net GEX would cross zero (IVs held
                per strike)
  walls         the strikes with the largest call / put gamma exposure

Gamma: Black-76 on the put-call-parity forward, IV from the out-of-the-
money option at each strike (applied to both the call and the put there).

Sources: today = the live chain (Arrow), sampled every 5 minutes by a
background thread into results/gex/live/<day>.json; earlier days = the
1-min history (data/hist1m) at any time of day.
"""
import json
import math
import os
import threading
import time
import traceback
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import fyers_option_chain as chain_mod
import greeks as g
import market_calendar as mc
import paths

R = g.RISK_FREE_RATE
BAND = 0.08                         # strikes within +/-8% of NIFTY
GRID = np.linspace(-0.04, 0.04, 81)  # profile: NIFTY -4% .. +4%
LIVE_DIR = os.path.join(paths.BASE_DIR, "results", "gex", "live")
F_O_CLOSE_CHANGE = date(2026, 8, 3)
STEP_MIN = 5


# ---------------------------------------------------------------------------
def _gamma(F, K, T, iv):
    sq = iv * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * iv * iv * T) / sq
    return np.exp(-R * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (F * sq)


def exposure(spot, fwd, T, K, IV, OC, OP, detail=False):
    """Core numbers from per-strike arrays (IV as a fraction)."""
    gam = _gamma(fwd, K, T, IV)
    scale = spot * spot * 0.01 / 1e7                     # -> Rs crore per 1% move
    c, p = gam * OC * scale, gam * OP * scale
    net, tot = float(np.sum(c - p)), float(np.sum(c + p))
    nets = np.array([np.sum(_gamma(fwd * (1 + m), K, T, IV) * (OC - OP)) * (spot * (1 + m)) ** 2 * 0.01 / 1e7 for m in GRID])
    flip = None
    cross = np.where(np.diff(np.sign(nets)) != 0)[0]
    if len(cross):
        i = cross[np.argmin(np.abs(GRID[cross]))]
        m0 = GRID[i] - nets[i] * (GRID[i + 1] - GRID[i]) / (nets[i + 1] - nets[i])
        flip = round(float(spot * (1 + m0)), 1)
    out = {"spot": round(spot, 2), "fwd": round(fwd, 2), "net": round(net, 2), "tot": round(tot, 2), "flip": flip,
           "flip_dist_pct": round((spot / flip - 1) * 100, 2) if flip else None,
           "call_wall": int(K[np.argmax(c)]) if len(K) else None, "put_wall": int(K[np.argmax(p)]) if len(K) else None}
    if detail:
        out["strikes"] = [{"k": int(k), "call": round(float(a), 2), "put": round(float(b), 2), "iv": round(float(v) * 100, 2),
                           "oi_c": int(oc), "oi_p": int(op)} for k, a, b, v, oc, op in zip(K, c, p, IV, OC, OP)]
        out["profile"] = [{"level": round(spot * (1 + m), 1), "net": round(float(n), 2)} for m, n in zip(GRID, nets)]
    return out


def _arrays(rows, spot, fwd, T):
    """rows: {strike: {'c_px','p_px','oi_c','oi_p'}} -> K, IV, OC, OP (OTM-side IV)."""
    K, IV, OC, OP = [], [], [], []
    for k in sorted(rows):
        r = rows[k]
        if abs(k - spot) > BAND * spot or (not r.get("oi_c") and not r.get("oi_p")):
            continue
        iv = None
        order = (("c_px", True), ("p_px", False)) if k >= fwd else (("p_px", False), ("c_px", True))
        for key, call in order:
            px = r.get(key)
            if px and px >= 0.5:
                iv = g.implied_vol_fwd(float(px), fwd, k, T, R, call)
                if iv:
                    break
        if not iv:
            continue
        K.append(k); IV.append(iv); OC.append(r.get("oi_c") or 0); OP.append(r.get("oi_p") or 0)
    return np.array(K, float), np.array(IV, float), np.array(OC, float), np.array(OP, float)


def _expiry_close_ts(exp_iso):
    d = date.fromisoformat(exp_iso)
    t = "15:40" if d >= F_O_CLOSE_CHANGE else "15:30"
    return datetime.strptime(f"{exp_iso} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=mc.IST).timestamp()


# ---------------------------------------------------------------------------
# live (today)
def live_snapshot(expiry_ts=""):
    ch = chain_mod.get_chain(strikecount=40, expiry_timestamp=expiry_ts)
    spot, exp_ts = ch["spot"], ch.get("resolved_expiry_ts")
    T = (exp_ts - time.time()) / (365 * 86400) if exp_ts else None
    if not spot or not T or T <= 0:
        raise ValueError("no live chain / expiry")
    rows = {}
    for r in ch["strikes"]:
        ce, pe = r.get("ce") or {}, r.get("pe") or {}
        rows[int(r["strike"])] = {"c_px": ce.get("ltp"), "p_px": pe.get("ltp"), "oi_c": ce.get("oi") or 0, "oi_p": pe.get("oi") or 0}
    atm = min(rows, key=lambda k: abs(k - spot))
    near = [(k, rows[k]["c_px"], rows[k]["p_px"]) for k in sorted(rows) if abs(k - atm) <= 150 and rows[k]["c_px"] and rows[k]["p_px"]]
    fwd = g.synthetic_forward(near) or spot
    K, IV, OC, OP = _arrays(rows, spot, fwd, T)
    out = exposure(spot, fwd, T, K, IV, OC, OP, detail=True)
    out["expiry"] = datetime.fromtimestamp(exp_ts, mc.IST).date().isoformat()
    out["time"] = mc.now_ist().strftime("%H:%M")
    return out


def _live_path(day):
    return os.path.join(LIVE_DIR, f"{day}.json")


def sample_live():
    now = mc.now_ist()
    s = live_snapshot()
    rec = {k: s[k] for k in ("spot", "net", "tot", "flip", "call_wall", "put_wall")}
    rec.update(t=now.strftime("%H:%M"), expiry=s["expiry"])
    os.makedirs(LIVE_DIR, exist_ok=True)
    p = _live_path(now.date().isoformat())
    try:
        data = json.load(open(p))
    except (OSError, ValueError):
        data = {"samples": []}
    data["samples"] = [x for x in data["samples"] if x["t"] != rec["t"]] + [rec]
    paths.atomic_write_json(p, data)
    return rec


_thread = None


def _loop():
    while True:
        try:
            now = mc.now_ist()
            if mc.is_trading_day(now.date()) and "09:16" <= now.strftime("%H:%M") <= "15:31":
                sample_live()
        except Exception:
            traceback.print_exc()
        time.sleep(STEP_MIN * 60 - time.time() % (STEP_MIN * 60) + 5)   # on the 5-minute grid


def start_background():
    global _thread
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, daemon=True, name="gex-sampler")
    _thread.start()


# ---------------------------------------------------------------------------
# history (1-min files)
def _sim():
    import simulator as S
    return S


def history_days():
    """[{day, expiries:[...]}] for every trading day the 1-min option files cover (newest first)."""
    S = _sim()
    man = S._manifest()
    sp = S._load_spot()
    days = pd.Index(pd.to_datetime(sp["ts"], unit="s", utc=True).dt.tz_convert(mc.IST).dt.date.unique())
    out = {}
    for exp, r in man.items():
        if not isinstance(r, dict) or r.get("status") not in ("done", "live") or not r.get("week"):
            continue
        w0, w1 = date.fromisoformat(r["week"][0]), date.fromisoformat(exp)
        for d in days:
            if w0 < d <= w1:
                out.setdefault(d.isoformat(), []).append(exp)
    return [{"day": d, "expiries": sorted(v)} for d, v in sorted(out.items(), reverse=True)]


_hist_cache = {}


def history(day, expiry, at=None):
    """Intraday GEX series (every 5 min) for a past day + the full picture at `at` (HH:MM, default 15:25)."""
    key = (day, expiry, at)
    if key in _hist_cache:
        return _hist_cache[key]
    S = _sim()
    df = S._load_options(expiry)
    dd = df[df["day"] == date.fromisoformat(day)].sort_values("ts")
    if dd.empty:
        raise ValueError(f"no 1-min option data for {day} / {expiry}")
    sp = S._load_spot()
    st, sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
    exp_close = _expiry_close_ts(expiry)
    times = []
    t = datetime.strptime(f"{day} 09:20", "%Y-%m-%d %H:%M").replace(tzinfo=mc.IST)
    end = datetime.strptime(f"{day} 15:25", "%Y-%m-%d %H:%M").replace(tzinfo=mc.IST)
    while t <= end:
        times.append(t)
        t += timedelta(minutes=STEP_MIN)
    want = at or "15:25"
    series, snap = [], None
    for t in times:
        ts = int(t.timestamp())
        e = np.searchsorted(st, ts, side="right") - 1
        if e < 0 or st[e] < ts - 300:
            continue
        spot = float(sc[e])
        last = dd[dd.ts <= ts].groupby(["type", "strike"]).last()
        if last.empty:
            continue
        rows = {}
        for (typ, k), r in last.iterrows():
            x = rows.setdefault(int(k), {})
            x["c_px" if typ == "CE" else "p_px"] = float(r.close)
            x["oi_c" if typ == "CE" else "oi_p"] = float(r.oi)
        atm = min(rows, key=lambda k: abs(k - spot))
        near = [(k, rows[k].get("c_px"), rows[k].get("p_px")) for k in sorted(rows)
                if abs(k - atm) <= 150 and rows[k].get("c_px") and rows[k].get("p_px")]
        fwd = g.synthetic_forward(near) or spot
        T = (exp_close - ts) / (365 * 86400)
        if T <= 0:
            continue
        K, IV, OC, OP = _arrays(rows, spot, fwd, T)
        if len(K) < 8:
            continue
        hm = t.strftime("%H:%M")
        x = exposure(spot, fwd, T, K, IV, OC, OP, detail=(hm == want))
        series.append({"t": hm, **{k: x[k] for k in ("spot", "net", "tot", "flip", "call_wall", "put_wall")}})
        if hm == want:
            snap = dict(x, time=hm)
    if snap is None and series:                     # requested time missing: use the last sample with detail
        return history(day, expiry, series[-1]["t"])
    out = {"day": day, "expiry": expiry, "series": series, "snap": snap, "live": False}
    if len(_hist_cache) > 40:
        _hist_cache.clear()
    _hist_cache[key] = out
    return out


def live_now():
    """Live only on a trading day from 09:15: on weekends the broker feed carries NSE's mock-trading
    sessions (seen on Saturday 2026-10-10: NIFTY ticking ~22,367 vs Friday's 22,520 close)."""
    now = mc.now_ist()
    return mc.is_trading_day(now.date()) and now.strftime("%H:%M") >= "09:15"


def payload(day=None, expiry=None, at=None):
    today = mc.now_ist().date().isoformat()
    if not day and not live_now():                 # weekend / before the open: the last session from history
        last = next((d for d in history_days() if d["day"] < today), None)
        if last:
            out = history(last["day"], last["expiry"] if "expiry" in last else last["expiries"][0], at)
            return dict(out, note=f"Market not open — showing the last session ({last['day']}).")
    if not day or day == today:
        snap = live_snapshot()
        try:
            series = json.load(open(_live_path(today)))["samples"]
        except (OSError, ValueError):
            series = []
        return {"day": today, "expiry": snap["expiry"], "series": series, "snap": snap, "live": True}
    return history(day, expiry, at)
