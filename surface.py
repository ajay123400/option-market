"""surface.py -- NIFTY intraday implied-volatility surface, INFORMATION ONLY.

For a day and up to two expiries: the IV smile every 5 minutes from 09:20 to
15:25, on a moneyness grid (strike / forward - 1, -5% .. +5%), plus the
ATM-forward IV and where the smile is lowest, through the day. Shown on the
/surface page like TradingAlgo's 3D vol surface. The backtest of picking
strikes from the IV surface (iv_surface_research.py) found no edge, so
nothing here feeds the plan.

IV per strike = Black-76 on the put-call-parity forward, from the
out-of-the-money option (its price is the reliable one).
Sources: today = the live chain (Arrow) sampled every 5 minutes into
results/surface/live/<day>.json; earlier days = the 1-min history.
"""
import json
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
GRID = np.round(np.arange(-5.0, 5.0001, 0.25), 2)       # moneyness, %
STEP_MIN = 5
LIVE_DIR = os.path.join(paths.BASE_DIR, "results", "surface", "live")
INDEX_PATH = os.path.join(paths.BASE_DIR, "results", "surface", "coverage.json")
F_O_CLOSE_CHANGE = date(2026, 8, 3)


def _close_ts(exp_iso):
    t = "15:40" if date.fromisoformat(exp_iso) >= F_O_CLOSE_CHANGE else "15:30"
    return datetime.strptime(f"{exp_iso} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=mc.IST).timestamp()


def smile(rows, spot, fwd, T):
    """rows {strike: {'c_px','p_px'}} -> {'grid': [iv% or None per GRID point], 'atm', 'min_m', 'min_iv'}."""
    ks, ivs = [], []
    for k in sorted(rows):
        m = (k / fwd - 1) * 100
        if m < GRID[0] - 0.5 or m > GRID[-1] + 0.5:
            continue
        r = rows[k]
        key, call = ("c_px", True) if k >= fwd else ("p_px", False)
        px = r.get(key)
        if not px or px < 0.5:
            continue
        iv = g.implied_vol_fwd(float(px), fwd, k, T, R, call)
        if iv and 0.02 < iv < 2.0:
            ks.append(m); ivs.append(iv * 100)
    if len(ks) < 5:
        return None
    ks, ivs = np.array(ks), np.array(ivs)
    grid = np.interp(GRID, ks, ivs, left=np.nan, right=np.nan)
    atm = float(np.interp(0.0, ks, ivs))
    j = int(np.argmin(ivs))
    return {"grid": [None if np.isnan(v) else round(float(v), 2) for v in grid], "atm": round(atm, 2),
            "min_m": round(float(ks[j]), 2), "min_iv": round(float(ivs[j]), 2)}


def _fwd(rows, spot):
    atm = min(rows, key=lambda k: abs(k - spot))
    near = [(k, rows[k].get("c_px"), rows[k].get("p_px")) for k in sorted(rows)
            if abs(k - atm) <= 150 and rows[k].get("c_px") and rows[k].get("p_px")]
    return g.synthetic_forward(near) or spot


# ---------------------------------------------------------------------------
# history
def _sim():
    import simulator as S
    return S


def coverage():
    """{expiry: [first_day, last_day]} for every 1-min option file (cached; refreshed when files change)."""
    import pyarrow.parquet as pq
    S = _sim()
    opt_dir = os.path.join(paths.BASE_DIR, "data", "hist1m", "options")
    try:
        cache = json.load(open(INDEX_PATH))
    except (OSError, ValueError):
        cache = {}
    changed = False
    for f in os.listdir(opt_dir):
        if not f.endswith(".parquet"):
            continue
        exp, p = f[:-8], os.path.join(opt_dir, f)
        mt = os.path.getmtime(p)
        if cache.get(exp, {}).get("mtime") == mt:
            continue
        md = pq.ParquetFile(p).metadata
        col = md.schema.names.index("ts")
        lo, hi = [], []
        for i in range(md.num_row_groups):
            st = md.row_group(i).column(col).statistics
            if st is not None and st.has_min_max:
                lo.append(st.min); hi.append(st.max)
        if not lo:
            ts = pd.read_parquet(p, columns=["ts"]).ts
            lo, hi = [ts.min()], [ts.max()]
        d = lambda x: datetime.fromtimestamp(int(x), mc.IST).date().isoformat()
        cache[exp] = {"mtime": mt, "from": d(min(lo)), "to": d(max(hi))}
        changed = True
    if changed:
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        paths.atomic_write_json(INDEX_PATH, cache)
    return {e: [v["from"], v["to"]] for e, v in cache.items()}


def history_days():
    """[{day, expiries}] newest first; expiries = up to the 3 nearest whose file covers that day."""
    S = _sim()
    cov = coverage()
    sp = S._load_spot()
    days = sorted({d.isoformat() for d in pd.to_datetime(sp["ts"], unit="s", utc=True).dt.tz_convert(mc.IST).dt.date.unique()}, reverse=True)
    exps = sorted(cov)
    out = []
    for d in days:
        e = [x for x in exps if x >= d and cov[x][0] <= d <= cov[x][1]][:3]
        if e:
            out.append({"day": d, "expiries": e})
    return out


_cache = {}


def history(day, expiries):
    key = (day, tuple(expiries))
    if key in _cache:
        return _cache[key]
    S = _sim()
    sp = S._load_spot()
    st, sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
    times = []
    t = datetime.strptime(f"{day} 09:20", "%Y-%m-%d %H:%M").replace(tzinfo=mc.IST)
    while t.strftime("%H:%M") <= "15:25":
        times.append(t)
        t += timedelta(minutes=STEP_MIN)
    out = {"day": day, "grid": GRID.tolist(), "times": [x.strftime("%H:%M") for x in times], "expiries": [], "spot": [], "live": False}
    spot_by_t = {}
    for t in times:
        e = np.searchsorted(st, int(t.timestamp()), side="right") - 1
        spot_by_t[t] = float(sc[e]) if e >= 0 and st[e] >= t.timestamp() - 300 else None
    out["spot"] = [spot_by_t[t] for t in times]
    opt_dir = os.path.join(paths.BASE_DIR, "data", "hist1m", "options")
    for exp in expiries:
        if not os.path.exists(os.path.join(opt_dir, f"{exp}.parquet")):
            continue
        df = S._load_options(exp)
        dd = df[df["day"] == date.fromisoformat(day)].sort_values("ts")
        if dd.empty:
            continue
        exp_close = _close_ts(exp)
        rows_t = []
        for t in times:
            ts, spot = int(t.timestamp()), spot_by_t[t]
            T = (exp_close - ts) / (365 * 86400)
            if not spot or T <= 0:
                rows_t.append(None); continue
            last = dd[(dd.ts <= ts) & (dd.ts > ts - 1800)].groupby(["type", "strike"]).close.last()   # prices <= 30 min old
            rows = {}
            for (typ, k), px in last.items():
                rows.setdefault(int(k), {})["c_px" if typ == "CE" else "p_px"] = float(px)
            if not rows:
                rows_t.append(None); continue
            sm = smile(rows, spot, _fwd(rows, spot), T)
            rows_t.append(sm)
        out["expiries"].append({"expiry": exp, "slices": rows_t})
    if len(_cache) > 30:
        _cache.clear()
    _cache[key] = out
    return out


# ---------------------------------------------------------------------------
# live
def _live_slice(expiry_ts):
    ch = chain_mod.get_chain(strikecount=40, expiry_timestamp=expiry_ts)
    spot, exp_ts = ch["spot"], ch.get("resolved_expiry_ts")
    T = (exp_ts - time.time()) / (365 * 86400) if exp_ts else None
    if not spot or not T or T <= 0:
        return None, spot
    rows = {int(r["strike"]): {"c_px": (r.get("ce") or {}).get("ltp"), "p_px": (r.get("pe") or {}).get("ltp")} for r in ch["strikes"]}
    return smile(rows, spot, _fwd(rows, spot), T), spot


def _live_expiries():
    now = time.time()
    return [e for e in chain_mod.list_expiries() if float(e["expiry"]) > now][:2]


def sample_live():
    now = mc.now_ist()
    rec = {"t": now.strftime("%H:%M"), "expiries": {}}
    for e in _live_expiries():
        sm, spot = _live_slice(e["expiry"])
        d, m, y = e["date"].split("-")
        rec["expiries"][f"{y}-{m}-{d}"] = sm
        rec["spot"] = spot
    os.makedirs(LIVE_DIR, exist_ok=True)
    p = os.path.join(LIVE_DIR, f"{now.date().isoformat()}.json")
    try:
        data = json.load(open(p))
    except (OSError, ValueError):
        data = {"samples": []}
    data["samples"] = [x for x in data["samples"] if x["t"] != rec["t"]] + [rec]
    paths.atomic_write_json(p, data)
    return rec


def live():
    today = mc.now_ist().date().isoformat()
    try:
        S_ = json.load(open(os.path.join(LIVE_DIR, f"{today}.json")))["samples"]
    except (OSError, ValueError):
        S_ = []
    if not S_:                                         # app just started: take a first sample now
        try:
            S_ = [sample_live()]
        except Exception:
            pass
    exps = sorted({e for s in S_ for e in s.get("expiries", {})})[:2]
    return {"day": today, "grid": GRID.tolist(), "times": [s["t"] for s in S_], "spot": [s.get("spot") for s in S_], "live": True,
            "expiries": [{"expiry": e, "slices": [s.get("expiries", {}).get(e) for s in S_]} for e in exps]}


def live_now():
    now = mc.now_ist()
    return mc.is_trading_day(now.date()) and now.strftime("%H:%M") >= "09:15"


def rule_for(day):
    """The plan's IV rule on that day: its expiry, the threshold per check time and its own readings
    (sell_rules history for past days, results/iv_rule/today.json + today's thresholds live)."""
    import sell_rules as SR
    today = mc.now_ist().date().isoformat()
    try:
        if day == today:
            thr = SR.thresholds_for_today() or {}
            tj = SR._load_today() or {}
            rd = (tj.get("readings") or {}) if tj.get("day") == today else {}
            out = {"expiry": tj.get("expiry") if tj.get("day") == today else None,
                   "thr": {s: thr.get(s) for s in ("09:30", "12:00", "14:30")},
                   "readings": {s: (v or {}).get("atm_iv") for s, v in rd.items()}}
        else:
            h = SR.history()
            x = h[h["day"].astype(str) == day]
            if x.empty:
                return None
            out = {"expiry": str(x.expiry.iloc[0]), "dte": int(x.dte.iloc[0]),
                   "thr": {r.slot: (None if pd.isna(r.thr) else round(float(r.thr), 2)) for r in x.itertuples()},
                   "readings": {r.slot: (None if pd.isna(r.atm_iv) else round(float(r.atm_iv), 2)) for r in x.itertuples()}}
        out["sell"] = any(v is not None and out["thr"].get(s) is not None and v >= out["thr"][s] for s, v in out["readings"].items())
        return out
    except Exception:
        return None


def payload(day=None, expiries=None):
    out = _payload(day, expiries)
    out["rule"] = rule_for(out["day"])
    return out


def _payload(day=None, expiries=None):
    today = mc.now_ist().date().isoformat()
    if (not day or day == today) and live_now():
        return live()
    days = history_days()
    if not day or day == today:
        last = next((d for d in days if d["day"] < today), None)
        if not last:
            raise ValueError("no history")
        out = history(last["day"], last["expiries"][:2])
        return dict(out, note=f"Market not open — showing the last session ({last['day']}).")
    if not expiries:
        hit = next((d for d in days if d["day"] == day), None)
        expiries = (hit or {}).get("expiries", [])[:2]
    return history(day, expiries)


_thread = None


def _loop():
    while True:
        try:
            if live_now() and mc.now_ist().strftime("%H:%M") <= "15:31":
                sample_live()
        except Exception:
            traceback.print_exc()
        time.sleep(STEP_MIN * 60 - time.time() % (STEP_MIN * 60) + 20)


def start_background():
    global _thread
    if _thread and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, daemon=True, name="surface-sampler")
    _thread.start()
