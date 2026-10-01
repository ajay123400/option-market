"""sell_rules.py -- the premium-selling checklist that held up in the
backtests, as live information (never an order):

  IV level      ATM IV of the current weekly expiry >= the 60th percentile of
                the past 500 entry days' ATM IV at the same time of day
                (09:30 / 12:00 / 14:30 buckets) -> "sell day"
  expiry window 1-4 trading days left (expiry day and next week's expiry
                did worse per day held)
  put skew      25-delta put IV - 25-delta call IV >= the 60th percentile of
                the past 500 days at 09:30 -> put side preferred

History (one row per entry day x time, results/iv_rule/history.parquet) is
computed from the 1-min option history exactly like the template backtest
measured ATM IV, so the live check and the tested rule use the same yardstick.
The 19:00 daily job appends each new day (update_history).
"""
import json
import math
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import greeks as g
import market_calendar as mc
import paths
import simulator as S

DIR = os.path.join(paths.BASE_DIR, "results", "iv_rule")
HIST = os.path.join(DIR, "history.parquet")
TODAY = os.path.join(DIR, "today.json")
SLOTS = ("09:30", "12:00", "14:30")
LOOKBACK, MIN_ROWS, PCT = 500, 120, 0.6
F_O_CLOSE_CHANGE = date(2026, 8, 3)
R = g.RISK_FREE_RATE
# out-of-sample backtest results (iv_intraday_research / entry_timing / iv_surface reports), per 1 lot
BACKTEST = {
    "iv": {"straddle_sell": 4833, "straddle_normal": -11, "strangle_sell": 2905, "strangle_normal": 251,
           "late_sell": 3450, "faded": -259},
    "window": {"this_week_per_day": 309, "next_week_per_day": 155},
    "skew": {"put_steep": 1922, "call_steep": 248, "put_flat": 102, "call_flat": 505},
}


def _dte(day, exp_d):
    n, d = 0, day + timedelta(days=1)
    while d <= exp_d:
        n += mc.is_trading_day(d)
        d += timedelta(days=1)
    return n


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _iv_delta(price, fwd, k, T, call):
    iv = g.implied_vol_fwd(float(price), fwd, k, T, R, call)
    if not iv:
        return None, None
    d1 = (math.log(fwd / k) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    return iv, math.exp(-R * T) * (_ncdf(d1) if call else _ncdf(d1) - 1)


def measure(quotes, spot, T):
    """quotes {(strike, 'CE'|'PE'): price} -> (fwd, ATM IV %, 25-delta skew
    in vol points). The same steps as template_backtest: forward from the 6
    strikes nearest NIFTY with both sides, ATM = forward rounded to 50, IV =
    average of the ATM call and put."""
    ks = sorted({k for k, _ in quotes})
    near = sorted([(k, quotes[(k, "CE")], quotes[(k, "PE")]) for k in ks if (k, "CE") in quotes and (k, "PE") in quotes],
                  key=lambda x: abs(x[0] - spot))[:6]
    fwd = g.synthetic_forward(near) if near else None
    if not fwd or T <= 0:
        return None, None, None
    atm = int(round(fwd / 50) * 50)
    ivs = [_iv_delta(quotes[(atm, t)], fwd, atm, T, t == "CE")[0] for t in ("CE", "PE") if (atm, t) in quotes]
    ivs = [v for v in ivs if v]
    atm_iv = round(sum(ivs) / len(ivs) * 100, 2) if ivs else None
    best = {}
    for (k, t), p in quotes.items():
        call = t == "CE"
        if (call and k <= fwd) or (not call and k >= fwd) or p <= 0.5 or abs(k - fwd) > 2000:
            continue
        iv, dl = _iv_delta(p, fwd, k, T, call)
        if iv is None:
            continue
        gap = abs(abs(dl) - 0.25)
        if gap < 0.07 and (t not in best or gap < best[t][0]):
            best[t] = (gap, iv)
    skew = round((best["PE"][1] - best["CE"][1]) * 100, 2) if "PE" in best and "CE" in best else None
    return fwd, atm_iv, skew


def _exp_close_ts(exp):
    d = pd.Timestamp(exp).date()
    return int(pd.Timestamp(f"{exp} {'15:40' if d >= F_O_CLOSE_CHANGE else '15:30'}", tz=S.IST).timestamp())


def compute_expiry(exp, only_after=None):
    """History rows for every day 1-4 trading days before `exp`."""
    df = S._load_options(exp)
    mins = df["date"].dt.hour * 60 + df["date"].dt.minute
    df = df[mins <= 15 * 60 + 40]
    exp_d = pd.Timestamp(exp).date()
    sp = S._load_spot()
    st, sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
    ect = _exp_close_ts(exp)
    rows = []
    for D in sorted(df["day"].unique()):
        if D >= exp_d or (only_after and D.isoformat() <= only_after):
            continue
        dte = _dte(D, exp_d)
        if not 1 <= dte <= 4:
            continue
        dd = df[df["day"] == D].sort_values("ts")
        for slot in SLOTS:
            t = int(pd.Timestamp(f"{D} {slot}", tz=S.IST).timestamp())
            upto = dd[dd.ts <= t]
            if upto.empty or t - upto.ts.max() > 300:
                continue
            last = upto.groupby(["strike", "type"]).close.last()
            j = np.searchsorted(st, t, side="right") - 1
            if j < 0:
                continue
            spot = float(sc[j])
            q = {(int(k), ty): float(v) for (k, ty), v in last.items() if abs(k - spot) <= 1600}
            _, iv, skew = measure(q, spot, (ect - t) / (365 * 86400))
            if iv:
                rows.append({"day": D.isoformat(), "expiry": exp, "dte": dte, "slot": slot, "atm_iv": iv,
                             "skew25": skew if slot == "09:30" else None})
    return rows


def build_history(years=5.5, progress=print):
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live"))
    cutoff = (date.today() - timedelta(days=int(365.25 * years))).isoformat()
    rows = []
    for i, exp in enumerate(e for e in exps if e >= cutoff):
        rows += compute_expiry(exp)
        if progress and i % 20 == 0:
            progress(f"iv history: {exp} ({len(rows)} rows)")
    h = pd.DataFrame(rows).sort_values(["day", "slot"]).reset_index(drop=True)
    os.makedirs(DIR, exist_ok=True)
    h.to_parquet(HIST, index=False)
    return h


def update_history(progress=print):
    """Append the days after the last one stored (called by the 19:00 job)."""
    if not os.path.exists(HIST):
        return len(build_history(progress=progress))
    h = pd.read_parquet(HIST)
    last = h.day.max()
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live") and e > last)
    rows = []
    for exp in exps:
        rows += compute_expiry(exp, only_after=last)
    if rows:
        h = pd.concat([h, pd.DataFrame(rows)]).drop_duplicates(["day", "slot"], keep="last").sort_values(["day", "slot"])
        h.to_parquet(HIST, index=False)
    _cache.clear()
    return len(rows)


_cache = {}


def history():
    """History with the rolling thresholds (past rows only) per time bucket."""
    try:
        m = os.path.getmtime(HIST)
    except OSError:
        return None
    if _cache.get("m") == m:
        return _cache["h"]
    h = pd.read_parquet(HIST).sort_values(["slot", "day"])
    h["thr"] = h.groupby("slot").atm_iv.transform(lambda s: s.shift(1).rolling(LOOKBACK, min_periods=MIN_ROWS).quantile(PCT))
    sk = h[h.slot == "09:30"].skew25
    h.loc[sk.index, "skew_thr"] = sk.shift(1).rolling(LOOKBACK, min_periods=MIN_ROWS).quantile(PCT)
    h = h.sort_values(["day", "slot"]).reset_index(drop=True)
    _cache.update(m=m, h=h)
    return h


def thresholds_for_today():
    """Thresholds from all stored days before today."""
    h = history()
    if h is None:
        return None
    today = mc.now_ist().date().isoformat()
    out = {}
    for s in SLOTS:
        v = h[(h.slot == s) & (h.day < today)].atm_iv.tail(LOOKBACK)
        out[s] = round(float(v.quantile(PCT)), 2) if len(v) >= MIN_ROWS else None
    v = h[(h.slot == "09:30") & (h.day < today)].skew25.dropna().tail(LOOKBACK)
    out["skew"] = round(float(v.quantile(PCT)), 2) if len(v) >= MIN_ROWS else None
    return out


def last_day():
    """The latest stored day: IV vs threshold at each time + skew, for when
    the market is closed (after hours the chain's prices give a misleading IV)."""
    h = history()
    if h is None or h.empty:
        return None
    d = h.day.max()
    x = h[h.day == d]
    slots = {r.slot: {"iv": r.atm_iv, "thr": None if pd.isna(r.thr) else round(float(r.thr), 2),
                      "sell": bool(pd.notna(r.thr) and r.atm_iv >= r.thr)} for r in x.itertuples()}
    s930 = x[x.slot == "09:30"]
    sk = s930.skew25.iloc[0] if len(s930) else None
    skt = s930.skew_thr.iloc[0] if len(s930) else None
    return {"day": d, "expiry": x.expiry.iloc[0], "dte": int(x.dte.iloc[0]), "slots": slots,
            "skew25": None if sk is None or pd.isna(sk) else float(sk), "skew_thr": None if skt is None or pd.isna(skt) else round(float(skt), 2),
            "skew_steep": bool(sk is not None and skt is not None and pd.notna(sk) and pd.notna(skt) and sk >= skt)}


def _bucket(now):
    hm = now.strftime("%H:%M")
    if hm < "09:30":
        return None
    return "09:30" if hm < "12:00" else "12:00" if hm < "14:30" else "14:30"


def _load_today():
    try:
        with open(TODAY) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def status(chain):
    """Live checklist for the chain being shown. Uses only the chain's own
    prices (no extra Fyers calls)."""
    now = mc.now_ist()
    thr = thresholds_for_today() or {}
    exp_ts = chain.get("resolved_expiry_ts")
    spot = chain.get("spot")
    out = {"time": now.strftime("%H:%M"), "thresholds": thr, "backtest": BACKTEST, "market": mc.market_state().get("state")}
    try:
        out["last_day"] = last_day()
    except Exception:
        out["last_day"] = None
    if not exp_ts or not spot:
        return out
    exp_d = datetime.fromtimestamp(float(exp_ts), tz=now.tzinfo).date()
    dte = _dte(now.date(), exp_d)
    out["expiry"] = exp_d.isoformat()
    out["dte"] = dte
    out["window"] = "ok" if 1 <= dte <= 4 else "expiry_day" if dte == 0 else "later"
    q = {}
    for r in chain.get("strikes") or []:
        for side, t in (("ce", "CE"), ("pe", "PE")):
            p = (r.get(side) or {}).get("ltp")
            if p:
                q[(int(r["strike"]), t)] = float(p)
    T = (float(exp_ts) - now.timestamp()) / (365 * 86400)
    _, iv, skew = measure(q, float(spot), T)
    bucket = _bucket(now) if out["market"] == "OPEN" else None
    live = bucket is not None
    out.update(atm_iv=iv if live else None, skew25=skew if live else None, bucket=bucket, live=live)
    t_now = thr.get(bucket) if bucket else None
    out["iv_thr_now"] = t_now
    out["iv_ok"] = bool(iv is not None and t_now is not None and iv >= t_now)
    out["sell_day"] = bool(out["iv_ok"] and out["window"] == "ok")
    out["skew_steep"] = bool(skew is not None and thr.get("skew") is not None and skew >= thr["skew"])
    # today's readings at the tested times: the first live reading in each bucket
    # (while the app runs); the 19:00 job stores the exact values from history
    today = now.date().isoformat()
    tj = _load_today()
    if tj.get("day") != today or tj.get("expiry") != out["expiry"]:
        tj = {"day": today, "expiry": out["expiry"], "readings": {}}
    if bucket and iv is not None and bucket not in tj["readings"] and out["window"] == "ok":
        slot_t = datetime.combine(now.date(), datetime.strptime(bucket, "%H:%M").time(), tzinfo=now.tzinfo)
        if (now - slot_t).total_seconds() <= 15 * 60:
            tj["readings"][bucket] = {"atm_iv": iv, "skew25": skew if bucket == "09:30" else None, "at": now.strftime("%H:%M")}
            os.makedirs(DIR, exist_ok=True)
            paths.atomic_write_json(TODAY, tj)
    out["readings"] = tj["readings"] if tj.get("expiry") == out["expiry"] else {}
    return out


def history_payload():
    h = history()
    if h is None:
        return None
    num = lambda v, n=2: None if v is None or pd.isna(v) else round(float(v), n)
    series = {s: [{"day": r.day, "iv": num(r.atm_iv), "thr": num(r.thr), "dte": int(r.dte)}
                  for r in h[h.slot == s].itertuples()] for s in SLOTS}
    sk = [{"day": r.day, "skew": num(r.skew25), "thr": num(r.skew_thr)}
          for r in h[h.slot == "09:30"].itertuples() if pd.notna(r.skew25)]
    return {"series": series, "skew": sk, "backtest": BACKTEST, "thresholds_today": thresholds_for_today()}

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    h = build_history()
    print(len(h), "rows", h.day.min(), "->", h.day.max())
    print(thresholds_for_today())
