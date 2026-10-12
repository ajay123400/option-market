"""oi_flow_research.py -- does the 09:30 option-chain flow tell NIFTY's direction?

User's idea (2026-10-12): at 09:25-09:30, on the top 3 strikes of each side,
  put change-of-OI < call change-of-OI  AND  put volume > call volume  -> bearish
  call change-of-OI < put change-of-OI  AND  call volume > put volume -> bullish

Per trading day, current weekly expiry, reading after the 09:29 bar closes:
  change of OI = OI now - OI at the previous session's last bar (as the chain shows it)
  volume       = 09:15-09:29 volume
"Top 3" tried three ways: by OI now, by volume, and the 3 OTM strikes nearest ATM.
Outcome: NIFTY spot from 09:30 to 10:30 / 12:00 / 15:25.

Usage:  python oi_flow_research.py
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

import paths

H = os.path.join(paths.BASE_DIR, "data", "hist1m")
OUT = os.path.join(paths.BASE_DIR, "results", "oi_flow")
IST = "Asia/Kolkata"
HORIZONS = {"10:30": 10 * 60 + 30, "12:00": 12 * 60, "close": 15 * 60 + 25}


def _spot():
    s = pd.read_parquet(os.path.join(H, "NIFTY50_1m.parquet"))
    t = pd.to_datetime(s.ts, unit="s", utc=True).dt.tz_convert(IST)
    s["d"], s["m"] = t.dt.date, t.dt.hour * 60 + t.dt.minute
    return s


def _price_at(day_spot, m):
    """close of the last bar starting before minute m (i.e. price at m)."""
    x = day_spot[day_spot.m < m]
    return float(x.close.iloc[-1]) if len(x) else np.nan


def _top3(side, how, spot):
    if how == "oi":
        return side.nlargest(3, "oi_now")
    if how == "vol":
        return side.nlargest(3, "vol")
    otm = side[side.strike >= spot] if side.type.iloc[0] == "CE" else side[side.strike <= spot]
    return otm.assign(dist=(otm.strike - spot).abs()).nsmallest(3, "dist")


def run():
    spot = _spot()
    days = sorted(spot.d.unique())
    exps = sorted(date.fromisoformat(f[:10]) for f in os.listdir(os.path.join(H, "options")) if f.endswith(".parquet"))
    rows = []
    cache = {}
    for i, d in enumerate(days[1:], 1):
        exp = next((e for e in exps if e >= d), None)
        if exp is None:
            continue
        if exp not in cache:
            cache.clear()
            o = pd.read_parquet(os.path.join(H, "options", f"{exp}.parquet"))
            t = pd.to_datetime(o.ts, unit="s", utc=True).dt.tz_convert(IST)
            o["d"], o["m"] = t.dt.date, t.dt.hour * 60 + t.dt.minute
            cache[exp] = o
        o = cache[exp]
        prev = days[i - 1]
        ds = spot[spot.d == d]
        s930 = _price_at(ds, 9 * 60 + 30)
        if np.isnan(s930):
            continue
        p = o[o.d == prev]
        if p.empty:
            continue
        oi_prev = p.sort_values("ts").groupby("symbol").oi.last()
        today = o[(o.d == d) & (o.m >= 9 * 60 + 15) & (o.m <= 9 * 60 + 29)]
        if today.empty:
            continue
        g = today.sort_values("ts").groupby("symbol")
        snap = pd.DataFrame({"type": g.type.first(), "strike": g.strike.first(), "oi_now": g.oi.last(), "vol": g.volume.sum()})
        snap["oi_prev"] = oi_prev.reindex(snap.index)
        snap = snap.dropna(subset=["oi_prev"])
        snap["doi"] = snap.oi_now - snap.oi_prev
        ce, pe = snap[snap.type == "CE"], snap[snap.type == "PE"]
        if len(ce) < 3 or len(pe) < 3:
            continue
        rec = {"day": d, "expiry": exp, "dte": (exp - d).days, "spot930": s930}
        for k, m in HORIZONS.items():
            rec["ret_" + k] = (_price_at(ds, m + 1) / s930 - 1) * 100
        for how in ("oi", "vol", "atm"):
            c, q = _top3(ce, how, s930), _top3(pe, how, s930)
            if len(c) < 3 or len(q) < 3:
                rec["sig_" + how] = np.nan
                continue
            c_doi, p_doi, c_v, p_v = c.doi.sum(), q.doi.sum(), c.vol.sum(), q.vol.sum()
            sig = -1 if (p_doi < c_doi and p_v > c_v) else 1 if (c_doi < p_doi and c_v > p_v) else 0
            rec["sig_" + how] = sig
        rows.append(rec)
    return pd.DataFrame(rows)


def _ttest(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def summarize(df):
    out = []
    mid = df.day.iloc[len(df) // 2]
    for how in ("oi", "vol", "atm"):
        for h in HORIZONS:
            r = df["ret_" + h]
            for part, sel in (("all", df.day.notna()), ("1st half", df.day < mid), ("2nd half", df.day >= mid)):
                x = df[sel]
                base_up = (x["ret_" + h] > 0).mean()
                for sig, name in ((1, "bullish"), (-1, "bearish")):
                    y = x[x["sig_" + how] == sig]["ret_" + h].dropna()
                    signed = y * sig
                    out.append({"top3": how, "horizon": h, "part": part, "signal": name, "days": len(y),
                                "hit%": round((signed > 0).mean() * 100, 1) if len(y) else np.nan,
                                "base_up%": round(base_up * 100, 1),
                                "avg_move%": round(y.mean(), 3) if len(y) else np.nan,
                                "t": round(_ttest(signed), 2)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    df = run()
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "days.csv"), index=False)
    s = summarize(df)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"{len(df)} days {df.day.min()} -> {df.day.max()}")
    for how in ("oi", "vol", "atm"):
        print(how, df["sig_" + how].value_counts().to_dict())
    print(s.to_string(index=False))
