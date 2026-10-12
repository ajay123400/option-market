"""futures_history.py -- NIFTY monthly futures, 1-min OHLC + volume + OI,
one file per contract in data/hist1m/futures/NIFTY_FUT_<expiry>.parquet
(columns: ts, open, high, low, close, volume, oi; ts = bar start, epoch s).

  backfill()   expired contracts from Fyers' expired-F&O API, once
               (the old Fyers plan was extended to 16 Oct 2026; after that the
               app takes no data from Fyers -- see arrow-migration)
  update()     the listed contracts (near / next / far month) from Arrow's
               candle API, re-fetched in full each evening (called by
               daily_history.update)

Usage:  python futures_history.py backfill [YYYY-MM-DD start]
        python futures_history.py update
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import paths

DIR = os.path.join(paths.BASE_DIR, "data", "hist1m", "futures")
COLS = ["ts", "open", "high", "low", "close", "volume", "oi"]
LIFE_DAYS = 100          # a monthly contract trades ~3 months
F_O_CLOSE_CHANGE = date(2026, 8, 3)   # NSE F&O closes at 15:40 from this day (15:30 before)


def path(expiry_iso):
    return os.path.join(DIR, f"NIFTY_FUT_{expiry_iso}.parquet")


def _save(expiry_iso, df):
    df = df[COLS].dropna(subset=["close"]).drop_duplicates("ts", keep="last").sort_values("ts")
    df["ts"] = df["ts"].astype("int64")
    for c in ("volume", "oi"):
        df[c] = df[c].fillna(0).astype("int64")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float32")
    os.makedirs(DIR, exist_ok=True)
    tmp = path(expiry_iso) + ".tmp"
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path(expiry_iso))
    return len(df)


def monthly_expiries(start, end):
    """[[expiries of the month, latest first], ...] -- the monthly contract usually expires on the month's last
    listed expiry, but a holiday can move it (Mar 2023: 29th, Jun 2023: 28th, Apr 2025: 24th while a later date
    was listed), so callers try each date from the latest down."""
    import history_downloader as hd
    exps = sorted(date.fromisoformat(e) for e in hd.list_expiries(start, end))
    by = {}
    for e in exps:
        by.setdefault((e.year, e.month), []).append(e)
    return [sorted(v, reverse=True) for _, v in sorted(by.items())]


def backfill(start=date(2020, 1, 1), end=None, log=print):
    """Expired monthly futures from Fyers (skips contracts already on disk)."""
    import history_downloader as hd
    end = end or date.today() - timedelta(days=1)
    done = skipped = 0
    for month in monthly_expiries(start, end):
        if any(os.path.exists(path(e.isoformat())) for e in month) or month[0] >= date.today():
            skipped += 1
            continue
        exp, syms = None, []
        for e in month:
            u = hd._get(f"{hd.EXP_API}/underlying-symbols", {"symbol": hd.INDEX, "expiry_date": e.isoformat()})
            syms = (((u.get("data") or {}).get("contracts") or {}).get("futures") or [])
            if syms:
                exp = e
                break
        if not syms:
            log(f"futures {month[0]:%Y-%m}: no futures contract listed on any expiry of the month")
            continue
        h = hd._get(f"{hd.EXP_API}/historical-data", {"symbol": syms[0], "resolution": "1", "date_format": 1,
                                                      "range_from": (exp - timedelta(days=LIFE_DAYS)).isoformat(),
                                                      "range_to": exp.isoformat(), "include_oi": 1})
        c = h.get("candles") or []
        if not c:
            log(f"futures {exp} {syms[0]}: no candles ({h.get('s')} {h.get('message')})")
            continue
        cols = h.get("columns") or ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
        df = pd.DataFrame(c, columns=cols[:len(c[0])]).rename(columns={"timestamp": "ts", "open_interest": "oi"})
        if "oi" not in df:
            df["oi"] = 0
        n = _save(exp.isoformat(), df)
        log(f"futures {exp} {syms[0]}: {n} bars")
        done += 1
    return done, skipped


def update(log=print):
    """Listed contracts from Arrow, full life, chunked (1-min requests are capped at ~1 month)."""
    import arrow_chain
    import arrow_history
    today = date.today()
    n = 0
    for f in arrow_chain.instruments()["futs"]:
        exp = f["date"]
        rows = arrow_history.candles_1m(f["fyers"], exp - timedelta(days=LIFE_DAYS), min(today, exp))
        if not rows:
            log(f"futures {exp} (arrow): no candles")
            continue
        df = pd.DataFrame(rows, columns=COLS)
        old = pd.read_parquet(path(exp.isoformat())) if os.path.exists(path(exp.isoformat())) else None
        if old is not None:
            df = pd.concat([old, df], ignore_index=True)
        log(f"futures {exp} (arrow): {_save(exp.isoformat(), df)} bars")
        n += 1
    return n


def contracts():
    """[{expiry, bars, first, last}] for every contract on disk (newest first)."""
    out = []
    if not os.path.isdir(DIR):
        return out
    import pyarrow.parquet as pq
    for f in sorted(os.listdir(DIR), reverse=True):
        if f.startswith("NIFTY_FUT_") and f.endswith(".parquet"):
            md = pq.ParquetFile(os.path.join(DIR, f)).metadata
            out.append({"expiry": f[10:-8], "bars": md.num_rows})
    return out


def _load(expiry_iso):
    df = pd.read_parquet(path(expiry_iso))
    df["contract"] = expiry_iso
    return df


def near_month(start, end):
    """Continuous near-month series: each session taken from the nearest contract not yet expired that day."""
    exps = sorted(c["expiry"] for c in contracts())
    frames = []
    prev = None
    for e in exps:
        lo, hi = (prev or "0000"), e
        prev = e
        if hi < start.isoformat() or lo >= end.isoformat():
            continue
        df = _load(e)
        d = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.date.astype(str)
        frames.append(df[(d > lo) & (d <= hi)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS + ["contract"])


def bars(contract, tf=5, start=None, end=None):
    """{'bars': [{t, o, h, l, c, v, oi, contract}], ...} for a contract (expiry ISO) or 'near', resampled to tf
    minutes (1-5) from the 09:15 open; start / end are dates (inclusive)."""
    tf = max(1, min(5, int(tf)))
    if contract == "near":
        end = end or date.today()
        start = start or end - timedelta(days=14)
        df = near_month(start, end)
    else:
        df = _load(contract)
    if df.empty:
        return {"bars": []}
    t = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    hm = t.dt.hour * 60 + t.dt.minute
    # regular session: 09:15-15:30 until 31 Jul 2026, 09:15-15:40 from 3 Aug 2026 (NSE F&O close moved to 15:40)
    close_m = np.where(t.dt.date >= F_O_CLOSE_CHANGE, 15 * 60 + 39, 15 * 60 + 29)
    keep = (hm >= 9 * 60 + 15) & (hm <= close_m)
    df, t = df[keep], t[keep]
    day = t.dt.date
    if start:
        df, t, day = df[day >= start], t[day >= start], day[day >= start]
    if end:
        df, t, day = df[day <= end], t[day <= end], day[day <= end]
    if df.empty:
        return {"bars": []}
    mins = (t.dt.hour * 60 + t.dt.minute - (9 * 60 + 15)) // tf * tf + 9 * 60 + 15
    key = t.dt.normalize() + pd.to_timedelta(mins, unit="min")
    g = df.assign(k=key).groupby("k", sort=True)
    out = pd.DataFrame({"o": g.open.first(), "h": g.high.max(), "l": g.low.min(), "c": g.close.last(),
                        "v": g.volume.sum(), "oi": g.oi.last(), "contract": g.contract.last()})
    # epoch seconds of the IST wall clock (the chart shows times as-is)
    out["t"] = (out.index.tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    recs = out.reset_index(drop=True)
    return {"bars": [{"t": int(r.t), "o": round(float(r.o), 2), "h": round(float(r.h), 2), "l": round(float(r.l), 2), "c": round(float(r.c), 2),
                      "v": int(r.v), "oi": int(r.oi), "contract": r.contract} for r in recs.itertuples()]}


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    if cmd == "backfill":
        st = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2020, 1, 1)
        import fyers_auth
        fyers_auth.login()
        print(backfill(st))
    else:
        print(update())
