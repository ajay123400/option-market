"""price_action_research.py -- does higher-timeframe price action call NIFTY's
direction over the next 1 / 3 / 5 sessions? (for debit spreads, 2026-10-12)

Signals, read at each day's close (15:30) from NIFTY spot 1-min bars resampled
to 30-min and 2-hour candles (from 09:15):
  structure  swing points = fractals (high/low beyond 2 candles each side,
             confirmed only after those 2 candles close);
             up = last swing high > previous AND last swing low > previous (HH+HL),
             down = LH+LL, else none
  bos        last close above the latest confirmed swing high -> up, below the
             latest swing low -> down; the state holds until the other side breaks
  dma20      daily close above / below its 20-day average (the baseline)
Outcome: spot close-to-close over the next 1 / 3 / 5 sessions.

Usage:  python price_action_research.py
"""
import os
import sys

import numpy as np
import pandas as pd

import paths

H = os.path.join(paths.BASE_DIR, "data", "hist1m")
OUT = os.path.join(paths.BASE_DIR, "results", "price_action")
IST = "Asia/Kolkata"
AHEAD = (1, 3, 5)


def _spot():
    s = pd.read_parquet(os.path.join(H, "NIFTY50_1m.parquet"))
    t = pd.to_datetime(s.ts, unit="s", utc=True).dt.tz_convert(IST)
    m = t.dt.hour * 60 + t.dt.minute
    s = s[(m >= 9 * 60 + 15) & (m <= 15 * 60 + 29)].copy()
    s["t"] = t[s.index]
    return s


def candles(s, mins):
    t = s.t
    k = (t.dt.hour * 60 + t.dt.minute - (9 * 60 + 15)) // mins * mins + 9 * 60 + 15
    key = t.dt.normalize() + pd.to_timedelta(k, unit="min")
    g = s.assign(k=key).groupby("k")
    c = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    c["d"] = c.index.date
    return c


def states(c, n=2):
    """structure and bos state after each candle closes (no look-ahead: a fractal at i is known at i+n)."""
    hi, lo, cl = c.high.values, c.low.values, c.close.values
    N = len(c)
    sh, sl = [], []                        # confirmed swing highs / lows (values)
    struct, bos = np.zeros(N, int), np.zeros(N, int)
    b = 0
    for j in range(N):
        i = j - n                          # candle i is confirmed once j = i + n has closed
        if i - n >= 0:
            if hi[i] > hi[i - n:i].max() and hi[i] > hi[i + 1:j + 1].max():
                sh.append(hi[i])
            if lo[i] < lo[i - n:i].min() and lo[i] < lo[i + 1:j + 1].min():
                sl.append(lo[i])
        if len(sh) >= 2 and len(sl) >= 2:
            if sh[-1] > sh[-2] and sl[-1] > sl[-2]:
                struct[j] = 1
            elif sh[-1] < sh[-2] and sl[-1] < sl[-2]:
                struct[j] = -1
        if sh and cl[j] > sh[-1]:
            b = 1
        elif sl and cl[j] < sl[-1]:
            b = -1
        bos[j] = b
    return pd.DataFrame({"structure": struct, "bos": bos, "d": c.d.values}, index=c.index)


def run():
    s = _spot()
    daily = s.groupby(s.t.dt.date).close.last()
    df = pd.DataFrame({"close": daily})
    df["dma20"] = np.sign(df.close - df.close.rolling(20).mean()).fillna(0).astype(int)
    for k in AHEAD:
        df[f"ret{k}"] = (df.close.shift(-k) / df.close - 1) * 100
    for name, mins in (("30m", 30), ("2h", 120)):
        st = states(candles(s, mins)).groupby("d").last()
        df[f"structure_{name}"] = st.structure.reindex(df.index).fillna(0).astype(int)
        df[f"bos_{name}"] = st.bos.reindex(df.index).fillna(0).astype(int)
    return df.iloc[20:]


def summarize(df):
    sigs = ["dma20", "structure_30m", "bos_30m", "structure_2h", "bos_2h"]
    mid = df.index[len(df) // 2]
    out = []
    for sg in sigs:
        for k in AHEAD:
            for part, sel in (("all", df.index == df.index), ("1st half", df.index < mid), ("2nd half", df.index >= mid)):
                x = df[sel].dropna(subset=[f"ret{k}"])
                for v, name in ((1, "up"), (-1, "down")):
                    y = x[x[sg] == v][f"ret{k}"] * v
                    t = y.mean() / (y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 2 else np.nan
                    out.append({"signal": sg, "ahead": k, "part": part, "call": name, "days": len(y),
                                "hit%": round((y > 0).mean() * 100, 1), "base_up%": round((x[f"ret{k}"] > 0).mean() * 100, 1),
                                "avg_move%": round(y.mean(), 3), "t": round(t, 2)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    df = run()
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "days.csv"))
    s = summarize(df)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(f"{len(df)} days {df.index.min()} -> {df.index.max()}")
    for c in [c for c in df.columns if c.startswith(("dma", "structure", "bos"))]:
        flips = (df[c].diff() != 0).sum()
        print(c, df[c].value_counts().to_dict(), "state changes", flips)
    print(s[s.part == "all"].to_string(index=False))
    print(s[s.part != "all"].pivot_table(index=["signal", "ahead", "call"], columns="part", values=["avg_move%", "hit%"]).round(2).to_string())
