"""supertrend_research.py -- does Supertrend call NIFTY's direction over the next
1 / 3 / 5 sessions? (2026-10-12, same frame as price_action_research.py)

Supertrend (ATR = Wilder's RMA of true range): band = (high+low)/2 +/- mult x ATR,
trailing; direction flips when the close crosses the active band. Read at each
day's close on daily, 2-hour and 30-min candles (from 09:15), settings (10,3), (7,3).
Outcome: spot close-to-close over the next 1 / 3 / 5 sessions; both halves.

Usage:  python supertrend_research.py
"""
import os
import sys

import numpy as np
import pandas as pd

import price_action_research as PA

OUT = os.path.join(PA.paths.BASE_DIR, "results", "supertrend")
SETTINGS = ((10, 3.0), (7, 3.0))


def supertrend(c, n, mult):
    h, l, cl = c.high.to_numpy(), c.low.to_numpy(), c.close.to_numpy()
    pc = np.r_[cl[0], cl[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    mid = (h + l) / 2
    ub, lb = mid + mult * atr, mid - mult * atr
    fu, fl = ub.copy(), lb.copy()
    d = np.ones(len(c), int)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cl[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or cl[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if cl[i] < fl[i] else 1
        else:
            d[i] = 1 if cl[i] > fu[i] else -1
    d[:n] = 0
    return pd.Series(d, index=c.index)


def run():
    s = PA._spot()
    daily = s.groupby(s.t.dt.date).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    df = pd.DataFrame({"close": daily.close})
    for k in PA.AHEAD:
        df[f"ret{k}"] = (df.close.shift(-k) / df.close - 1) * 100
    for n, m in SETTINGS:
        df[f"st{n}_daily"] = supertrend(daily, n, m)
        for name, mins in (("2h", 120), ("30m", 30)):
            c = PA.candles(s, mins)
            st = supertrend(c, n, m)
            df[f"st{n}_{name}"] = st.groupby(c.d.values).last().reindex(df.index).fillna(0).astype(int)
    return df.iloc[30:]


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    df = run()
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "days.csv"))
    sigs = [c for c in df.columns if c.startswith("st")]
    for c in sigs:
        print(c, "state changes", int((df[c].diff() != 0).sum()), df[c].value_counts().to_dict())
    mid = df.index[len(df) // 2]
    rows = []
    for sg in sigs:
        for k in PA.AHEAD:
            x = df.dropna(subset=[f"ret{k}"])
            for v, nm in ((1, "up"), (-1, "down")):
                y = x[x[sg] == v][f"ret{k}"] * v
                y1, y2 = x[(x[sg] == v) & (x.index < mid)][f"ret{k}"] * v, x[(x[sg] == v) & (x.index >= mid)][f"ret{k}"] * v
                rows.append({"signal": sg, "ahead": k, "call": nm, "days": len(y), "hit%": (y > 0).mean() * 100,
                             "avg%": y.mean(), "t": y.mean() / (y.std(ddof=1) / np.sqrt(len(y))),
                             "hit_1st": (y1 > 0).mean() * 100, "hit_2nd": (y2 > 0).mean() * 100, "avg_1st": y1.mean(), "avg_2nd": y2.mean()})
    r = pd.DataFrame(rows).round(2)
    r.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    base = {k: round((df[f"ret{k}"] > 0).mean() * 100, 1) for k in PA.AHEAD}
    print("base up% by horizon", base, "| halves split at", mid)
    pd.set_option("display.width", 220, "display.max_rows", 100)
    print(r.to_string(index=False))
