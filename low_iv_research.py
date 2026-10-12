"""low_iv_research.py -- what (if anything) works on LOW-IV days? (2026-10-12)

The plan only sells when ATM IV is high (A / B). For the other days, three
textbook low-IV trades, entered at 09:30 from the IV-rule history, by the
day's IV percentile (09:30 ATM IV vs the previous 500 days' 09:30 readings,
causal) -- buckets <20, 20-40, 40-60 and >=60 (the sell days, for reference):

  long straddle ATM / long strangle 16 delta
      current weekly expiry (1-4 days left)  and  next week's expiry
      exits: same day 15:15 | 2 sessions later 15:15 | hold to expiry |
             +50% target / -50% stop (else expiry)
  double calendar  sell this week's ATM CE+PE, buy next week's same strike,
      entered 1-4 days before the near expiry, closed at 15:15 on the near
      expiry day (both legs at market)
  short straddle on the same days -- the mirror, to see if low-IV selling loses

1 lot, charges (charges.py) and 0.5 pt slippage per leg per side.
Output: results/low_iv/trades.parquet + summary printed.

Usage:  python low_iv_research.py
"""
import os
import sys

import numpy as np
import pandas as pd

import ashish_rules_bt as AR
import charges
import paths
import sell_rules as SR

OUT = os.path.join(paths.BASE_DIR, "results", "low_iv")
LOT, SLIP, YR = AR.LOT, AR.SLIP, AR.YR
BUCKETS = [(0, 20, "<20"), (20, 40, "20-40"), (40, 60, "40-60"), (60, 101, ">=60")]


def iv_pct():
    h = SR.history()
    h = h[h.slot == "09:30"].sort_values("day").reset_index(drop=True)
    v = h.atm_iv.to_numpy()
    pct = np.full(len(h), np.nan)
    for i in range(len(h)):
        past = v[max(0, i - 500):i]
        if len(past) >= 200:
            pct[i] = (past < v[i]).mean() * 100
    h["pct"] = pct
    return h.dropna(subset=["pct"])


def _bucket(p):
    return next(n for lo, hi, n in BUCKETS if lo <= p < hi)


def _ch(side, prices):
    return sum(charges.order_charges(side, p, LOT) for p in prices)


class Books:
    def __init__(self):
        self.c = {}

    def get(self, exp):
        if exp not in self.c:
            if len(self.c) >= 3:
                self.c.pop(min(self.c))
            self.c[exp] = AR.Book(exp)
        return self.c[exp]


def _exit_index(book, day_iso, hm):
    return book.idx(AR._ts(day_iso, hm))


def long_trade(book, legs, i0, sessions, settle_spot):
    """long legs bought at i0; returns {variant: rs}."""
    keys = [(k, t) for t, k in legs]
    ent = [book.px[x][i0] for x in keys]
    paid = sum(ent) + SLIP * len(keys)
    path = sum(book.px[x] for x in keys)
    ch_in = _ch("BUY", ent)
    out = {}

    def close_at(j, settle=False):
        if settle:
            spot = settle_spot
            val = sum(max(0.0, spot - k) if t == "CE" else max(0.0, k - spot) for t, k in legs)
            ch = _ch("SELL", [max(0.0, spot - k) if t == "CE" else max(0.0, k - spot) for t, k in legs if
                              (spot > k if t == "CE" else spot < k)])
            return (val - paid) * LOT - ch_in - ch
        px = [book.px[x][j] for x in keys]
        return (sum(px) - SLIP * len(keys) - paid) * LOT - ch_in - _ch("SELL", px)

    exp_i = len(book.ts) - 1
    j_same = _exit_index(book, sessions[0], "15:15")
    out["same_day"] = close_at(j_same) if j_same > i0 else np.nan
    if len(sessions) > 2 and sessions[2] <= book.exp:
        out["2_days"] = close_at(_exit_index(book, sessions[2], "15:15"))
    else:
        out["2_days"] = close_at(exp_i, settle=True)
    out["expiry"] = close_at(exp_i, settle=True)
    seg = path[i0 + 1:]
    hit = np.flatnonzero((seg >= paid * 1.5) | (seg <= paid * 0.5))
    out["tgt50_stop50"] = close_at(i0 + 1 + hit[0]) if len(hit) else out["expiry"]
    return out


def run():
    h = iv_pct()
    man = AR.S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live"))
    spot = AR.Spot()
    days = sorted(spot.daily.index)
    books = Books()
    rows = []
    for n, r in enumerate(h.itertuples()):
        if not (1 <= r.dte <= 4) or r.expiry not in exps:
            continue
        nxt = [x for x in exps if x > r.expiry]
        nxt = nxt[0] if nxt and (pd.Timestamp(nxt[0]) - pd.Timestamp(r.day)).days <= 12 else None
        t0 = AR._ts(r.day, "09:30")
        sp = spot.at(t0)
        k = days.index(r.day) if r.day in days else None
        if k is None:
            continue
        sess = days[k:k + 3]
        base = {"day": r.day, "expiry": r.expiry, "dte": int(r.dte), "iv": r.atm_iv, "pct": r.pct, "bucket": _bucket(r.pct)}
        legs_by = {}
        for which, exp in (("this_week", r.expiry), ("next_week", nxt)):
            if exp is None:
                continue
            b = books.get(exp)
            i0 = b.idx(t0)
            if i0 < 0:
                continue
            q = b.quotes(i0, sp, since=int(np.searchsorted(b.ts, AR._ts(r.day, "09:15"))))
            f = b.fwd(q, sp)
            T = (b.close_ts - t0) / YR
            if not f or T <= 0:
                continue
            for strat, target in (("straddle", None), ("strangle", 0.16)):
                legs = AR.pick(b, q, f, T, target)
                if not legs:
                    continue
                if strat == "straddle":
                    legs_by[which] = (b, legs, i0)
                for v, rs in long_trade(b, legs, i0, sess, spot.settle(b.exp)).items():
                    rows.append({**base, "trade": f"long {strat} {which}", "exit": v, "rs": rs})
                if which == "this_week":
                    # the short mirror, held to expiry (what A does on green days)
                    keys = [(kk, t) for t, kk in legs]
                    ent = [b.px[x][i0] for x in keys]
                    st = spot.settle(b.exp)
                    intr = [max(0.0, st - kk) if t == "CE" else max(0.0, kk - st) for t, kk in legs]
                    rs = (sum(ent) - SLIP * len(keys) - sum(intr)) * LOT - _ch("SELL", ent) - _ch("BUY", [x for x in intr if x > 0])
                    rows.append({**base, "trade": f"short {strat} this_week", "exit": "expiry", "rs": rs})
        # double calendar: short this week's ATM straddle, long next week's same strike; close 15:15 near expiry day
        if "this_week" in legs_by and nxt:
            b1, legs, i1 = legs_by["this_week"]
            b2 = books.get(nxt)
            i2 = b2.idx(t0)
            keys = [(kk, t) for t, kk in legs]
            if i2 >= 0 and all(x in b2.px for x in keys):
                j1 = _exit_index(b1, r.expiry, "15:15")
                j2 = b2.idx(AR._ts(r.expiry, "15:15"))
                s_in = [b1.px[x][i1] for x in keys]
                l_in = [b2.px[x][i2] for x in keys]
                s_out = [b1.px[x][j1] for x in keys]
                l_out = [b2.px[x][j2] for x in keys]
                if all(np.isfinite(s_in + l_in + s_out + l_out)):
                    pts = (sum(s_in) - sum(s_out)) + (sum(l_out) - sum(l_in)) - SLIP * 8
                    ch = _ch("SELL", s_in) + _ch("BUY", s_out) + _ch("BUY", l_in) + _ch("SELL", l_out)
                    rows.append({**base, "trade": "double calendar", "exit": "near expiry 15:15", "rs": pts * LOT - ch})
        if n % 50 == 0:
            print(f"\r{n}/{len(h)} {r.day}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def summarize(df):
    df = df.copy()
    df["half"] = np.where(df.day < "2024-04-01", "1st", "2nd")
    g = df.groupby(["trade", "exit", "bucket"])
    s = g.rs.agg(n="count", avg="mean", win=lambda x: (x > 0).mean() * 100, worst="min", total="sum")
    s["t"] = g.rs.apply(lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan)
    hv = df.pivot_table(index=["trade", "exit", "bucket"], columns="half", values="rs", aggfunc="mean")
    s = s.join(hv.rename(columns={"1st": "avg_1st", "2nd": "avg_2nd"}))
    return s.round(1)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    df = run()
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, "trades.parquet"), index=False)
    s = summarize(df)
    s.to_csv(os.path.join(OUT, "summary.csv"))
    pd.set_option("display.width", 220, "display.max_rows", 400)
    print(df.drop_duplicates("day").bucket.value_counts().to_dict(), df.day.min(), df.day.max())
    print(s.to_string())
