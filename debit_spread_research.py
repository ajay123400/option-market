"""debit_spread_research.py -- 200-pt debit spreads at >= 1:2 risk-reward, direction
from "whichever side's premium is lower" (user's idea, 2026-10-12).

Each day at 09:30 (IV-rule history days), this week's expiry (1-4 days left)
and next week's:
  spread    long K, short K+200 (call) / K-200 (put); K = the strike nearest ATM
            whose debit (with slippage) <= 200/3 (max profit >= 2x max loss)
  direction cheap_leg   ATM CE < ATM PE -> bullish (call spread), else bearish
            cheap_spread ATM call spread (ATM/ATM+200) cheaper than the ATM put
                         spread -> bullish, else bearish
            always_call / always_put   baselines (NIFTY's drift)
  exits     expiry (settled on spot)  |  +100% target / -50% stop, else expiry

1 lot, charges and 0.5 pt slippage per leg per side.
Output: results/debit_spread/trades.parquet + summary.

Usage:  python debit_spread_research.py
"""
import os
import sys

import numpy as np
import pandas as pd

import ashish_rules_bt as AR
import low_iv_research as L

OUT = os.path.join(L.paths.BASE_DIR, "results", "debit_spread")
W, MAXDEBIT = 200, 200 / 3
LOT, SLIP = L.LOT, L.SLIP


def spread_for(q, atm, side):
    """(long K, short K, debit) nearest ATM with debit <= MAXDEBIT, or None."""
    t = "CE" if side == "call" else "PE"
    sgn = 1 if side == "call" else -1
    for step in range(0, 21):
        k = atm + sgn * 50 * step
        s = k + sgn * W
        if (k, t) not in q or (s, t) not in q:
            continue
        debit = q[(k, t)] - q[(s, t)] + 2 * SLIP
        if 0 < debit <= MAXDEBIT:
            return k, s, debit
    return None


def run_trade(book, side, k, s, debit, i0, settle):
    t = "CE" if side == "call" else "PE"
    a, b = book.px[(k, t)], book.px[(s, t)]
    ch_in = L._ch("BUY", [a[i0]]) + L._ch("SELL", [b[i0]])
    intr = lambda x, kk: max(0.0, x - kk) if t == "CE" else max(0.0, kk - x)
    val_exp = intr(settle, k) - intr(settle, s)
    ch_exp = L._ch("SELL", [intr(settle, k)] if intr(settle, k) > 0 else []) + L._ch("BUY", [intr(settle, s)] if intr(settle, s) > 0 else [])
    out = {"expiry": (val_exp - debit) * LOT - ch_in - ch_exp}
    path = a[i0 + 1:] - b[i0 + 1:] - 2 * SLIP
    hit = np.flatnonzero((path >= 2 * debit) | (path <= 0.5 * debit))
    if len(hit):
        j = i0 + 1 + hit[0]
        out["tgt100_stop50"] = (path[hit[0]] - debit) * LOT - ch_in - L._ch("SELL", [a[j]]) - L._ch("BUY", [b[j]])
    else:
        out["tgt100_stop50"] = out["expiry"]
    return out


def run():
    h = L.iv_pct()
    man = AR.S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live"))
    spot = AR.Spot()
    books = L.Books()
    rows = []
    for n, r in enumerate(h.itertuples()):
        if not (1 <= r.dte <= 4) or r.expiry not in exps:
            continue
        nxt = [x for x in exps if x > r.expiry]
        nxt = nxt[0] if nxt and (pd.Timestamp(nxt[0]) - pd.Timestamp(r.day)).days <= 12 else None
        t0 = AR._ts(r.day, "09:30")
        sp = spot.at(t0)
        for which, exp in (("this_week", r.expiry), ("next_week", nxt)):
            if exp is None:
                continue
            b = books.get(exp)
            i0 = b.idx(t0)
            if i0 < 0:
                continue
            q = b.quotes(i0, sp, since=int(np.searchsorted(b.ts, AR._ts(r.day, "09:15"))))
            f = b.fwd(q, sp)
            if not f:
                continue
            atm = int(round(sp / 50) * 50)
            if (atm, "CE") not in q or (atm, "PE") not in q:
                continue
            cs = (atm, "CE") in q and (atm + W, "CE") in q and q[(atm, "CE")] - q[(atm + W, "CE")]
            ps = (atm, "PE") in q and (atm - W, "PE") in q and q[(atm, "PE")] - q[(atm - W, "PE")]
            if not cs or not ps:
                continue
            calls = {"cheap_leg": "call" if q[(atm, "CE")] < q[(atm, "PE")] else "put",
                     "cheap_spread": "call" if cs < ps else "put",
                     "always_call": "call", "always_put": "put"}
            settle = spot.settle(exp)
            res = {}
            for side in ("call", "put"):
                sp_ = spread_for(q, atm, side)
                if sp_:
                    k, s, debit = sp_
                    res[side] = (k, s, debit, run_trade(b, side, k, s, debit, i0, settle))
            for rule, side in calls.items():
                if side not in res:
                    continue
                k, s, debit, out = res[side]
                for ex, rs in out.items():
                    rows.append({"day": r.day, "expiry": exp, "which": which, "pct": r.pct, "bucket": L._bucket(r.pct),
                                 "rule": rule, "side": side, "long_k": k, "otm": abs(k - sp), "debit": debit, "exit": ex, "rs": rs})
        if n % 50 == 0:
            print(f"\r{n}/{len(h)} {r.day}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def summarize(df, by):
    df = df.copy()
    df["half"] = np.where(df.day < "2024-04-01", "1st", "2nd")
    g = df.groupby(by)
    s = g.rs.agg(n="count", avg="mean", win=lambda x: (x > 0).mean() * 100, worst="min", total="sum")
    s["t"] = g.rs.apply(lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan)
    hv = df.pivot_table(index=by, columns="half", values="rs", aggfunc="mean")
    return s.join(hv.rename(columns={"1st": "avg_1st", "2nd": "avg_2nd"})).round(1)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    df = run()
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, "trades.parquet"), index=False)
    pd.set_option("display.width", 220, "display.max_rows", 400)
    print(df.day.min(), df.day.max())
    one = df[df.rule == "always_call"].drop_duplicates(["day", "which"])
    print("call side: debit avg", round(one.debit.mean(), 1), "long strike OTM avg pts", round(one.otm.mean()))
    one = df[df.rule == "always_put"].drop_duplicates(["day", "which"])
    print("put side: debit avg", round(one.debit.mean(), 1), "long strike OTM avg pts", round(one.otm.mean()))
    for rule in ("cheap_leg", "cheap_spread"):
        x = df[(df.rule == rule) & (df.exit == "expiry")].drop_duplicates(["day", "which"])
        print(rule, "picks", x.side.value_counts().to_dict())
    s = summarize(df, ["which", "exit", "rule"])
    print(s.to_string())
    s.to_csv(os.path.join(OUT, "summary.csv"))
    print(summarize(df[df.exit == "expiry"], ["which", "rule", "bucket"]).to_string())
