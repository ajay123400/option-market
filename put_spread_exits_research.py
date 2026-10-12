"""put_spread_exits_research.py -- exits for the put debit spreads of
callwrite_put_spread_research.py (2026-10-12): does booking early ("+100% target",
next-day exit) beat holding to expiry?

Uses results/callwrite_spread/days.parquet (one row per day: nearest expiry with
>= 2 sessions left; entry 15:25, 200-wide put spread, debit <= 66.7). For each
trade the 1-min spread value (long - short - exit slippage) after entry:
  expiry        settle on spot (the earlier result)
  t100          +100% target, else expiry
  t50           +50% target, else expiry
  t100_s50      +100% target / -50% stop, else expiry
  t100_d1       +100% target, else close next session 15:25
  d1            close next session 15:25
  d2            close 2 sessions later 15:25
Groups: every day, IV low + crush, IV low + crush + call writing (user's rule).

Usage:  python put_spread_exits_research.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import charges
import paths

H = os.path.join(paths.BASE_DIR, "data", "hist1m")
SRC = os.path.join(paths.BASE_DIR, "results", "callwrite_spread", "days.parquet")
OUT = os.path.join(paths.BASE_DIR, "results", "callwrite_spread", "exits.parquet")
LOT, SLIP, W = 75, 0.5, 200
IST = "Asia/Kolkata"


def _ch(side, p):
    return charges.order_charges(side, p, LOT) if p > 0 else 0.0


def main():
    d = pd.read_parquet(SRC).sort_values(["day", "expiry"]).drop_duplicates("day").dropna(subset=["put_k"])
    import manual_trades
    global LOT
    LOT = manual_trades.LOT_SIZE
    sp = pd.read_parquet(os.path.join(H, "NIFTY50_1m.parquet"))
    sp["d"] = pd.to_datetime(sp.ts, unit="s", utc=True).dt.tz_convert(IST).dt.date.astype(str)
    close = sp.groupby("d").close.last()
    days = list(close.index)
    rows = []
    for exp, g in d.groupby("expiry"):
        ks = sorted({int(k) for k in g.put_k} | {int(k) - W for k in g.put_k})
        o = pq.read_table(os.path.join(H, "options", f"{exp}.parquet"), columns=["type", "strike", "ts", "close"],
                          filters=[("type", "=", "PE"), ("strike", "in", ks)]).to_pandas()
        t = pd.to_datetime(o.ts, unit="s", utc=True).dt.tz_convert(IST)
        o["d"], o["hm"] = t.dt.date.astype(str), t.dt.strftime("%H:%M")
        px = o.pivot_table(index="ts", columns="strike", values="close").sort_index().ffill()
        meta = o.drop_duplicates("ts").set_index("ts").loc[px.index, ["d", "hm"]]
        settle = float(close[exp])
        for r in g.itertuples():
            k, s = int(r.put_k), int(r.put_k) - W
            if k not in px or s not in px:
                continue
            ent = (meta.d == r.day) & (meta.hm <= "15:25")
            if not ent.any():
                continue
            i0 = np.flatnonzero(ent.to_numpy())[-1]
            a, b = px[k].to_numpy(), px[s].to_numpy()
            debit = a[i0] - b[i0] + 2 * SLIP
            ch_in = _ch("BUY", a[i0]) + _ch("SELL", b[i0])
            after = (meta.d.to_numpy()[i0 + 1:] <= exp) & (meta.hm.to_numpy()[i0 + 1:] <= "15:29")
            path = (a[i0 + 1:] - b[i0 + 1:] - 2 * SLIP)[after]
            pa, pb = a[i0 + 1:][after], b[i0 + 1:][after]
            pd_, ph = meta.d.to_numpy()[i0 + 1:][after], meta.hm.to_numpy()[i0 + 1:][after]

            def at(j):
                return (path[j] - debit) * LOT - ch_in - _ch("SELL", pa[j]) - _ch("BUY", pb[j])

            iv_ = lambda x: max(0.0, x - settle)
            exp_rs = (iv_(k) - iv_(s) - debit) * LOT - ch_in - _ch("SELL", iv_(k)) - _ch("BUY", iv_(s))

            def first(cond, fallback):
                j = np.flatnonzero(cond)
                return at(j[0]) if len(j) else fallback

            def day_close(n):
                i = days.index(r.day)
                if i + n >= len(days) or days[i + n] >= exp:
                    return exp_rs
                j = np.flatnonzero((pd_ == days[i + n]) & (ph <= "15:25"))
                return at(j[-1]) if len(j) else exp_rs

            d1 = day_close(1)
            i1 = days.index(r.day) + 1
            in_d1 = (pd_ <= days[i1]) if i1 < len(days) else np.ones(len(pd_), bool)
            rows.append({"day": r.day, "expiry": exp, "iv_low": r.iv_low, "crush": r.crush, "ce_doi": r.ce_doi, "pe_doi": r.pe_doi,
                         "debit": debit, "max_value": float(path.max()) if len(path) else np.nan,
                         "expiry_rs": exp_rs,
                         "t100": first(path >= 2 * debit, exp_rs),
                         "t50": first(path >= 1.5 * debit, exp_rs),
                         "t100_s50": first((path >= 2 * debit) | (path <= 0.5 * debit), exp_rs),
                         "t100_d1": first((path >= 2 * debit) & in_d1, d1),
                         "d1": d1, "d2": day_close(2)})
        print(f"\r{exp}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    x = main()
    x.to_parquet(OUT, index=False)
    cw = (x.ce_doi > 0) & (x.ce_doi >= 1.5 * x.pe_doi.clip(lower=0))
    groups = {"every day": x.day.notna(), "IV low + crush": x.iv_low & x.crush, "user's rule (+ call writing)": x.iv_low & x.crush & cw}
    cols = ["expiry_rs", "t100", "t50", "t100_s50", "t100_d1", "d1", "d2"]
    for name, m in groups.items():
        y = x[m]
        print(f"\n== {name}: {len(y)} trades;  touched +100% at some point: {(y.max_value >= 2 * y.debit).mean()*100:.0f}%")
        out = []
        for c in cols:
            v = y[c]
            h1, h2 = y[y.day < "2024-07-01"][c], y[y.day >= "2024-07-01"][c]
            out.append({"exit": c, "avg": v.mean(), "win%": (v > 0).mean() * 100, "worst": v.min(), "total": v.sum(),
                        "t": v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), "avg_1st": h1.mean(), "avg_2nd": h2.mean()})
        print(pd.DataFrame(out).round(1).to_string(index=False))
        print(y.assign(yr=y.day.str[:4]).groupby("yr")[["expiry_rs", "t100", "d1"]].mean().round(0).T.to_string())
