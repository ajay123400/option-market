"""callwrite_put_spread_research.py -- user's idea (2026-10-12, from the 7 Oct 2026 week):
"IV below the threshold + IV crushed + heavy call writing -> bearish -> buy a put debit
spread with the best risk-reward".

Each day D (current weekly expiry with >= 2 sessions left after D), read at D's close:
  iv_low     D's 14:30 ATM IV < that slot's threshold (sell_rules history)
  crush      D's 14:30 ATM IV < the previous session's 14:30 reading
  callwrite  near-ATM (spot +/-500) CE OI change (D close - previous close) >= RATIO x
             PE OI change, and CE change > 0   (RATIO 1.5 / 2 / 3)
  mirror     put writing (PE change >= RATIO x CE change) -> bullish call spread
Trade at D 15:25: 200-wide debit spread, the strike nearest ATM with debit <= 66.7
(>= 1:2), held to expiry; also a 'take +100%' exit. Baselines: the same spread on
every iv_low day / every day.

Usage:  python callwrite_put_spread_research.py
"""
import os
import sys

import numpy as np
import pandas as pd

import ashish_rules_bt as AR
import charges
import paths
import sell_rules as SR

OUT = os.path.join(paths.BASE_DIR, "results", "callwrite_spread")
H = os.path.join(paths.BASE_DIR, "data", "hist1m")
LOT, SLIP, W, MAXD = AR.LOT, 0.5, 200, 200 / 3
IST = "Asia/Kolkata"


def _ch(side, p):
    return charges.order_charges(side, p, LOT) if p > 0 else 0.0


def spread(q, atm, side):
    t, sg = ("CE", 1) if side == "call" else ("PE", -1)
    for step in range(21):
        k = atm + sg * 50 * step
        s = k + sg * W
        if (k, t) in q and (s, t) in q:
            d = q[(k, t)] - q[(s, t)] + 2 * SLIP
            if 0 < d <= MAXD:
                return k, s, d
    return None


def run():
    hist = SR.history()
    iv = hist[hist.slot == "14:30"].set_index("day")[["atm_iv", "thr"]]
    sp = pd.read_parquet(os.path.join(H, "NIFTY50_1m.parquet"))
    t = pd.to_datetime(sp.ts, unit="s", utc=True).dt.tz_convert(IST)
    sp["d"] = t.dt.date.astype(str)
    close = sp.groupby("d").close.last()
    days = list(close.index)
    exps = sorted(f[:10] for f in os.listdir(os.path.join(H, "options")) if f.endswith(".parquet"))
    rows = []
    for e in exps:
        if e < "2022-06-01":
            continue
        o = pd.read_parquet(os.path.join(H, "options", f"{e}.parquet"))
        tt = pd.to_datetime(o.ts, unit="s", utc=True).dt.tz_convert(IST)
        o["d"], o["hm"] = tt.dt.date.astype(str), tt.dt.strftime("%H:%M")
        o = o[o.hm <= "15:25"].sort_values("ts")
        last = o.groupby(["d", "strike", "type"]).agg(px=("close", "last"), oi=("oi", "last"))
        ed = [d for d in days if d <= e]
        if e not in close.index:
            continue
        settle = float(close[e])
        wk = [d for d in sorted(o.d.unique()) if d in close.index and d < e]
        for d in wk:
            k = days.index(d)
            if k == 0 or days[k - 1] not in o.d.unique():
                continue
            left = len([x for x in ed if x > d])
            if left < 2 or d not in iv.index or days[k - 1] not in iv.index:
                continue
            s = float(close[d])
            cur, prv = last.loc[d], last.loc[days[k - 1]]
            near = lambda x: x[(x.index.get_level_values(0) >= s - 500) & (x.index.get_level_values(0) <= s + 500)]
            dOI = (near(cur).oi - near(prv).oi.reindex(near(cur).index)).dropna()
            by = dOI.groupby(level=1)
            if by.size().get("CE", 0) < 5 or by.size().get("PE", 0) < 5:
                continue
            ce, pe = float(by.sum()["CE"]), float(by.sum()["PE"])
            q = {(int(a), b): float(v) for (a, b), v in cur.px.items()}
            atm = int(round(s / 50) * 50)
            rec = {"day": d, "expiry": e, "left": left, "iv": iv.atm_iv[d], "thr": iv.thr[d],
                   "iv_low": iv.atm_iv[d] < iv.thr[d], "crush": iv.atm_iv[d] < iv.atm_iv[days[k - 1]], "ce_doi": ce, "pe_doi": pe}
            for side in ("put", "call"):
                sp_ = spread(q, atm, side)
                if not sp_:
                    continue
                kk, ss, debit = sp_
                ty = "PE" if side == "put" else "CE"
                intr = lambda x: max(0.0, settle - x) if ty == "CE" else max(0.0, x - settle)
                val = intr(kk) - intr(ss)
                ch = _ch("BUY", q[(kk, ty)]) + _ch("SELL", q[(ss, ty)]) + _ch("SELL", intr(kk)) + _ch("BUY", intr(ss))
                rec[f"{side}_rs"] = (val - debit) * LOT - ch
                rec[f"{side}_k"] = kk
                rec[f"{side}_debit"] = debit
            rows.append(rec)
        print(f"\r{e}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def table(df):
    out = []
    mid = "2024-07-01"

    def add(name, x, col):
        x = x.dropna(subset=[col])
        r = x[col]
        out.append({"rule": name, "n": len(r), "avg": r.mean(), "win%": (r > 0).mean() * 100, "worst": r.min(),
                    "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else np.nan,
                    "avg_1st": x[x.day < mid][col].mean(), "avg_2nd": x[x.day >= mid][col].mean()})

    add("put spread: every day", df, "put_rs")
    add("put spread: iv_low", df[df.iv_low], "put_rs")
    add("put spread: iv_low + crush", df[df.iv_low & df.crush], "put_rs")
    for R in (1.5, 2, 3):
        cw = (df.ce_doi > 0) & (df.ce_doi >= R * df.pe_doi.clip(lower=0))
        add(f"put spread: callwrite x{R}", df[cw], "put_rs")
        add(f"put spread: iv_low + crush + callwrite x{R}  <- user's rule", df[df.iv_low & df.crush & cw], "put_rs")
        pw = (df.pe_doi > 0) & (df.pe_doi >= R * df.ce_doi.clip(lower=0))
        add(f"call spread: iv_low + crush + putwrite x{R} (mirror)", df[df.iv_low & df.crush & pw], "call_rs")
    add("call spread: every day", df, "call_rs")
    return pd.DataFrame(out).round(1)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    df = run()
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, "days.parquet"), index=False)
    pd.set_option("display.width", 220)
    print(df.day.min(), df.day.max(), len(df))
    print(table(df).to_string(index=False))
    print(df[df.day >= "2026-10-01"].round(1).to_string(index=False))
