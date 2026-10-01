"""Diagnostic (uses FUTURE spot ON PURPOSE, only to measure a data-construction bias): is the availability of
Stage 2A metrics related to how far spot ACTUALLY moved between the snapshot and expiry?
`history_downloader.py` chose each expiry's strike window from that week's realised low/high, so strike coverage
(hence ATM / 25-delta availability) can depend on the future path. Usage:
    python research_output/stage2a/validation/selection_bias_check.py research_output/stage2a/full"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
d = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2a/full"
s = pd.read_csv(f"{d}/smiles.csv")
sp = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet", columns=["ts", "close"]).sort_values("ts")
ts, cl = sp.ts.to_numpy(), sp.close.to_numpy()
ok = s[s.forward_status.notna()].copy()


def t(day, hhmm):
    return int(datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=IST).timestamp())


rng = []
for r in ok.itertuples():
    a = np.searchsorted(ts, t(r.day, r.time), side="left"); b = np.searchsorted(ts, t(r.expiry, "15:30"), side="right")
    seg = cl[a:b]
    rng.append((seg.max() - seg.min()) / r.spot * 100 if len(seg) > 1 else np.nan)
ok["future_range_pct"] = rng
ok = ok[ok.T_days > 1.0].copy()            # expiry-eve snapshots excluded: their tiny horizon would confound the range quartiles
ok["tenor"] = pd.cut(ok.T_days, [1, 3, 7, 14, 30, 1e9], labels=["1-3d", "3-7d", "7-14d", "14-30d", ">30d"])
ok["q"] = ok.groupby("tenor", observed=True).future_range_pct.transform(lambda x: pd.qcut(x, 4, labels=["Q1 calm", "Q2", "Q3", "Q4 volatile"], duplicates="drop"))
ok["fwd_ok"] = ok.forward_status == "ok"; ok["has_atm"] = ok.atm_iv.notna()
ok["has_rr"] = ok.rr25.notna()
print("Share of snapshot-expiries with forward OK / ATM IV / (T>1d) RR-BF, by tenor and by quartile of the FUTURE spot range to expiry")
t1 = ok.groupby(["tenor", "q"], observed=True).agg(n=("fwd_ok", "size"), forward_ok=("fwd_ok", "mean"), atm=("has_atm", "mean"),
                                                    rr_bf=("has_rr", "mean")).round(3)
print(t1.to_string())
for ten in ("1-3d", "3-7d", "7-14d", "14-30d", ">30d"):
    g = ok[ok.tenor == ten]
    print(f"{ten}: Spearman(future range, forward_ok) = {g.future_range_pct.corr(g.fwd_ok.astype(float), method='spearman'):.3f}   "
          f"Spearman(future range, has_atm) = {g.future_range_pct.corr(g.has_atm.astype(float), method='spearman'):.3f}")
