"""Empirical checks of the sampling and annualization conventions (real data, complete regular sessions only).
Independent of the module: own pandas code.   Usage: python .../sampling_and_annualization_check.py [out_dir]"""
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2b"
dq = pd.read_csv(f"{OUT}/data_quality.csv")
good = set(pd.to_datetime(dq[(dq.session_type == "regular") & dq.complete].session_date).map(lambda d: (d.date() - pd.Timestamp("1970-01-01").date()).days))
df = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet")
ist = df.ts + 19800
df["dn"], df["slot"] = ist // 86400, (ist % 86400) // 60 - 555
df = df[df.dn.isin(good) & (df.slot >= 0) & (df.slot < 375)].sort_values(["dn", "slot"])
cl = df.pivot(index="dn", columns="slot", values="close")
op = df[df.slot == 0].set_index("dn").open
print(f"complete regular sessions used: {len(cl)}")
print("\n=== signature plot: mean daily intraday variance by sampling interval (sum of squared k-minute close-to-close returns) ===")
base = None
rows = []
for k in (1, 2, 3, 5, 10, 15, 25, 75, 375):
    c = cl[[j for j in range(k - 1, 375, k)]]               # closes at the end of minutes k-1, 2k-1, ... (non-overlapping)
    r = np.log(c.div(c.shift(1, axis=1))).iloc[:, 1:]
    first = np.log(c.iloc[:, 0] / op)                       # open -> first k-minute close
    ss = (r ** 2).sum(axis=1) + first ** 2
    m = ss.mean()
    base = base or m
    rows.append((k, int(r.shape[1] + 1), m, 100 * np.sqrt(252 * m), m / base))
print(pd.DataFrame(rows, columns=["minutes", "returns/session", "mean_daily_var", "ann_vol_pct", "ratio_to_1min"]).round(8).to_string(index=False))
print("(ratio_to_1min ~ 1 means no material microstructure bias in the 1-minute sampling; >1 / <1 shows noise / mean reversion)")

print("\n=== consistency of the high-frequency daily variance with the daily close-to-close second moment ===")
d = pd.read_csv(f"{OUT}/daily_rv.csv")
d = d[(d.session_type == "regular") & d.complete & d.cc_return.notna() & d.overnight_return.notna()].copy()
d["year"] = d.session_date.str[:4]
t = d.groupby("year").agg(n=("cc_return", "size"), mean_cc2=("cc_variance", "mean"), mean_intra=("intraday_variance", "mean"),
                          mean_on2=("overnight_variance", "mean"), mean_hyb=("hybrid_variance", "mean"))
t["hyb/cc2"] = t.mean_hyb / t.mean_cc2; t["overnight_share_of_hyb"] = t.mean_on2 / t.mean_hyb
print(t.round(8).to_string())
tot = d[["cc_variance", "intraday_variance", "overnight_variance", "hybrid_variance"]].mean()
print(f"ALL: mean(cc^2) {tot.cc_variance:.6e}  mean(hybrid) {tot.hybrid_variance:.6e}  ratio hybrid/cc^2 = {tot.hybrid_variance / tot.cc_variance:.4f}   "
      f"overnight share of hybrid = {tot.overnight_variance / tot.hybrid_variance:.3f}   intraday-only / cc^2 = {tot.intraday_variance / tot.cc_variance:.3f}")
print("=> intraday-only RV is NOT comparable with close-to-close RV: it omits the overnight component (share above).")

print("\n=== annualization: sessions per calendar year (regular sessions) ===")
reg = pd.read_csv(f"{OUT}/data_quality.csv"); reg = reg[reg.session_type == "regular"]
cnt = reg.groupby(reg.session_date.str[:4]).size()
print(cnt.to_dict(), " (2021 partial from Sep 16, 2026 partial to Sep 30)")
full = cnt.loc[["2022", "2023", "2024", "2025"]]
print(f"full-year mean {full.mean():.1f} sessions -> sqrt(252/{full.mean():.1f}) - 1 = {np.sqrt(252 / full.mean()) - 1:+.4%} relative vol difference vs a sessions-per-year convention")
print(f"effect of the convention on any RV (vol scales with sqrt(A)): A=246 (observed): x{np.sqrt(246 / 252):.4f}   A=248: x{np.sqrt(248 / 252):.4f}   "
      f"A=250: x{np.sqrt(250 / 252):.4f}   A=252 (default): x1   A=365 (calendar-day basis, NOT used): x{np.sqrt(365 / 252):.4f}")
