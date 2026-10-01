"""Results BEFORE vs AFTER the data-quality exclusions (real data). Independent pandas code (no module imports).

BEFORE (naive):  every record of an IST date, returns between consecutive records of the whole file (overnight gaps,
                 pre-open / evening / Saturday bars and missing minutes all become ordinary one-minute returns),
                 rolling windows over every calendar date that has bars.
AFTER  (module): strict method from research_output/stage2b/rolling_rv.csv.
Usage: python research_output/stage2b/validation/quality_exclusion_comparison.py [out_dir]"""
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2b"
df = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet").sort_values("ts").reset_index(drop=True)
ist = df.ts + 19800
df["date"] = pd.to_datetime((ist // 86400) * 86400, unit="s").dt.strftime("%Y-%m-%d")
df["r"] = np.log(df.close / df.close.shift(1))                   # consecutive RECORDS, regardless of any gap
naive = df.groupby("date").r.agg(n="count", ss=lambda x: float((x.dropna() ** 2).sum()))
naive["naive_intraday_vol"] = 100 * np.sqrt(252 * naive.ss)
d = pd.read_csv(f"{OUT}/daily_rv.csv").set_index("session_date")
m = d.join(naive, how="left")
reg = m[m.session_type == "regular"]
print("=== daily intraday vol %: naive vs module (complete regular sessions only for the module) ===")
both = reg[reg.complete & reg.naive_intraday_vol.notna()]
diff = both.naive_intraday_vol - both.intraday_vol_ann_pct
print(f"sessions compared {len(both)}: median diff {diff.median():+.3f} vol pts, p95 |diff| {diff.abs().quantile(.95):.3f}, max {diff.abs().max():.3f}")
print("  (the naive series wrongly adds the previous close -> 09:15 open overnight gap as a one-minute return on EVERY session)")
big = diff.abs().sort_values(ascending=False).head(5)
print("  five largest daily differences:", {k: round(float(both.loc[k, 'naive_intraday_vol'] - both.loc[k, 'intraday_vol_ann_pct']), 2) for k in big.index})
print("\n=== sessions the naive calculation would have used but the module excludes ===")
ex = d[(d.session_type != "regular") | (~d.complete.astype(bool))]
print(ex.session_type.value_counts().to_dict(), "| flags:", ex.quality_flags.value_counts().to_dict())
for k, r in ex.iterrows():
    nv = naive.loc[k] if k in naive.index else None
    print(f"  {k} {r.session_type:16s} bars={int(r.n_records):4d} regular-window={int(r.n_regular_bars):3d} flags={r.quality_flags} | naive daily 'vol' = " + (f"{nv.naive_intraday_vol:7.2f} %" if nv is not None else "n/a"))
print("\n=== rolling 20-session RV: before (naive, all dates) vs after (strict) ===")
nv = naive.copy(); nv["roll20_naive"] = 100 * np.sqrt(252 * nv.ss.rolling(20).mean())
roll = pd.read_csv(f"{OUT}/rolling_rv.csv"); s = roll[(roll.method == "strict") & (roll.measure == "intraday") & (roll.window_sessions == 20)].set_index("session_date")
j = s.join(nv.roll20_naive, how="inner")
print(f"rows with a strict value: {int(j.rv_ann_pct.notna().sum())} of {len(j)}  |  rows where naive gives a number but strict does not: {int((j.rv_ann_pct.isna() & j.roll20_naive.notna()).sum())}")
dd = (j.roll20_naive - j.rv_ann_pct).dropna()
print(f"where both exist: median naive-strict {dd.median():+.3f} vol pts, p95 |diff| {dd.abs().quantile(.95):.3f}, max |diff| {dd.abs().max():.3f}")
print("\n=== effect of excluding ONLY the 4 incomplete regular sessions (data-quality), using coverage-qualified windows ===")
cq = roll[(roll.method == "coverage_qualified") & (roll.measure == "intraday") & (roll.window_sessions == 20)].set_index("session_date")
k = s[["rv_ann_pct", "n_obs"]].join(cq[["rv_ann_pct", "n_obs"]], rsuffix="_cq")
print(f"windows with strict value {int(k.rv_ann_pct.notna().sum())}; with coverage-qualified value {int(k.rv_ann_pct_cq.notna().sum())}; windows gained by cq {int((k.rv_ann_pct.isna() & k.rv_ann_pct_cq.notna()).sum())}")
both2 = k.dropna()
print(f"where both exist: max |cq - strict| = {(both2.rv_ann_pct_cq - both2.rv_ann_pct).abs().max():.6f} vol pts (they coincide when the window has no incomplete session)")
