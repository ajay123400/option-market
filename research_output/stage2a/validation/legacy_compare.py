"""Compare Stage 2A (09:30 snapshots) with the existing results/iv_surface/days.parquet (iv_surface_research.py).
Build first:  python -m optionsengine.research.build_surface --times 09:30 --max-dte 9 --out research_output/stage2a/legacy_cmp
Usage:        python research_output/stage2a/validation/legacy_compare.py research_output/stage2a/legacy_cmp"""
import sys

import numpy as np
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2a/legacy_cmp"
mine = pd.read_csv(f"{d}/smiles.csv")
old = pd.read_parquet("results/iv_surface/days.parquet")
m = mine.merge(old, on=["expiry", "day"], how="inner", suffixes=("", "_old"))
print(f"legacy rows {len(old)}; mine attempted {len(mine)}; matched on (expiry, day) {len(m)}")
print("my forward status on matched rows:", m.forward_status.value_counts(dropna=False).to_dict())
ok = m[m.forward_status == "ok"].copy()
ok["fwd_diff"] = ok.forward_used - ok.fwd
print(f"\nFORWARD  (mine fresh-quote parity gate vs legacy synthetic_forward, n={len(ok)}): median diff {ok.fwd_diff.median():.2f} pts, "
      f"median |diff| {ok.fwd_diff.abs().median():.2f}, p95 |diff| {ok.fwd_diff.abs().quantile(.95):.2f}, max {ok.fwd_diff.abs().max():.1f}")
a = ok[ok.atm_iv.notna()].copy()
a["d"] = (a.atm_iv - a.atm_iv_old) * 100
print(f"\nATM IV   (mine interpolated at F vs legacy mean(CE,PE) at round(F/50)*50, n={len(a)}): median diff {a.d.median():.3f} vol pts, "
      f"median |diff| {a.d.abs().median():.3f}, p95 |diff| {a.d.abs().quantile(.95):.3f}, p99 {a.d.abs().quantile(.99):.2f}")
for lab, g in (("T<=2d", a[a.T_days <= 2]), ("2-5d", a[(a.T_days > 2) & (a.T_days <= 5)]), ("5-9d", a[a.T_days > 5])):
    print(f"   {lab}: n={len(g)} median |diff| {g.d.abs().median():.3f} p95 {g.d.abs().quantile(.95):.3f}")
s = ok[ok.rr25.notna() & ok.skew25.notna()].copy()
s["mine_skew"] = -s.rr25 * 100              # legacy skew25 = IV(25d put) - IV(25d call) = -RR25
s["old_skew"] = s.skew25 * 100
s["d"] = s.mine_skew - s.old_skew
print(f"\nSKEW25   (mine -RR25 vs legacy put25-call25, n={len(s)}): median diff {s.d.median():.3f} vol pts, median |diff| {s.d.abs().median():.3f}, "
      f"p95 |diff| {s.d.abs().quantile(.95):.3f}; correlation {np.corrcoef(s.mine_skew, s.old_skew)[0,1]:.3f}")
print(f"coverage on matched OK rows: mine ATM {ok.atm_iv.notna().mean():.3f}; legacy has ATM on all {len(old)} rows by construction")
print(f"matched legacy rows where mine is NOT primary/available: {len(m) - len(ok)} (forward gate/unavailable)")
