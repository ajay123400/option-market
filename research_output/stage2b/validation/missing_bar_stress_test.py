"""How wrong is the coverage-qualified scaling  V * 375 / n_valid  when minutes are missing? (real complete sessions)
Bars are deleted from COMPLETE sessions on purpose and the scaled variance is compared with the true full-session
variance. Own numpy code (no module imports). Usage: python .../missing_bar_stress_test.py [out_dir]"""
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
O = df.pivot(index="dn", columns="slot", values="open").to_numpy(); C = df.pivot(index="dn", columns="slot", values="close").to_numpy()
rng = np.random.default_rng(42)
pick = rng.choice(len(C), 400, replace=False)


def stats(c, o, drop):
    c = c.copy(); o = o.copy(); c[drop] = np.nan; o[drop] = np.nan
    r = np.full(375, np.nan); r[0] = np.log(c[0] / o[0]); r[1:] = np.log(c[1:] / c[:-1])
    ok = np.isfinite(r)
    return float(np.sum(r[ok] ** 2)), int(ok.sum())


rows = []
for target in (0.99, 0.98, 0.95, 0.90):
    lost = round((1 - target) * 375)                         # returns to lose
    for scen in ("scattered", "open_block", "midday_block", "close_block"):
        errs, covs = [], []
        for i in pick:
            c, o = C[i], O[i]
            true, _ = stats(c, o, np.array([], dtype=int))
            if scen == "scattered":
                k = max(1, lost // 2)
                drop = rng.choice(np.arange(1, 374), k, replace=False)
            else:
                B = max(1, lost - 1)                          # B consecutive missing bars lose B+1 returns
                start = {"open_block": 0, "close_block": 375 - B, "midday_block": int(rng.integers(100, 375 - B - 100))}[scen]
                drop = np.arange(start, start + B)
            ss, n = stats(c, o, drop)
            errs.append((ss * 375 / n) / true - 1); covs.append(n / 375)
        e = np.array(errs)
        rows.append(dict(target_coverage=target, scenario=scen, mean_actual_coverage=round(float(np.mean(covs)), 4), mean_bias_pct=round(100 * e.mean(), 2),
                         median_abs_error_pct=round(100 * np.median(np.abs(e)), 2), p95_abs_error_pct=round(100 * np.quantile(np.abs(e), .95), 2),
                         worst_underestimate_pct=round(100 * e.min(), 1), worst_overestimate_pct=round(100 * e.max(), 1)))
print(pd.DataFrame(rows).to_string(index=False))
print("\nreading: error is on the session VARIANCE; the annualised vol error is about half of it (sqrt).")
