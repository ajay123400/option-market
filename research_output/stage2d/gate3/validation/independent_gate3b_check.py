"""Independent recomputation of Stage 2D Gate 3b (T11) quantities (no optionsengine imports except the synthetic-world writer used as the object under test).

Tier A: a small synthetic world is written to disk; the realised total variance of every truth row is recomputed from the RAW 1-minute bars (log-return squares of the bars after the
snapshot bar plus overnight-gap squares and the full later sessions up to the expiry), and the written ATM option prices are inverted with scipy brentq (Black-76, r = 6.5%) and
compared with the generator's true IV (the difference is only the 0.05 tick rounding).
Tier B: coverage, Wilson intervals, band flags, contrast coverage / rejection rates and Wald rejection rates are recomputed from the per-history CSVs and compared with the aggregated CSVs.
Usage: python research_output/stage2d/gate3/validation/independent_gate3b_check.py [gate3_dir]
"""
import math
import os
import sys
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import scipy.optimize as so

from optionsengine.research.stage2d import synth_pipeline as SPL

G3 = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2d/gate3"
bad = 0
R = 0.065


def check(name, mine, theirs, tol=1e-9):
    global bad
    ok = abs(mine - theirs) <= tol * max(1.0, abs(theirs))
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {name:80s} mine={mine:.10g} output={theirs:.10g}")


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# ------------------------------------------------------------------ Tier A
print("Tier A: realised variance from raw bars and IV from written option prices")
d = SPL.TierADGP(n_weeks=8, c=1.3)
with tempfile.TemporaryDirectory() as tmp:
    root, pdir = os.path.join(tmp, "w"), os.path.join(tmp, "p")
    truth = SPL.generate_world(d, 5, root, pdir, write=True)
    bars = pd.read_parquet(os.path.join(root, "NIFTY50_1m.parquet"))
    bars["day"] = pd.to_datetime(bars.ts, unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.date.astype(str)
    days = sorted(bars.day.unique())
    per = {}
    prev_close = None
    for dy in days:
        b = bars[bars.day == dy].sort_values("ts")
        o, c = b.open.to_numpy(), b.close.to_numpy()
        gap = 0.0 if prev_close is None else math.log(o[0] / prev_close)
        per[dy] = (np.log(c / o) ** 2, gap ** 2)
        prev_close = c[-1]
    slot_bar = {"10:00": 45, "13:00": 225, "15:00": 345}                    # snapshot bar = the bar that starts at the slot time (observation = its close, 10:01 etc.)
    worst = 0.0
    sample = truth.sample(min(150, len(truth)), random_state=0)
    for _, r in sample.iterrows():
        i = days.index(r.day); e = days.index(r.expiry)
        v = per[r.day][0][slot_bar[r.time] + 1:].sum()
        for j in range(i + 1, e + 1):
            v += per[days[j]][1] + per[days[j]][0].sum()
        worst = max(worst, abs(v - r.V_true) / r.V_true)
    check("max relative error of V_true vs raw-bar recomputation (150 rows)", worst, 0.0, 1e-10)
    # IV from option prices
    opt = pd.read_parquet(os.path.join(root, "options", f"{truth.expiry.iloc[len(truth) // 2]}.parquet"))
    exp = truth.expiry.iloc[len(truth) // 2]
    tt = truth[truth.expiry == exp]
    errs = []
    for _, r in tt.sample(min(12, len(tt)), random_state=1).iterrows():
        t_bar = int(pd.Timestamp(f"{r.day} 09:15", tz="Asia/Kolkata").timestamp()) + 60 * slot_bar[r.time]
        q = opt[opt.ts == t_bar]
        F = r.spot * math.exp(R * r.T_years)
        k0 = q.strike.iloc[(q.strike - F).abs().argsort().iloc[0]]
        for typ in ("CE", "PE"):
            px = float(q[(q.strike == k0) & (q.type == typ)].close.iloc[0])
            df = math.exp(-R * r.T_years)
            def f(s):
                sd = s * math.sqrt(r.T_years); d1 = (math.log(F / k0) + 0.5 * sd * sd) / sd; d2 = d1 - sd
                N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
                return df * (F * N(d1) - k0 * N(d2)) - px if typ == "CE" else df * (k0 * N(-d2) - F * N(-d1)) - px
            iv = so.brentq(f, 0.01, 3.0)
            # the smile is skewed: compare with the generator's skew formula at this strike
            x = math.log(k0 / F)
            iv_gen = r.iv_true * max(1 + d.skew * x + d.curv * x * x, 0.3)
            errs.append(abs(iv - iv_gen) * 100)
    print(f"  max |IV(brentq on written price) - generator IV at that strike| = {max(errs):.5f} vol pts (tick rounding only)")
    bad += max(errs) > 0.15

# ------------------------------------------------------------------ Tier B
print("Tier B: coverage / contrast / Wald aggregation from the per-history CSVs")
cov = pd.read_csv(os.path.join(G3, "t11b_coverage.csv"))
truth_b = pd.read_csv(os.path.join(G3, "t11b_truth.csv"))
hi_ = pd.read_csv(os.path.join(G3, "t11b_history_intervals.csv"))
for (sc, m, es), g in hi_.groupby(["scenario", "method", "estimand"]):
    t = float(truth_b[(truth_b.scenario == sc) & (truth_b.estimand == es)].population_value.iloc[0])
    g = g[np.isfinite(g.ci_lo) & np.isfinite(g.ci_hi)]
    k = int(((g.ci_lo <= t) & (t <= g.ci_hi)).sum())
    row = cov[(cov.scenario == sc) & (cov.method == m) & (cov.estimand == es)].iloc[0]
    check(f"coverage {sc}/{m}/{es}", k / len(g), row.coverage)
    lo, hi = wilson(k, len(g))
    check(f"  wilson_lo {sc}/{m}/{es}", lo, row.wilson_lo, 1e-6)
    check(f"  band flag {sc}/{m}/{es}", float(0.93 <= k / len(g) <= 0.97), float(row.in_target_band_93_97))
    check(f"  undercovers flag {sc}/{m}/{es}", float(hi < 0.93), float(row.undercovers))

con = pd.read_csv(os.path.join(G3, "t11b_split_contrast.csv"))
hc = pd.read_csv(os.path.join(G3, "t11b_history_contrasts.csv"))
for (sc, tk, b, es), g in hc.groupby(["scenario", "truth_kind", "block", "estimand"]):
    row = con[(con.scenario == sc) & (con.truth_kind == tk) & (con.block == b) & (con.estimand == es)].iloc[0]
    t = row.true_difference
    g = g[np.isfinite(g.ci_lo) & np.isfinite(g.ci_hi)]
    check(f"contrast coverage {sc}/{tk}/b{b}/{es}", float(((g.ci_lo <= t) & (t <= g.ci_hi)).mean()), row.coverage_of_true_difference)
    check(f"  rejection {sc}/{tk}/b{b}/{es}", float((g.p_value < 0.05).mean()), row.rejection_rate_p_lt_005)

wl = pd.read_csv(os.path.join(G3, "t11b_wald.csv"))
hw = pd.read_csv(os.path.join(G3, "t11b_history_wald.csv"))
for (sc, tst, st_), g in hw.groupby(["scenario", "test", "statistic"]):
    row = wl[(wl.scenario == sc) & (wl.test == tst) & (wl.statistic == st_)].iloc[0]
    check(f"wald rejection {sc}/{tst}/{st_}", float((g.p_value.dropna() < 0.05).mean()), row.rejection_rate_alpha_005)

print("\nRESULT:", "ALL CHECKS PASSED" if bad == 0 else f"{bad} CHECKS FAILED")
sys.exit(1 if bad else 0)
