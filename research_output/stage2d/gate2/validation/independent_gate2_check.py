"""Independent recomputation of Stage 2D Gate 2 quantities (no optionsengine imports; csv/math/pandas/scipy only).

* T5: dev and holdout point estimates and differences (S1 calendar, R1, R2), the post-stratified holdout S1/R1, Holm adjustment
* T6: stratum point estimates for every dimension (S1 calendar, R1), Mincer-Zarnowitz OLS slope, expiry-day stratum (F5 hybrid)
* T7: the 5-minute and 15-minute expiry-aligned RV from the RAW 1-minute spot bars (pure Python) and the resulting S1 calendar / R1 point estimates
* T7 rate variants: Black-76 implied volatility re-solved by bisection from the stored price, forward and T of sampled points (r = 6.5% baseline, 5.5%, 7.5% runs)
Usage: python research_output/stage2d/gate2/validation/independent_gate2_check.py [gate2_dir] [aligned_csv] [stage2c_dir]
"""
import csv
import math
import os
import random
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import scipy.stats as st

G2 = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2d/gate2"
AL = sys.argv[2] if len(sys.argv) > 2 else "research_output/stage2c/annualization_audit/aligned_observations.csv"
bad = 0


def check(name, mine, theirs, tol=1e-9):
    global bad
    ok = (math.isnan(mine) and math.isnan(theirs)) or abs(mine - theirs) <= tol * max(1.0, abs(theirs))
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {name:70s} mine={mine:.10g} output={theirs:.10g}")


def median(x):
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


rows = [r for r in csv.DictReader(open(AL))]
exp = [r for r in rows if r["horizon"] == "EXP" and r["measure"] == "hybrid"]
f = lambda r, k: float(r[k])
print(f"EXP/hybrid rows {len(exp)}")


def s1cal(rs):
    return median([f(r, "spread_calendar") for r in rs])


def r1(rs):
    v = [math.log(f(r, "implied_total_variance") / f(r, "rv_total_variance")) for r in rs if f(r, "rv_total_variance") > 0]
    return math.exp(sum(v) / len(v))


def r2(rs):
    return median([f(r, "implied_total_variance") / f(r, "rv_total_variance") for r in rs if f(r, "rv_total_variance") > 0])


# ------------------------------------------------------------------------------ T5
print("\n[T5] dev vs holdout")
con = {r["estimand"]: r for r in csv.DictReader(open(f"{G2}/t5_split_contrasts.csv"))}
dev = [r for r in exp if r["split"] == "dev"]
hold = [r for r in exp if r["split"] == "holdout"]
for name, fn, key in (("S1 calendar", s1cal, "S1_calendar_all_obs_median"), ("R1", r1, "R1_geometric_mean_all_obs"), ("R2", r2, "R2_median_ratio_all_obs")):
    check(f"{name} dev", fn(dev), float(con[key]["dev_estimate"]))
    check(f"{name} holdout", fn(hold), float(con[key]["holdout_estimate"]))
    check(f"{name} difference", fn(hold) - fn(dev), float(con[key]["difference_holdout_minus_dev"]))
# Holm over the four family-A p-values
fam = ["S1_calendar_all_obs_median", "S2_calendar_median_of_expiry_medians", "R1_geometric_mean_all_obs", "R2_median_ratio_all_obs"]
ps_ = [float(con[k]["p_two_sided_bootstrap"]) for k in fam]
order = sorted(range(4), key=lambda i: ps_[i])
run, adj = 0.0, [0.0] * 4
for rank, i in enumerate(order):
    run = max(run, (4 - rank) * ps_[i])
    adj[i] = min(1.0, run)
for k, a in zip(fam, adj):
    check(f"Holm adjusted p {k}", a, float(con[k]["p_holm_family_A"]))
# post-stratified holdout S1 calendar / R1 with the dev composition
levels = sorted({r["iv_quartile"] for r in exp})
dev_share = {l: sum(1 for r in dev if r["iv_quartile"] == l) / len(dev) for l in levels}
hq = defaultdict(list)
for r in hold:
    hq[r["iv_quartile"]].append(r)
w = {l: dev_share[l] / len(hq[l]) for l in levels if hq[l]}
vals = sorted((f(r, "spread_calendar"), w[r["iv_quartile"]]) for l in hq for r in hq[l])
tot = sum(x[1] for x in vals)
cum = 0.0
wm = None
for v_, w_ in vals:
    cum += w_
    if cum >= 0.5 * tot:
        wm = v_
        break
ps = {r["estimand"]: r for r in csv.DictReader(open(f"{G2}/t5_post_stratified_contrast.csv"))}
check("post-stratified holdout S1 calendar (dev IV-quartile composition)", wm, float(ps["S1_calendar_ps"]["holdout_reweighted_to_dev_composition"]))
num = sum(w[r["iv_quartile"]] * math.log(f(r, "implied_total_variance") / f(r, "rv_total_variance")) for l in hq for r in hq[l] if f(r, "rv_total_variance") > 0)
den = sum(w[r["iv_quartile"]] for l in hq for r in hq[l] if f(r, "rv_total_variance") > 0)
check("post-stratified holdout R1", math.exp(num / den), float(ps["R1_geometric_mean_ps"]["holdout_reweighted_to_dev_composition"]))
strad = defaultdict(set)
for r in exp:
    strad[r["expiry"]].add(r["split"])
n_str = sum(1 for v in strad.values() if len(v) > 1)
check("straddling expiries (rows in both splits)", float(n_str), float(len(list(csv.DictReader(open(f"{G2}/t5_straddling_expiries.csv"))))))

# ------------------------------------------------------------------------------ T6
print("\n[T6] strata")
het = list(csv.DictReader(open(f"{G2}/t6_heterogeneity_tests.csv")))
pb = [float(r["p_value"]) for r in het]
ordb = sorted(range(len(pb)), key=lambda i: pb[i])
runb, adjb = 0.0, [0.0] * len(pb)
for rank, i in enumerate(ordb):
    runb = max(runb, (len(pb) - rank) * pb[i])
    adjb[i] = min(1.0, runb)
check("Holm adjustment of the 12 family-B p-values: max |difference|", max(abs(a - float(r["p_holm_family_B"])) for a, r in zip(adjb, het)), 0.0)
tab = list(csv.DictReader(open(f"{G2}/t6_strata_estimates.csv")))
dims = {"dte_bucket": lambda r: r["dte_bucket"], "time": lambda r: r["time"], "iv_quartile": lambda r: r["iv_quartile"], "recent_rv_regime": lambda r: r["recent_rv_regime"],
        "expiry_year": lambda r: r["expiry"][:4], "weekend_in_window": lambda r: r["weekend_in_window"]}
maxdev = 0.0
n_checked = 0
for t in tab:
    if t["statistic"] not in ("S1_calendar", "R1_geometric_mean"):
        continue
    rs = [r for r in exp if str(dims[t["dimension"]](r)) == t["level"]]
    mine = s1cal(rs) if t["statistic"] == "S1_calendar" else r1(rs)
    maxdev = max(maxdev, abs(mine - float(t["estimate"])))
    assert len(rs) == int(t["n_rows"]), (t["dimension"], t["level"], len(rs), t["n_rows"])
    n_checked += 1
check(f"all {n_checked} stratum estimates (S1 calendar, R1) and row counts: max |difference|", maxdev, 0.0)
mz = [r for r in csv.DictReader(open(f"{G2}/t6_mincer_zarnowitz.csv")) if r["sample"] == "all" and r["scale"] == "level"][0]
x = [f(r, "iv_pct") for r in exp]
y = [f(r, "rv_calendar_pct") for r in exp]
xm, ym = sum(x) / len(x), sum(y) / len(y)
slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)
check("Mincer-Zarnowitz slope (all rows, level scale)", slope, float(mz["slope"]))
check("Mincer-Zarnowitz intercept", ym - slope * xm, float(mz["intercept"]))
fr = [r for r in rows if r["horizon"] == "F5" and r["measure"] == "hybrid"]
ed = {r["level"]: r for r in csv.DictReader(open(f"{G2}/t6_expiry_day_stratum.csv")) if r.get("horizon") == "F5" and r.get("statistic") == "S1_calendar" and r.get("level") in ("expiry day", "not expiry day")}
check("F5 expiry-day S1 calendar", median([f(r, "spread_calendar") for r in fr if r["expiry_day"] == "True"]), float(ed["expiry day"]["estimate"]))
check("F5 non-expiry-day S1 calendar", median([f(r, "spread_calendar") for r in fr if r["expiry_day"] == "False"]), float(ed["not expiry day"]["estimate"]))

# ------------------------------------------------------------------------------ T7 sampling from raw bars
print("\n[T7] 5- and 15-minute realized variance from the raw 1-minute bars (pure Python)")
df = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet")
ist = df.ts.to_numpy(dtype="int64") + 19800
dn, slot = ist // 86400, (ist % 86400) // 60 - 555
o_, h_, l_, c_ = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
ok_ = np.isfinite(o_) & np.isfinite(h_) & np.isfinite(l_) & np.isfinite(c_) & (o_ > 0) & (h_ >= l_) & (o_ >= l_) & (o_ <= h_) & (c_ >= l_) & (c_ <= h_)
win = (slot >= 0) & (slot < 375) & ok_
bars = {}
for d, s, o, c in zip(dn[win], slot[win], o_[win], c_[win]):
    bars.setdefault(int(d), {})[int(s)] = (float(o), float(c))
chain = [d for d in sorted(bars) if (d + 3) % 7 < 5 and len(bars[d]) >= 150]
pos = {d: i for i, d in enumerate(chain)}
complete = {d: len(bars[d]) == 375 for d in chain}
to_dn = lambda iso: (date.fromisoformat(iso) - date(1970, 1, 1)).days


def sess_var(d, prev_close, k):
    b = bars[d]
    P = [b[0][0]] + [b[i][1] for i in range(375)]
    g = P[::k]
    if g[-1] != P[-1]:
        g.append(P[-1])
    v = sum(math.log(g[i] / g[i - 1]) ** 2 for i in range(1, len(g)))
    return v + math.log(b[0][0] / prev_close) ** 2


def exp_var(day, hhmm, expiry, k):
    p, q = pos[day], pos[expiry]
    s0 = int(hhmm[:2]) * 60 + int(hhmm[3:]) - 555
    b = bars[day]
    idxs = list(range(s0, 374, k)) + [374]
    tot = sum(math.log(b[idxs[i + 1]][1] / b[idxs[i]][1]) ** 2 for i in range(len(idxs) - 1))
    for j in range(p + 1, q + 1):
        d = chain[j]
        tot += sess_var(d, bars[chain[j - 1]][374][1], k)
    return tot


curve = {}
for r in csv.DictReader(open(f"{G2}/t7_t8_specification_curve.csv")):
    if r["estimand"] in ("S1_calendar_all_obs_median", "R1_geometric_mean_all_obs") and r["estimate"] != "":
        curve[(r["spec"], r["estimand"])] = float(r["estimate"])
for k in (1, 5, 15):
    sc, lr = [], []
    for r in exp:
        d, e = to_dn(r["day"]), to_dn(r["expiry"])
        V = exp_var(d, r["time"], e, k)
        span = f(r, "span_days")
        iv = f(r, "iv_pct")
        sc.append(iv - 100 * math.sqrt(365 * V / span))
        lr.append(math.log(f(r, "implied_total_variance") / V))
    spec = "baseline" if k == 1 else f"rv_sampling_{k}min"
    check(f"{k}-minute sampling: S1 calendar (all {len(exp)} rows)", median(sc), curve[(spec, "S1_calendar_all_obs_median")], 1e-9)
    check(f"{k}-minute sampling: R1 geometric mean", math.exp(sum(lr) / len(lr)), curve[(spec, "R1_geometric_mean_all_obs")], 1e-9)

# ------------------------------------------------------------------------------ T7 rate: Black-76 re-solve
print("\n[T7] Black-76 implied volatility re-solved from stored price / forward / T")


def b76(F, K, T, r, sigma, kind):
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    return disc * (F * st.norm.cdf(d1) - K * st.norm.cdf(d2)) if kind == "call" else disc * (K * st.norm.cdf(-d2) - F * st.norm.cdf(-d1))


def solve(price, F, K, T, r, kind):
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if b76(F, K, T, r, mid, kind) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


for name, rate, path in (("baseline r=6.5%", 0.065, "research_output/stage2a/full/points.csv"), ("r=5.5% run", 0.055, f"{G2}/variants/rate_5.5/stage2a/points.csv"),
                         ("r=7.5% run", 0.075, f"{G2}/variants/rate_7.5/stage2a/points.csv")):
    if not os.path.exists(path):
        print(f"  (skipped {name}: {path} not present)")
        continue
    pts = pd.read_csv(path, usecols=["T_days", "forward", "strike", "kind", "price", "iv", "iv_status", "reliable", "resolution_limited", "used", "forward_status"])
    pts = pts[(pts.forward_status == "ok") & (pts.iv_status == "converged") & pts.reliable.astype(bool) & ~pts.resolution_limited.astype(bool) & pts.used.astype(bool) & (pts.T_days > 2)]
    smp = pts.sample(300, random_state=11)
    worst = 0.0
    for q in smp.itertuples():
        mine = solve(q.price, q.forward, q.strike, q.T_days / 365.0, rate, q.kind)
        worst = max(worst, abs(mine - q.iv))
    check(f"{name}: max |IV(independent) - IV(stored)| over 300 reliable OTM points (decimal vol)", worst, 0.0, 1e-6)

print("\nRESULT:", "ALL AGREE" if not bad else f"{bad} DISAGREEMENTS")
sys.exit(1 if bad else 0)
