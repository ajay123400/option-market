"""Independent recomputation of Stage 2D Gate 1 quantities (no optionsengine imports; csv + math + scipy only).

Recomputes from the Stage 2C aligned observations: the nine estimands, per-expiry series, ICC / design effect, ACF and Ljung-Box, Newey-West,
the exact sign test, all 260 leave-one-expiry-out estimates, delete-1 and delete-block jackknife SEs, the window-overlap summary and the
non-overlapping-subsample rule, and compares them with the Gate 1 output tables.
Usage: python research_output/stage2d/gate1/validation/independent_gate1_check.py [gate1_dir] [aligned_csv]
"""
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime

import scipy.stats as st

G1 = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2d/gate1"
AL = sys.argv[2] if len(sys.argv) > 2 else "research_output/stage2c/annualization_audit/aligned_observations.csv"
bad = 0


def check(name, mine, theirs, tol=1e-9):
    global bad
    ok = abs(mine - theirs) <= tol * max(1.0, abs(theirs))
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {name:62s} mine={mine:.10g} output={theirs:.10g}")


def median(x):
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


rows = []
with open(AL) as f:
    for r in csv.DictReader(f):
        if r["horizon"] == "EXP" and r["measure"] == "hybrid":
            rows.append(r)
print(f"EXP/hybrid rows: {len(rows)}")
by = defaultdict(list)
for r in rows:
    by[r["expiry"]].append(r)
exps = sorted(by)
f_ = lambda r, k: float(r[k])


def estimands(subset_exps):
    rs = [r for e in subset_exps for r in by[e]]
    ss = [f_(r, "spread_session") for r in rs]
    sc = [f_(r, "spread_calendar") for r in rs]
    ratio_rows = [r for r in rs if f_(r, "rv_total_variance") > 0]
    lr = [math.log(f_(r, "implied_total_variance") / f_(r, "rv_total_variance")) for r in ratio_rows]
    ra = [f_(r, "implied_total_variance") / f_(r, "rv_total_variance") for r in ratio_rows]
    per_s, per_c, per_lr, per_r = [], [], [], []
    for e in subset_exps:
        g = by[e]
        per_s.append(median([f_(r, "spread_session") for r in g]))
        per_c.append(median([f_(r, "spread_calendar") for r in g]))
        gv = [r for r in g if f_(r, "rv_total_variance") > 0]
        if gv:
            per_lr.append(sum(math.log(f_(r, "implied_total_variance") / f_(r, "rv_total_variance")) for r in gv) / len(gv))
            per_r.append(median([f_(r, "implied_total_variance") / f_(r, "rv_total_variance") for r in gv]))
    return [median(ss), median(per_s), median(sc), median(per_c), math.exp(sum(lr) / len(lr)), math.exp(sum(per_lr) / len(per_lr)), median(ra), median(per_r),
            sum(f_(r, "implied_total_variance") for r in rs) / sum(f_(r, "rv_total_variance") for r in rs)], per_s, per_c, per_lr, per_r


names = ["S1_session_all_obs_median", "S2_session_median_of_expiry_medians", "S1_calendar_all_obs_median", "S2_calendar_median_of_expiry_medians",
         "R1_geometric_mean_all_obs", "R1e_geometric_mean_expiry_weighted", "R2_median_ratio_all_obs", "R2e_median_of_expiry_median_ratios", "R3_ratio_of_summed_total_variance"]
theta, per_s, per_c, per_lr, per_r = estimands(exps)
out = {r["estimand"]: float(r["estimate"]) for r in csv.DictReader(open(f"{G1}/estimands_point.csv"))}
print("\n[1] estimands")
for n, v in zip(names, theta):
    check(n, v, out[n])

# ---------------------------------------------------------------------------- dependence
dd = {r["series"]: r for r in csv.DictReader(open(f"{G1}/dependence_diagnostics.csv"))}


def acvf(x, k):
    n = len(x)
    m = sum(x) / n
    return sum((x[t] - m) * (x[t + k] - m) for t in range(n - k)) / n


def acf(x, k):
    return acvf(x, k) / acvf(x, 0)


def icc(vals, groups):
    gs = defaultdict(list)
    for v, g in zip(vals, groups):
        gs[g].append(v)
    k, n = len(gs), len(vals)
    gm = sum(vals) / n
    ssb = sum(len(v) * (sum(v) / len(v) - gm) ** 2 for v in gs.values())
    ssw = sum((x - sum(v) / len(v)) ** 2 for v in gs.values() for x in v)
    n0 = (n - sum(len(v) ** 2 for v in gs.values()) / n) / (k - 1)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    c = (msb - msw) / (msb + (n0 - 1) * msw)
    return c, 1 + (n / k - 1) * max(c, 0.0)


def nw(x, L):
    n = len(x)
    lrv = acvf(x, 0) + 2 * sum((1 - k / (L + 1)) * acvf(x, k) for k in range(1, L + 1))
    return math.sqrt(lrv / n), lrv / acvf(x, 0)


print("\n[2] dependence diagnostics")
series = {"median_spread_session": ([f_(r, "spread_session") for e in exps for r in by[e]], per_s),
          "median_spread_calendar": ([f_(r, "spread_calendar") for e in exps for r in by[e]], per_c),
          "mean_log_ratio": ([math.log(f_(r, "implied_total_variance") / f_(r, "rv_total_variance")) for e in exps for r in by[e] if f_(r, "rv_total_variance") > 0], per_lr)}
grp_exp = [e for e in exps for r in by[e]]
for name, (rowvals, per) in series.items():
    d = dd[name]
    labels = [e for e in exps for r in by[e] if (name != "mean_log_ratio" or f_(r, "rv_total_variance") > 0)]
    c, de = icc(rowvals, labels)
    check(f"{name}: ICC by expiry", c, float(d["icc_by_expiry"]))
    check(f"{name}: design effect by expiry", de, float(d["design_effect_by_expiry"]))
    check(f"{name}: ACF lag 1", acf(per, 1), float(d["acf_lag1"]))
    n = len(per)
    q = n * (n + 2) * sum(acf(per, k) ** 2 / (n - k) for k in range(1, 11))
    check(f"{name}: Ljung-Box Q(10)", q, float(d["ljung_box_q10"]))
    check(f"{name}: Ljung-Box p(10) vs scipy chi2", st.chi2.sf(q, 10), float(d["ljung_box_p10"]), 1e-8)
    se, infl = nw(per, int(d["nw_lags_rule"]))
    check(f"{name}: Newey-West SE (rule lags)", se, float(d["nw_se_rule"]))
    check(f"{name}: Newey-West variance inflation", infl, float(d["nw_variance_inflation_rule"]))
    se8, _ = nw(per, 8)
    check(f"{name}: Newey-West SE (8 lags)", se8, float(d["nw_se_lag8"]))

# ---------------------------------------------------------------------------- ratio tests
print("\n[3] ratio inference")
ri = list(csv.DictReader(open(f"{G1}/ratio_inference.csv")))
k = sum(1 for v in per_r if v > 1)
n = sum(1 for v in per_r if v != 1)
check("sign test p (scipy binomtest, two-sided)", st.binomtest(k, n, 0.5).pvalue, float(ri[0]["p_two_sided"]), 1e-9)
check("share of expiries with median ratio > 1", k / n, float(ri[0]["statistic"]))
hac4 = [r for r in ri if r["test"] == "hac_t_mean_log_ratio"][0]
check("HAC t (rule lags)", (sum(per_lr) / len(per_lr)) / nw(per_lr, int(float(hac4["block"])))[0], float(hac4["statistic"]))

# ---------------------------------------------------------------------------- leave-one-out and jackknife
print("\n[4] leave-one-expiry-out (all 260) and jackknife")
loo_out = list(csv.DictReader(open(f"{G1}/influence_leave_one_expiry_out.csv")))
worst = 0.0
loo_vals = {n_: [] for n_ in names}
for row in loo_out:
    e = row["expiry"]
    th, *_ = estimands([x for x in exps if x != e])
    for n_, v in zip(names, th):
        worst = max(worst, abs(v - float(row[n_])))
        loo_vals[n_].append(v)
check("max |LOO difference| over 260 expiries x 9 estimands", worst, 0.0, 1e-9)
inf = {r["estimand"]: r for r in csv.DictReader(open(f"{G1}/influence_summary.csv"))}
for n_ in ("R1_geometric_mean_all_obs", "R3_ratio_of_summed_total_variance", "S1_calendar_all_obs_median"):
    v = loo_vals[n_]
    m = sum(v) / len(v)
    nn = len(v)
    check(f"jackknife SE {n_}", math.sqrt((nn - 1) / nn * sum((x - m) ** 2 for x in v)), float(inf[n_]["jackknife_se"]))
b = 5
reps = []
for j in range(len(exps) - b + 1):
    th, *_ = estimands([e for i, e in enumerate(exps) if not (j <= i < j + b)])
    reps.append(th[names.index("S2_calendar_median_of_expiry_medians")])
m = sum(reps) / len(reps)
check("block jackknife SE (b=5) S2_calendar", math.sqrt((len(exps) - b) / (b * len(reps)) * sum((x - m) ** 2 for x in reps)), float(inf["S2_calendar_median_of_expiry_medians"]["block_jackknife_se"]))

# ---------------------------------------------------------------------------- overlap and subsample
print("\n[5] window overlap and non-overlapping subsample")
iso = lambda s: datetime.fromisoformat(s).timestamp()
win = []
for e in exps:
    g = by[e]
    win.append((e, min(iso(r["target_start_ts"]) for r in g), max(iso(r["target_end_ts"]) for r in g)))
win.sort(key=lambda w: w[2])
shares = []
for i, (e, s, en) in enumerate(win):
    ov = max(0.0, min(en, win[i + 1][2]) - max(s, win[i + 1][1])) if i + 1 < len(win) else 0.0
    shares.append(ov / (en - s))
meta_ov = __import__("json").load(open(f"{G1}/run_metadata.json"))["window_overlap_summary"]
check("mean overlap share with the next expiry's window", sum(shares) / len(shares), meta_ov["mean_overlap_with_next_share"])
cand = {}
for e in exps:
    c = [r for r in by[e] if r["time"] == "10:00" and f_(r, "T_days") <= 6.0]
    if c:
        cand[e] = sorted(c, key=lambda r: (-f_(r, "T_days"), r["day"]))[0]
sel = sorted(cand.values(), key=lambda r: iso(r["target_end_ts"]))
kept, last = [], None
for r in sel:
    if last is not None and iso(r["target_start_ts"]) < last:
        continue
    kept.append(r)
    last = iso(r["target_end_ts"])
mine_sel = [(r["expiry"], r["day"]) for r in kept]
theirs = [(r["expiry"], r["day"]) for r in csv.DictReader(open(f"{G1}/nonoverlap_subsample_selection.csv"))]
bad += mine_sel != theirs
print(f"  {'ok ' if mine_sel == theirs else 'BAD'} subsample selection identical ({len(mine_sel)} expiries)")
sub = {r["estimand"]: float(r["estimate"]) for r in csv.DictReader(open(f"{G1}/nonoverlap_subsample_estimates.csv"))}
sub_rows = {r["expiry"]: [r] for r in kept}
saved = dict(by)
by.clear(); by.update(sub_rows)
th_sub, *_ = estimands(sorted(sub_rows))
for n_, v in zip(names, th_sub):
    check(f"subsample {n_}", v, sub[n_])
by.clear(); by.update(saved)
print("\nRESULT:", "ALL AGREE (tolerance 1e-9)" if not bad else f"{bad} DISAGREEMENTS")
sys.exit(1 if bad else 0)
