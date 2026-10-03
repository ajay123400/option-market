"""Independent recomputation of Stage 2D Gate 3a quantities (no optionsengine imports; csv/pandas/numpy/scipy only).

T9: universe layers and reasons from the raw Stage 2A smiles; the known-at-t covariates (previous-session range, |overnight gap|, strict hybrid ln RV20) and the outcome ln Y1
(F5 hybrid RV after the snapshot session) from the RAW 1-minute spot bars for a random sample of session dates; the primary ridge-logit outcome coefficient and the covariate-only
IPW estimates refit with scipy; Manski bounds and break-down shares.
T10: Black-76 re-solved with scipy brentq for two perturbed-price scenarios (all 6,490 rows), the tipping-bias medians, the ATM reconstruction and the used-quote age range.
Usage: python research_output/stage2d/gate3/validation/independent_gate3a_check.py [gate3_dir]
"""
import math
import random
import sys

import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.stats as st

G3 = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2d/gate3"
bad = 0


def check(name, mine, theirs, tol=1e-9):
    global bad
    ok = (isinstance(mine, float) and math.isnan(mine) and isinstance(theirs, float) and math.isnan(theirs)) or abs(mine - theirs) <= tol * max(1.0, abs(theirs))
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {name:78s} mine={mine:.10g} output={theirs:.10g}")


al = pd.read_csv("research_output/stage2c/annualization_audit/aligned_observations.csv")
e = al[(al.horizon == "EXP") & (al.measure == "hybrid")].reset_index(drop=True)
sm = pd.read_csv("research_output/stage2a/full/smiles.csv", usecols=["smile_id", "day", "time", "expiry", "expiry_day", "T_days", "forward_status", "group_reason", "atm_iv", "forward_used"])
lo = pd.read_csv("research_output/stage2c/iv_rv_observations.csv", usecols=["obs_id", "horizon", "measure", "target_available", "unavailable_reason"])
grp = pd.read_csv(f"{G3}/t9_groups.csv")

# ------------------------------------------------------------------------------ T9 universe
print("[T9] universe")
a14 = sm[sm.T_days <= 14]
kept = (a14.forward_status == "ok") & a14.atm_iv.notna()
obs = set(e.obs_id)
check("attempted groups <= 14 DTE", float(len(a14)), float(len(grp)))
nonexp = ~a14.expiry_day.astype(bool)
l1 = nonexp & ~kept
l2 = nonexp & kept & ~a14.smile_id.isin(obs)
exp_day = ~nonexp
for name, mine, layer in (("observed", float(a14.smile_id.isin(obs).sum()), "observed"), ("expiry-day by construction", float(exp_day.sum()), "expiry_day_by_construction"),
                          ("L1 rejected (non-expiry-day)", float(l1.sum()), "L1_rejected"), ("L2 target-unavailable", float(l2.sum()), "L2_no_exp_target")):
    check(f"layer count: {name}", mine, float((grp.layer == layer).sum()))
un = lo[(lo.horizon == "EXP") & (lo.measure == "hybrid") & (~lo.target_available)].set_index("obs_id").unavailable_reason
for reason, n in un[un.index.isin(a14[l2].smile_id)].value_counts().items():
    check(f"L2 reason {reason}", float(n), float((grp.l2_reason == reason).sum()))

# ------------------------------------------------------------------------------ T9 covariates and outcome from raw bars
print("\n[T9] known-at-t covariates and the outcome from the raw 1-minute bars (random sample of 50 dates)")
df = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet")
ist = df.ts.to_numpy(dtype="int64") + 19800
dn, slot = ist // 86400, (ist % 86400) // 60 - 555
o_, h_, l_, c_ = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
ok_ = np.isfinite(o_) & np.isfinite(h_) & np.isfinite(l_) & np.isfinite(c_) & (o_ > 0) & (h_ >= l_) & (o_ >= l_) & (o_ <= h_) & (c_ >= l_) & (c_ <= h_)
win = (slot >= 0) & (slot < 375) & ok_
bars = {}
for d, s, o, h, l, c in zip(dn[win], slot[win], o_[win], h_[win], l_[win], c_[win]):
    bars.setdefault(int(d), {})[int(s)] = (float(o), float(h), float(l), float(c))
chain = [d for d in sorted(bars) if (d + 3) % 7 < 5 and len(bars[d]) >= 150]
pos = {d: i for i, d in enumerate(chain)}
complete = {d: len(bars[d]) == 375 for d in chain}
to_dn = lambda iso: (pd.Timestamp(iso).date() - pd.Timestamp("1970-01-01").date()).days


def hybrid_var(d):
    j = pos[d]
    if j == 0 or not complete[d] or 374 not in bars[chain[j - 1]]:
        return None
    b = bars[d]
    v = math.log(b[0][3] / b[0][0]) ** 2 + sum(math.log(b[i][3] / b[i - 1][3]) ** 2 for i in range(1, 375))
    return v + math.log(b[0][0] / bars[chain[j - 1]][374][3]) ** 2


days = sorted(set(grp.day))
random.Random(5).shuffle(days)
mx = {"gap_abs": 0.0, "prev_range": 0.0, "ln_rv20": 0.0, "ln_y1": 0.0}
n_cmp = {k: 0 for k in mx}
for day in days[:50]:
    d = to_dn(day)
    if d not in pos:
        continue
    row = grp[grp.day == day].iloc[0]
    j = pos[d]
    if j > 0:
        pb = bars[chain[j - 1]]
        mine_gap = abs(math.log(bars[d][0][0] / pb[374][3])) if 374 in pb and 0 in bars[d] else None
        if mine_gap is not None and np.isfinite(row.gap_abs):
            mx["gap_abs"] = max(mx["gap_abs"], abs(mine_gap - row.gap_abs)); n_cmp["gap_abs"] += 1
        mine_rng = math.log(max(v[1] for v in pb.values()) / min(v[2] for v in pb.values()))
        if np.isfinite(row.prev_range):
            mx["prev_range"] = max(mx["prev_range"], abs(mine_rng - row.prev_range)); n_cmp["prev_range"] += 1
        win20 = [hybrid_var(chain[k]) for k in range(j - 20, j)] if j >= 20 else [None]
        if all(x is not None for x in win20) and np.isfinite(row.ln_rv20):
            mine = math.log(100 * math.sqrt(252 * sum(win20) / 20))
            mx["ln_rv20"] = max(mx["ln_rv20"], abs(mine - row.ln_rv20)); n_cmp["ln_rv20"] += 1
        elif np.isfinite(row.ln_rv20) and any(x is None for x in win20):
            bad += 1; print("  BAD: output has ln_rv20 where the strict window is incomplete", day)
    nxt = [hybrid_var(chain[k]) for k in range(j + 1, j + 6)] if j + 5 < len(chain) else [None]
    if all(x is not None for x in nxt) and np.isfinite(row.ln_y1):
        mine = math.log(100 * math.sqrt(252 * sum(nxt) / 5))
        mx["ln_y1"] = max(mx["ln_y1"], abs(mine - row.ln_y1)); n_cmp["ln_y1"] += 1
for k in mx:
    check(f"{k}: max |difference| over {n_cmp[k]} sampled dates", mx[k], 0.0, 1e-8)

# ------------------------------------------------------------------------------ T9 ridge logit refit and IPW
print("\n[T9] ridge-logit outcome coefficient and IPW refit with scipy")
u = grp[grp.layer != "expiry_day_by_construction"].copy()
u["missing"] = (u.layer != "observed").astype(float)
series_days = set(grp.day[np.isfinite(grp.ln_y1)])


def design(g, outcome):
    cols = [np.ones(len(g))]
    cols += [(g.year == y).to_numpy(float) for y in (2022, 2023, 2024, 2025, 2026)]
    cols += [(g.time == "13:00").to_numpy(float), (g.time == "15:00").to_numpy(float)]
    cols += [(g.wd == k).to_numpy(float) for k in (1, 2, 3, 4)]
    cols += [g.T_days.to_numpy(float), g.ln_rv20.to_numpy(float), g.prev_range.to_numpy(float), g.gap_abs.to_numpy(float)]
    if outcome:
        cols.append(g.ln_y1.to_numpy(float))
    X = np.column_stack(cols)
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / np.where(X[:, 1:].std(axis=0) > 0, X[:, 1:].std(axis=0), 1.0)
    return X


def ridge(X, y, lam=1.0):
    pen = np.full(X.shape[1], lam); pen[0] = 0.0

    def f(b):
        eta = X @ b
        return -(y @ eta - np.sum(np.logaddexp(0, eta))) + 0.5 * np.sum(pen * b * b)

    def gr(b):
        mu = 1 / (1 + np.exp(-(X @ b)))
        return -(X.T @ (y - mu)) + pen * b

    return so.minimize(f, np.zeros(X.shape[1]), jac=gr, method="BFGS", options=dict(gtol=1e-9, maxiter=500)).x


cc = u[np.isfinite(u[["ln_rv20", "prev_range", "gap_abs", "ln_y1"]].to_numpy(float)).all(axis=1) & u.day.isin(series_days)]
b = ridge(design(cc, True), cc.missing.to_numpy(float))
t = pd.read_csv(f"{G3}/t9_outcome_dependence_test.csv").iloc[0]
check("primary outcome coefficient (standardized, ridge lambda=1)", float(b[-1]), float(t.outcome_coefficient_standardized), 1e-5)
check("primary model: complete-case groups", float(len(cc)), float(t.n_groups))
check("primary model: flagged (missing) groups", float(cc.missing.sum()), float(t.n_flagged))
cc2 = u[np.isfinite(u[["ln_rv20", "prev_range", "gap_abs"]].to_numpy(float)).all(axis=1)].copy()
p = 1 / (1 + np.exp(-(design(cc2, False) @ ridge(design(cc2, False), cc2.missing.to_numpy(float)))))
w = np.minimum(1 / np.clip(1 - p, 1e-6, 1), 10.0)          # observed = 1 - missing: P(observed | X) = 1 - P(missing | X)
obs_rows = cc2[cc2.layer == "observed"]
av = e.set_index("obs_id")
sc = av.loc[obs_rows.smile_id, "spread_calendar"].to_numpy(float)
ss = av.loc[obs_rows.smile_id, "spread_session"].to_numpy(float)
lr = np.log(av.loc[obs_rows.smile_id, "total_variance_ratio"].to_numpy(float))
wo = w[(cc2.layer == "observed").to_numpy()]


def wmed(x, ww):
    o = np.argsort(x, kind="stable"); cw = np.cumsum(ww[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


ipw = pd.read_csv(f"{G3}/t9_ipw_estimates.csv").set_index("estimand")
check("IPW S1 calendar", wmed(sc, wo), float(ipw.loc["S1_calendar", "ipw"]), 1e-5)
check("IPW S1 session", wmed(ss, wo), float(ipw.loc["S1_session", "ipw"]), 1e-5)
check("IPW R1 geometric mean", math.exp(float(np.sum(wo * lr) / np.sum(wo))), float(ipw.loc["R1_geometric_mean", "ipw"]), 1e-5)
check("unweighted complete-case S1 calendar", float(np.median(sc)), float(ipw.loc["S1_calendar", "unweighted_complete_case"]))

# ------------------------------------------------------------------------------ Manski
print("\n[T9] Manski bounds and break-down shares")
mb = pd.read_csv(f"{G3}/t9_manski_bounds.csv")


def med_with(v, k, fill):
    return float(np.median(np.concatenate([v, np.full(k, fill)])))


ratio = e.total_variance_ratio[e.total_variance_ratio_valid].to_numpy(float)
for est, vals, null in (("S1_calendar spread (null 0)", e.spread_calendar.to_numpy(float), 0.0), ("S1_session spread (null 0)", e.spread_session.to_numpy(float), 0.0), ("R2 median ratio (null 1)", ratio, 1.0)):
    for scope, k in (("all missing (L1+L2)", 188), ("L1 rejections only", 73), ("L2 target losses only", 115)):
        r = mb[(mb.estimand == est) & (mb.missing_scope == scope)].iloc[0]
        check(f"{est} | {scope}: lower bound", med_with(vals, k, -np.inf), float(r.manski_lower))
        check(f"{est} | {scope}: upper bound", med_with(vals, k, np.inf), float(r.manski_upper))
    n0 = int((vals <= null).sum()); n = len(vals)
    kstar = n - 2 * n0                                       # (n0 + k)/(n + k) >= 1/2
    r = mb[(mb.estimand == est)].iloc[0]
    check(f"{est}: break-down k, |closed form (n - 2 n0) - output| (tolerance 1)", float(kstar) - float(r.breakdown_k_needed), 0.0, 1.0)

# ------------------------------------------------------------------------------ T10
print("\n[T10] Black-76 scenarios re-solved with scipy brentq (all rows)")
pts = pd.read_csv("research_output/stage2a/full/points.csv", usecols=["smile_id", "strike", "kind", "price", "age_min", "log_moneyness", "iv", "used", "forward", "T_days"])
check("used points: minimum quote age (minutes)", float(pts[pts.used].age_min.min()), float(pd.read_csv(f"{G3}/t9_strike_universe_check.csv").min_age_min.iloc[0]))
check("used points: maximum quote age (minutes)", float(pts[pts.used].age_min.max()), float(pd.read_csv(f"{G3}/t9_strike_universe_check.csv").max_age_min.iloc[0]))
p = pts[pts.smile_id.isin(set(e.obs_id))]
lo_ = p[p.log_moneyness < 0].sort_values(["smile_id", "strike"]).groupby("smile_id").tail(1).set_index("smile_id")
hi_ = p[p.log_moneyness >= 0].sort_values(["smile_id", "strike"]).groupby("smile_id").head(1).set_index("smile_id")
ids = e.obs_id.to_numpy()
L, H = lo_.loc[ids], hi_.loc[ids]
F, T = L.forward.to_numpy(float), L.T_days.to_numpy(float) / 365.0
R = 0.065


def b76(Fv, K, Tv, sig, call):
    sd = sig * math.sqrt(Tv); d1 = (math.log(Fv / K) + 0.5 * sd * sd) / sd; d2 = d1 - sd
    disc = math.exp(-R * Tv)
    return disc * (Fv * st.norm.cdf(d1) - K * st.norm.cdf(d2)) if call else disc * (K * st.norm.cdf(-d2) - Fv * st.norm.cdf(-d1))


def solve(price, Fv, K, Tv, call):
    try:
        return so.brentq(lambda s: b76(Fv, K, Tv, s, call) - price, 1e-4, 5.0, xtol=1e-13, rtol=1e-13)
    except ValueError:
        return float("nan")


def scenario(kind, par):
    ivs = []
    for i in range(len(ids)):
        out = []
        for P, call in ((L, L.kind.iloc[i] == "call"), (H, H.kind.iloc[i] == "call")):
            pr = P.price.iloc[i]
            pr = max(pr * (1 + par), 0.05) if kind == "rel" else max(pr + par, 0.05)
            out.append(solve(pr, F[i], P.strike.iloc[i], T[i], call))
        x0, x1 = L.log_moneyness.iloc[i], H.log_moneyness.iloc[i]
        w = (0 - x0) / (x1 - x0)
        ivs.append(out[0] + w * (out[1] - out[0]))
    return np.array(ivs)


sc_tab = pd.read_csv(f"{G3}/t10_price_bias_scenarios.csv")
for kind, par, name in (("abs", 5.0, "abs +5"), ("rel", -0.02, "rel -0.02")):
    iv = scenario(kind, par)
    okm = np.isfinite(iv)
    spread = 100 * iv - e.rv_calendar_pct.to_numpy(float)
    r1 = math.exp(float(np.mean(np.log((iv[okm] ** 2 * T[okm]) / e.rv_total_variance.to_numpy(float)[okm]))))
    out = sc_tab[sc_tab.scenario == name].set_index("estimand")
    check(f"{name}: S1 calendar (independent brentq, {int(okm.sum())} rows)", float(np.median(spread[okm])), float(out.loc["S1_calendar_all_obs_median", "estimate"]), 1e-7)
    check(f"{name}: R1 geometric mean", r1, float(out.loc["R1_geometric_mean_all_obs", "estimate"]), 1e-8)
atm_stored = sm.set_index("smile_id").atm_iv.loc[ids].to_numpy(float)
iv0 = scenario("abs", 0.0)
check("zero-shift ATM IV vs stored Stage 2A ATM IV: max |difference|", float(np.nanmax(np.abs(iv0 - atm_stored))), 0.0, 1e-9)


def vega(Fv, K, Tv, sig):
    sd = sig * math.sqrt(Tv); d1 = (math.log(Fv / K) + 0.5 * sd * sd) / sd
    return math.exp(-R * Tv) * Fv * st.norm.pdf(d1) * math.sqrt(Tv) / 100.0


w_ = (0 - L.log_moneyness.to_numpy(float)) / (H.log_moneyness.to_numpy(float) - L.log_moneyness.to_numpy(float))
v = np.array([(1 - w_[i]) * vega(F[i], L.strike.iloc[i], T[i], L.iv.iloc[i]) + w_[i] * vega(F[i], H.strike.iloc[i], T[i], H.iv.iloc[i]) for i in range(len(ids))])
tip = pd.read_csv(f"{G3}/t10_tipping_bias.csv").iloc[0]
s1 = float(np.median(e.spread_calendar))
check("tipping bias: median price bias (Rs) = median(S1 * vega)", float(np.median(s1 * v)), float(tip.price_bias_rs_median), 1e-7)
price = (1 - w_) * L.price.to_numpy(float) + w_ * H.price.to_numpy(float)
check("tipping bias: median % of bracketing price", float(np.median(s1 * v / price) * 100), float(tip.price_bias_pct_of_price_median), 1e-7)
r1b = math.exp(float(np.log(e.total_variance_ratio[e.total_variance_ratio_valid]).mean()))
check("IV factor for a unit geometric-mean ratio = 1/sqrt(R1)", 1 / math.sqrt(r1b), float(tip.iv_factor_for_unit_ratio))
print("\nRESULT:", "ALL AGREE" if not bad else f"{bad} DISAGREEMENTS")
sys.exit(1 if bad else 0)
