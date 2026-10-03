"""Stage 2D Gate 3a builder (T9 selection audit, T10 quote-noise sensitivity). Writes ONLY under a `stage2d` directory.

python -m optionsengine.research.stage2d.build_gate3a --out research_output/stage2d/gate3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import loaders
from . import build_gate1 as g1, clusters as C, estimands as E, quote_noise as Q, selection as SEL, spec_curve as SC, strata as ST

SEED = 20260101
BLOCK = 5
BAL_VARS = ["T_days", "ln_rv20", "prev_range", "gap_abs", "year", "wd", "ln_y1"]
ABS_SHIFTS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
REL_SHIFTS = [0.005, 0.01, 0.02, 0.05, 0.10]
OPP_SHIFTS = [0.5, 1.0, 2.0, 5.0]
NOISE_ABS = [0.5, 1.0, 2.0, 5.0]
NOISE_REL = [0.005, 0.01, 0.02]
MC_DRAWS = 200
HEADLINE = ["S1_session_all_obs_median", "S1_calendar_all_obs_median", "R1_geometric_mean_all_obs", "R2_median_ratio_all_obs", "R3_ratio_of_summed_total_variance"]


def _write(df: pd.DataFrame, out: str, name: str) -> None:
    df.to_csv(os.path.join(out, name), index=False)


# ------------------------------------------------------------------------------ T9
def universe_counts(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, col in (("all", None), ("year", "year"), ("time", "time"), ("dte_bucket", "dte_bucket")):
        gg = g.assign(all="all", dte_bucket=g.T_days.map(lambda t: "<=1d" if t <= 1 else ("1-3d" if t <= 3 else ("3-7d" if t <= 7 else "7-14d"))))
        for lvl, d in gg.groupby(col or "all"):
            vc = d.layer.value_counts()
            rej = d.layer.isin(["L1_rejected"]).sum()
            rows.append(dict(dimension=dim, level=str(lvl), attempted_groups=len(d), observed=int(vc.get("observed", 0)), expiry_day_by_construction=int(vc.get("expiry_day_by_construction", 0)),
                             L1_rejected=int(rej), L2_no_exp_target=int(vc.get("L2_no_exp_target", 0)), share_L1_rejected_of_non_expiry_day=float(rej / max(1, (d.layer != "expiry_day_by_construction").sum())),
                             share_missing_of_u_exp=float((d.layer.isin(["L1_rejected", "L2_no_exp_target"])).sum() / max(1, (d.layer != "expiry_day_by_construction").sum()))))
    reasons = g[g.layer.isin(["L1_rejected", "L2_no_exp_target"])].assign(reason=lambda d: np.where(d.layer == "L1_rejected", d.l1_reason, d.l2_reason))
    for (layer, reason), d in reasons.groupby(["layer", "reason"]):
        rows.append(dict(dimension="reason", level=f"{layer}: {reason}", attempted_groups=len(d), observed=0, expiry_day_by_construction=0, L1_rejected=len(d) if layer == "L1_rejected" else 0,
                         L2_no_exp_target=len(d) if layer == "L2_no_exp_target" else 0, share_L1_rejected_of_non_expiry_day=np.nan, share_missing_of_u_exp=np.nan))
    return pd.DataFrame(rows)


def run_t9(smiles: pd.DataFrame, feats: pd.DataFrame, aligned_all: pd.DataFrame, exp_unavail: Dict[str, str], points_used_ages: pd.DataFrame, out: str,
           perm_draws: int, boot_reps: int) -> dict:
    e = aligned_all[(aligned_all.horizon == "EXP") & (aligned_all.measure == "hybrid")].reset_index(drop=True)
    g = SEL.build_groups(smiles, feats, e.obs_id, exp_unavail)
    g["time_1300"] = (g.time == "13:00").astype(float)
    g["time_1500"] = (g.time == "15:00").astype(float)
    _write(universe_counts(g), out, "t9_universe_counts.csv")
    _write(g[["smile_id", "day", "time", "expiry", "T_days", "year", "wd", "layer", "l1_reason", "l2_reason", "ln_rv20", "prev_range", "gap_abs", "ln_y1"]], out, "t9_groups.csv")
    uexp = g[g.in_u_exp].reset_index(drop=True)
    uexp["missing"] = (uexp.layer != "observed").astype(float)
    n_obs, n_l1, n_l2 = int((uexp.layer == "observed").sum()), int((uexp.layer == "L1_rejected").sum()), int((uexp.layer == "L2_no_exp_target").sum())
    # -------- balance (descriptive; the outcome ln_y1 appears here only as an audit variable)
    bal = []
    for name, mask_flag, mask_pop in (("missing (L1+L2) vs observed", uexp.missing.to_numpy() > 0.5, np.ones(len(uexp), bool)),
                                      ("L1 rejected vs observed", (uexp.layer == "L1_rejected").to_numpy(), (uexp.layer != "L2_no_exp_target").to_numpy()),
                                      ("L2 target-unavailable vs observed", (uexp.layer == "L2_no_exp_target").to_numpy(), (uexp.layer != "L1_rejected").to_numpy())):
        sub = uexp[mask_pop].reset_index(drop=True)
        t = SEL.balance_table(sub, mask_flag[mask_pop], BAL_VARS, BLOCK, boot_reps, SEED + 1)
        t.insert(0, "contrast", name)
        bal.append(t)
    # L2 audit extras: IV level and the F5 calendar spread (outcome, descriptive only)
    f5 = aligned_all[(aligned_all.horizon == "F5") & (aligned_all.measure == "hybrid")].set_index("obs_id")
    uexp["iv_pct"] = uexp.atm_iv * 100.0
    uexp["f5_spread_calendar"] = uexp.smile_id.map(f5.spread_calendar)
    sub = uexp[uexp.layer != "L1_rejected"].reset_index(drop=True)
    t = SEL.balance_table(sub, (sub.layer == "L2_no_exp_target").to_numpy(), ["iv_pct", "f5_spread_calendar"], BLOCK, boot_reps, SEED + 2)
    t.insert(0, "contrast", "L2 target-unavailable vs observed (IV level, F5 calendar spread; descriptive)")
    bal.append(t)
    _write(pd.concat(bal, ignore_index=True), out, "t9_balance.csv")
    # -------- inclusion model and outcome-dependence test
    series_days = list(feats.day[np.isfinite(feats.ln_y1)])
    series = feats.ln_y1[np.isfinite(feats.ln_y1)].to_numpy()
    results = []
    models = []
    for label, pop_mask, flag in (("primary: missing (L1+L2) vs observed in U_exp", uexp.layer.notna(), uexp.missing),
                                  ("secondary: L1 rejected vs observed (L2 groups excluded)", uexp.layer != "L2_no_exp_target", (uexp.layer == "L1_rejected").astype(float))):
        pop = uexp[pop_mask].reset_index(drop=True)
        fl = flag[pop_mask].reset_index(drop=True).to_numpy(float)
        cc = SEL.complete_cases(pop, need_outcome=True)
        cc_fl = fl[np.isfinite(pop[["ln_rv20", "prev_range", "gap_abs", SEL.OUTCOME_COLUMN]].to_numpy(float)).all(axis=1)]
        ccg = cc[cc.day.isin(set(series_days))].reset_index(drop=True)
        cc_fl = cc_fl[cc.day.isin(set(series_days)).to_numpy()]
        X, names = SEL.design_matrix(ccg, with_outcome=True)
        Z = SEL.standardize(X)
        beta = SEL.ridge_logit(Z, cc_fl)
        perm = SEL.date_shift_permutation_test(ccg, cc_fl, series_days, series, perm_draws, 20, 1.0, SEED + 3)
        cs = C.ClusterSet(ccg.expiry.to_numpy(), {"X": X, "m": cc_fl})

        def ev(rows, seq, cs=cs):
            Zr = SEL.standardize(cs.cols["X"][rows])
            return np.array([SEL.ridge_logit(Zr, cs.cols["m"][rows])[-1]])

        dr = C.bootstrap(cs, ev, "moving_block", BLOCK, boot_reps, SEED + 4, k=1)
        lo, hi = C.percentile_ci(dr)
        results.append(dict(model=label, n_groups=len(ccg), n_flagged=int(cc_fl.sum()), n_dropped_incomplete_covariates=int(len(pop) - len(ccg)), outcome_coefficient_standardized=perm["coefficient"],
                            ci_lo=lo[0], ci_hi=hi[0], p_permutation_two_sided=perm["p_two_sided"], permutation_null_sd=perm["null_sd"], permutation_draws=perm["draws"],
                            ci_method=f"expiry moving-block bootstrap b={BLOCK}, {boot_reps} refits", null_method="circular shift of the date-level ln Y1 series (offset >= 20 sessions)"))
        for nme, b in zip(names, beta):
            models.append(dict(model=label, term=nme, standardized_coefficient=float(b)))
    _write(pd.DataFrame(results), out, "t9_outcome_dependence_test.csv")
    _write(pd.DataFrame(models), out, "t9_inclusion_model_coefficients.csv")
    # -------- IPW (covariate-only model, no outcome in the model)
    cc = SEL.complete_cases(uexp, need_outcome=False)
    ob = aligned_all[(aligned_all.horizon == "EXP") & (aligned_all.measure == "hybrid")].set_index("obs_id")
    valid = ob.total_variance_ratio_valid.to_numpy(bool)
    cols = E.cluster_columns(ob.reset_index())
    lookup = pd.DataFrame(dict(sc=cols["sc"], ss=cols["ss"], lr=cols["loglr"], ratio=cols["ratio"]), index=ob.index)
    cc = cc.join(lookup, on="smile_id")
    X, names = SEL.design_matrix(cc, with_outcome=False)
    cs = C.ClusterSet(cc.expiry.to_numpy(), {"X": X, "observed": (cc.layer == "observed").to_numpy(float), "sc": cc.sc.fillna(0).to_numpy(float), "ss": cc.ss.fillna(0).to_numpy(float),
                                              "lr": cc.lr.fillna(0).to_numpy(float), "ratio": cc.ratio.fillna(1).to_numpy(float)})
    ev = SEL.IPWEvaluator(cs)
    pt = ev.point()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dr = C.bootstrap(cs, ev.evaluate, "moving_block", BLOCK, boot_reps, SEED + 5, k=2 * len(SEL.IPW_NAMES))
        lo, hi = C.percentile_ci(dr)
    w, p = ev.weights(np.arange(cs.n_rows))
    rows = []
    for i, n in enumerate(SEL.IPW_NAMES):
        rows.append(dict(estimand=n, unweighted_complete_case=pt[4 + i], ipw=pt[i], difference=pt[i] - pt[4 + i], ipw_ci_lo=lo[i], ipw_ci_hi=hi[i], unweighted_ci_lo=lo[4 + i], unweighted_ci_hi=hi[4 + i],
                         n_groups=cs.n_rows, n_observed=int(cs.cols["observed"].sum()), n_dropped_incomplete_covariates=int(len(uexp) - len(cc)), weights_capped=int((1.0 / np.clip(p, 1e-6, 1) > SEL.WEIGHT_CAP).sum()),
                         weight_min=float(w.min()), weight_max=float(w.max()), baseline_all_observed_rows=np.nan))
    base = {"S1_calendar": e.spread_calendar.median(), "S1_session": e.spread_session.median(), "R2_median_ratio": float(np.median(e.total_variance_ratio[e.total_variance_ratio_valid]))}
    base["R1_geometric_mean"] = float(math.exp(np.log(e.total_variance_ratio[e.total_variance_ratio_valid]).mean()))
    for r in rows:
        r["baseline_all_observed_rows"] = base[r["estimand"]]
    _write(pd.DataFrame(rows), out, "t9_ipw_estimates.csv")
    # -------- Manski bounds / break-down
    rb = []
    k_all, k_l1, k_l2 = n_l1 + n_l2, n_l1, n_l2
    ratio = e.total_variance_ratio[e.total_variance_ratio_valid].to_numpy(float)
    for est, vals, null, direction in (("S1_calendar spread (null 0)", e.spread_calendar.to_numpy(float), 0.0, "above"), ("S1_session spread (null 0)", e.spread_session.to_numpy(float), 0.0, "above"),
                                       ("R2 median ratio (null 1)", ratio, 1.0, "above")):
        bd = SEL.breakdown_share(vals, null, direction)
        for scope, k in (("all missing (L1+L2)", k_all), ("L1 rejections only", k_l1), ("L2 target losses only", k_l2)):
            lo_b, hi_b = SEL.manski_median_bounds(vals, k)
            rb.append(dict(estimand=est, missing_scope=scope, n_observed=len(vals), n_missing=k, share_missing=k / (len(vals) + k), observed_median=bd["observed_median"], manski_lower=lo_b, manski_upper=hi_b,
                           breakdown_k_needed=bd["k_needed"], breakdown_share_needed=bd["share_needed"], interpretation="missing groups would have to be at least this share of the population, all at the extreme, to move the median to the null"))
    mean_lnr = float(np.log(ratio).mean())
    rb.append(dict(estimand="R1 geometric mean (null 1)", missing_scope="all missing (L1+L2)", n_observed=len(ratio), n_missing=k_all, share_missing=k_all / (len(ratio) + k_all), observed_median=math.exp(mean_lnr),
                   manski_lower=np.nan, manski_upper=np.nan, breakdown_k_needed=np.nan, breakdown_share_needed=np.nan,
                   interpretation=f"R1 is unbounded under worst-case values; the {k_all} missing groups would need a mean ln(ratio) of {SEL.required_missing_mean_log_ratio(mean_lnr, len(ratio), k_all):.3f} (ratio {math.exp(SEL.required_missing_mean_log_ratio(mean_lnr, len(ratio), k_all)):.2e}) for R1 = 1"))
    _write(pd.DataFrame(rb), out, "t9_manski_bounds.csv")
    # -------- layer 3: strike age check and the 14-30 DTE diagnostic (composition only)
    ages = points_used_ages
    _write(pd.DataFrame([dict(check="all USED points: quote age in [0, 5] minutes (no quote from after t, none older than the 5-minute freshness gate)", n_used_points=len(ages),
                              min_age_min=float(ages.age_min.min()), max_age_min=float(ages.age_min.max()), n_negative_age=int((ages.age_min < 0).sum()), n_older_than_5=int((ages.age_min > 5).sum()),
                              passes=bool((ages.age_min >= 0).all() and (ages.age_min <= 5).all()))]), out, "t9_strike_universe_check.csv")
    wide = SEL.build_groups(smiles, feats, [], {}, dte_max=30.0)
    wide["bucket"] = np.where(wide.T_days <= 14, "<=14 DTE (primary universe)", "14-30 DTE (DIAGNOSTIC ONLY)")
    diag = []
    for b, d in wide.groupby("bucket"):
        diag.append(dict(population_bucket=b, note="diagnostic / population-composition context only: known-at-t composition; no outcomes, spreads or ratios; no extrapolation to the primary universe",
                         attempted_groups=len(d), share_forward_ok=float((d.forward_status == "ok").mean()), share_strict_atm=float(d.atm_iv.notna().mean()), mean_T_days=float(d.T_days.mean()),
                         share_year_2021_2022=float(d.year.isin([2021, 2022]).mean()), share_time_1000=float((d.time == "10:00").mean()), share_time_1300=float((d.time == "13:00").mean()),
                         share_time_1500=float((d.time == "15:00").mean()), mean_ln_rv20=float(d.ln_rv20.mean()), share_with_rv20=float(d.ln_rv20.notna().mean())))
    _write(pd.DataFrame(diag), out, "t9_dte_composition_diagnostic.csv")
    return dict(n_obs=n_obs, n_l1=n_l1, n_l2=n_l2, n_groups=len(g))


# ------------------------------------------------------------------------------ T10
def run_t10(aligned_all: pd.DataFrame, smiles: pd.DataFrame, points: pd.DataFrame, root: str, out: str, boot_reps: int, mc_draws: int, cp_expiries_per_year: int = 10,
            cp_snaps_per_expiry: int = 5) -> dict:
    e = aligned_all[(aligned_all.horizon == "EXP") & (aligned_all.measure == "hybrid")].reset_index(drop=True)
    rd = Q.RowData(e, Q.bracket_table(points, e.obs_id))
    atm_stored = smiles.set_index("smile_id").atm_iv.loc[rd.base.obs_id].to_numpy(float)
    iv0 = rd.atm_iv(rd.p_lo, rd.p_hi)
    ivl, ivh = rd.solve(rd.p_lo, rd.p_hi)
    _write(pd.DataFrame([dict(rows=rd.n, rows_with_both_bracketing_points_used=int(rd.used.sum()), max_abs_interpolation_diff_stored_points_vs_stored_atm_iv=float(np.nanmax(np.abs(rd.atm_stored_points() - atm_stored))),
                              max_abs_resolved_bracket_iv_diff_vs_stored=float(max(np.nanmax(np.abs(ivl - rd.iv_lo_stored)), np.nanmax(np.abs(ivh - rd.iv_hi_stored)))),
                              max_abs_resolved_atm_iv_diff_vs_stored=float(np.nanmax(np.abs(iv0 - atm_stored))), n_unsolved=int(np.isnan(iv0).sum()),
                              note="independent Black-76 solver (r=6.5%) reproduces the stored Stage 2A IVs")]), out, "t10_atm_reconstruction.csv")
    base_spec = SC.Spec("baseline_reconstructed", "baseline", "0", rd.frame(iv0))
    cur = [SC.estimate_spec(base_spec, BLOCK, boot_reps, SEED + 10).assign(scenario="baseline (zero shift, reconstructed)", kind="none", parameter=0.0, n_lost=0)]
    scen: List[tuple] = [("abs", v) for v in ABS_SHIFTS[1:]] + [("abs", -v) for v in ABS_SHIFTS[1:]] + [("rel", v) for v in REL_SHIFTS] + [("rel", -v) for v in REL_SHIFTS] + [("opposite", v) for v in OPP_SHIFTS]
    for j, (kind, par) in enumerate(scen):
        pl, ph = Q.scenario_prices(rd, kind, par)
        iv = rd.atm_iv(pl, ph)
        sp = SC.Spec(f"{kind}_{par:+g}", kind, f"{par:+g}", rd.frame(iv))
        cur.append(SC.estimate_spec(sp, BLOCK, boot_reps, SEED + 11 + j).assign(scenario=f"{kind} {par:+g}", kind=kind, parameter=par, n_lost=int(np.isnan(iv).sum())))
    curve = pd.concat(cur, ignore_index=True)
    curve["label"] = "ASSUMPTION / BOUND (no bid/ask observed)"
    _write(curve.drop(columns=["spec", "dimension", "value", "note"]), out, "t10_price_bias_scenarios.csv")
    # zero-mean noise: Monte Carlo
    mc = []
    rng = np.random.default_rng(SEED + 99)
    for kind, pars in (("noise_abs", NOISE_ABS), ("noise_rel", NOISE_REL)):
        for par in pars:
            res, slopes, lost = [], [], []
            for _ in range(mc_draws):
                pl, ph = Q.scenario_prices(rd, kind, par, rng)
                iv = rd.atm_iv(pl, ph)
                f = rd.frame(iv)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    res.append(E.build(f).point())
                slopes.append(ST.ols(f.iv_pct.to_numpy(float), f.rv_calendar_pct.to_numpy(float))[1])
                lost.append(int(np.isnan(iv).sum()))
            res = np.array(res)
            for i, n in enumerate(E.EST_NAMES):
                if n in HEADLINE:
                    mc.append(dict(scenario=f"{kind} {par:g}", kind=kind, parameter=par, estimand=n, mean=float(res[:, i].mean()), sd=float(res[:, i].std(ddof=1)), q025=float(np.percentile(res[:, i], 2.5)),
                                   q975=float(np.percentile(res[:, i], 97.5)), draws=mc_draws, mean_rows_lost=float(np.mean(lost)), label="ASSUMPTION (zero-mean noise on the two bracketing prices; no bid/ask observed)"))
            mc.append(dict(scenario=f"{kind} {par:g}", kind=kind, parameter=par, estimand="MZ_slope_RVcal_on_IV", mean=float(np.mean(slopes)), sd=float(np.std(slopes, ddof=1)), q025=float(np.percentile(slopes, 2.5)),
                           q975=float(np.percentile(slopes, 97.5)), draws=mc_draws, mean_rows_lost=float(np.mean(lost)), label="ASSUMPTION (zero-mean noise on the two bracketing prices; no bid/ask observed)"))
    _write(pd.DataFrame(mc), out, "t10_random_noise_scenarios.csv")
    # tipping bias
    b = e
    s1 = float(b.spread_calendar.median())
    r1 = float(math.exp(np.log(b.total_variance_ratio[b.total_variance_ratio_valid]).mean()))
    tip = Q.tipping_bias(rd, s1, r1)
    _write(pd.DataFrame([dict(**tip, baseline_median_calendar_spread=s1, baseline_geometric_mean_ratio=r1,
                              reading="systematic bias that would be needed to explain the whole spread; a bound, not an estimate of any actual bias")]), out, "t10_tipping_bias.csv")
    # quote-age gradient
    mx = np.maximum(rd.age_lo, rd.age_hi)
    ag = []
    for j, lim in enumerate((1.0, 2.0, 5.0)):
        m = mx <= lim
        sp = SC.Spec(f"age_le_{lim:g}", "quote_age", f"<= {lim:g} min", rd.base[m].reset_index(drop=True))
        ag.append(SC.estimate_spec(sp, BLOCK, boot_reps, SEED + 200 + j).assign(max_bracket_quote_age_min=lim, share_of_rows=float(m.mean())))
    _write(pd.concat(ag, ignore_index=True).drop(columns=["spec", "dimension", "value", "note"]), out, "t10_quote_age_gradient.csv")
    # observed call/put disagreement (upper bound for quote noise)
    smp = e.assign(year=e.day.str[:4])
    rng = np.random.default_rng(SEED + 300)
    chosen = []
    for yr, d in smp.groupby("year"):
        exps = sorted(d.expiry.unique())
        pick = rng.choice(exps, size=min(cp_expiries_per_year, len(exps)), replace=False)
        for ex in pick:
            dd = d[d.expiry == ex]
            for i in rng.choice(len(dd), size=min(cp_snaps_per_expiry, len(dd)), replace=False):
                chosen.append(dd.iloc[i])
    samp = pd.DataFrame(chosen)
    files = dict(loaders.list_expiry_files(root))
    smi = smiles.set_index("smile_id")
    parts = []
    for ex, d in samp.groupby("expiry"):
        df = loaders.load_expiry_frame(files[ex])
        for r in d.itertuples():
            t0 = loaders.snapshot_ts(date.fromisoformat(r.day), r.time)
            qs = loaders.quotes_at(df, t0)
            row = smi.loc[r.obs_id]
            part = Q.callput_disagreement(qs, float(row.forward_used), float(row.T_days) / 365.0)
            if len(part):
                parts.append(part.assign(obs_id=r.obs_id, year=r.day[:4]))
    cp = pd.concat(parts, ignore_index=True)
    persnap = cp.groupby("obs_id").diff_vol_points.median()
    mad_sd = 1.4826 * float(np.median(np.abs(cp.diff_vol_points - cp.diff_vol_points.median())))
    summ = dict(n_snapshots=int(cp.obs_id.nunique()), n_strikes=len(cp), median_diff_vol_points=float(cp.diff_vol_points.median()), mean_diff_vol_points=float(cp.diff_vol_points.mean()),
                robust_sd_diff=mad_sd, p05=float(cp.diff_vol_points.quantile(.05)), p95=float(cp.diff_vol_points.quantile(.95)), median_abs_diff=float(cp.diff_vol_points.abs().median()),
                p90_abs_diff=float(cp.diff_vol_points.abs().quantile(.90)), median_of_per_snapshot_median=float(persnap.median()),
                reading="UPPER BOUND on quote noise: also contains forward-estimation error and ITM-side time-value effects; the median shows the sign of any call-vs-put (bid/ask-type) asymmetry")
    _write(pd.DataFrame([summ]), out, "t10_call_put_disagreement.csv")
    by_year = cp.groupby("year").diff_vol_points.agg(["count", "median", lambda s: s.abs().median()]).reset_index()
    by_year.columns = ["year", "n_strikes", "median_diff_vol_points", "median_abs_diff"]
    _write(by_year, out, "t10_call_put_disagreement_by_year.csv")
    # errors-in-variables scenarios
    ok = e[["iv_pct", "rv_calendar_pct"]].dropna()
    _, slope = ST.ols(ok.iv_pct.to_numpy(float), ok.rv_calendar_pct.to_numpy(float))
    var_iv = float(ok.iv_pct.var(ddof=0))
    vega = float(np.median(rd.vega_atm()))
    sds = {"call-put proxy: robust sd / sqrt(2)": mad_sd / math.sqrt(2), "call-put proxy: median |diff| / sqrt(2)": summ["median_abs_diff"] / math.sqrt(2), "call-put proxy: p90 |diff| / sqrt(2)": summ["p90_abs_diff"] / math.sqrt(2)}
    for par in NOISE_ABS:
        sds[f"scenario: Rs {par:g} price noise / median vega"] = par / vega
    eiv = [dict(noise_source=k, noise_sd_vol_points=v, share_of_iv_variance=v * v / var_iv, observed_slope=slope, corrected_slope=Q.eiv_corrected_slope(slope, var_iv, v),
                label="SCENARIO (classical measurement-error correction; not an estimate)") for k, v in sds.items()]
    _write(pd.DataFrame(eiv), out, "t10_eiv_scenarios.csv")
    return dict(n_rows=rd.n, call_put=summ, tipping=tip)


def build(aligned: str, stage2c: str, stage2a: str, out: str, spot: str = "data/hist1m/NIFTY50_1m.parquet", root: str = "data/hist1m", participant_dir: str = "data/participant_oi",
          perm_draws: int = 2000, boot_reps: int = 1000, mc_draws: int = MC_DRAWS, cp_expiries_per_year: int = 10, cp_snaps_per_expiry: int = 5) -> dict:
    if "stage2d" not in os.path.normpath(out).split(os.sep):
        raise ValueError("Stage 2D outputs must live under a 'stage2d' directory (existing research artifacts are never overwritten)")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    g1.load_baseline(aligned, stage2c)                                      # baseline reproduction guard
    aligned_all = pd.read_csv(aligned)
    smiles = pd.read_csv(os.path.join(stage2a, "smiles.csv"), usecols=["smile_id", "day", "time", "expiry", "split", "expiry_day", "T_days", "forward_status", "group_reason", "atm_iv", "forward_used"])
    points = pd.read_csv(os.path.join(stage2a, "points.csv"), usecols=["smile_id", "strike", "kind", "price", "age_min", "log_moneyness", "iv", "used", "forward", "T_days"])
    lo = pd.read_csv(os.path.join(stage2c, "iv_rv_observations.csv"), usecols=["obs_id", "horizon", "measure", "target_available", "unavailable_reason"])
    unavail = lo[(lo.horizon == "EXP") & (lo.measure == "hybrid") & (~lo.target_available)].set_index("obs_id").unavailable_reason.to_dict()
    feats = SEL.load_spot_features(spot, participant_dir)
    t9 = run_t9(smiles, feats, aligned_all, unavail, points[points.used.astype(bool)][["age_min"]], out, perm_draws, boot_reps)
    t10 = run_t10(aligned_all, smiles, points, root, out, boot_reps, mc_draws, cp_expiries_per_year, cp_snaps_per_expiry)
    meta = dict(git_commit=g1._git(), n_groups=t9["n_groups"], n_observed=t9["n_obs"], n_l1=t9["n_l1"], n_l2=t9["n_l2"], perm_draws=perm_draws, boot_reps=boot_reps, mc_draws=mc_draws, base_seed=SEED, block=BLOCK,
                seconds=round(time.time() - t0, 1), started=datetime.now().isoformat(timespec="seconds"), numpy=np.__version__, pandas=pd.__version__,
                plan="GATE3A_IMPLEMENTATION_PLAN.md")
    with open(os.path.join(out, "run_metadata_3a.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", default="research_output/stage2c/annualization_audit/aligned_observations.csv")
    ap.add_argument("--stage2c", default="research_output/stage2c")
    ap.add_argument("--stage2a", default="research_output/stage2a/full")
    ap.add_argument("--out", default="research_output/stage2d/gate3")
    ap.add_argument("--perm-draws", type=int, default=2000)
    ap.add_argument("--boot-reps", type=int, default=1000)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.aligned, a.stage2c, a.stage2a, a.out, perm_draws=a.perm_draws, boot_reps=a.boot_reps), indent=1))


if __name__ == "__main__":
    main()
