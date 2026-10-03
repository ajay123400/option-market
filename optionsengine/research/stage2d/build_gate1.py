"""Stage 2D Gate 1 builder (T1-T4). Reads the Stage 2C aligned observations, writes ONLY under a `stage2d` output directory.

python -m optionsengine.research.stage2d.build_gate1 --aligned research_output/stage2c/annualization_audit/aligned_observations.csv \
    --stage2c research_output/stage2c --out research_output/stage2d/gate1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from . import clusters as C, dependence as D, estimands as E, influence as I, ratio_inference as R

BASE_SEED = 20260101
HEADLINE_REPS = 5000
GRID_REPS = 2000
EXPIRY_BLOCK_GRID = [1, 2, 3, 4, 5, 6, 8, 12]
DATE_BLOCK_GRID = [1, 5, 10, 20, 40]
TOPK = [1, 5, 10, 20]
SERIES = {"median_spread_session": "S2-series: per-expiry median spread, session basis",
          "median_spread_calendar": "S2-series: per-expiry median spread, calendar basis",
          "mean_log_ratio": "R1e-series: per-expiry mean ln(W_imp/V)"}


def _git() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        return c + ("+dirty" if d else "")
    except Exception:
        return "unknown"


def load_baseline(aligned: str, stage2c: str) -> pd.DataFrame:
    df = pd.read_csv(aligned)
    e = df[(df.horizon == "EXP") & (df.measure == "hybrid")].reset_index(drop=True)
    ova = pd.read_csv(os.path.join(stage2c, "annualization_audit", "original_vs_aligned.csv"))
    tv = pd.read_csv(os.path.join(stage2c, "annualization_audit", "expiry_total_variance.csv"))
    o = ova[(ova.horizon == "EXP") & (ova.measure == "hybrid") & (ova.dimension == "overall")].iloc[0]
    t = tv[(tv.measure == "hybrid") & (tv.dimension == "overall")].iloc[0]
    est = E.build(e).point()
    chk = {"n_rows": (len(e), int(o.n_obs)), "S1_session": (est[0], float(o.spread_session_median)), "S1_calendar": (est[2], float(o.spread_calendar_median)),
           "R1_geometric_mean": (est[4], float(t.ratio_geometric_mean)), "R2_median_ratio": (est[6], float(t.ratio_median))}
    bad = {k: v for k, v in chk.items() if abs(v[0] - v[1]) > 1e-9}
    if bad:
        raise RuntimeError(f"baseline mismatch with committed Stage 2C results: {bad}")
    return e


def _interval_rows(est: E.Estimands, method: str, scheme: str, block, reps: int, seed: int, theta: np.ndarray) -> List[dict]:
    d = C.bootstrap(est.cs, est.evaluate, scheme, block, reps, seed, k=len(E.EST_NAMES))
    with warnings.catch_warnings():                                   # all-NaN columns (estimands undefined for this design) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = C.percentile_ci(d)
        se = np.nanstd(d, axis=0, ddof=1)
    return [dict(estimand=n, method=method, scheme=scheme, block=block, reps=reps, n_clusters=est.cs.n_clusters, estimate=theta[i],
                 ci_lo=lo[i], ci_hi=hi[i], width=hi[i] - lo[i], boot_se=se[i])
            for i, n in enumerate(E.EST_NAMES)]


NULL = {n: (0.0 if n.startswith("S") else 1.0) for n in E.EST_NAMES}


def build(aligned: str, stage2c: str, out: str, headline_reps: int = HEADLINE_REPS, grid_reps: int = GRID_REPS) -> dict:
    if "stage2d" not in os.path.normpath(out).split(os.sep):
        raise ValueError("Stage 2D outputs must live under a 'stage2d' directory (existing research artifacts are never overwritten)")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    e = load_baseline(aligned, stage2c)
    est = E.build(e)
    theta = est.point()
    per = est.per_cluster_table().rename(columns={"label": "expiry"})
    per["expiry_year"] = per.expiry.str[:4]
    per.to_csv(os.path.join(out, "expiry_series.csv"), index=False)
    pd.DataFrame(dict(estimand=E.EST_NAMES, estimate=theta, n_rows=len(e), n_expiries=est.cs.n_clusters, n_dates=e.day.nunique(),
                      null_value=[NULL[n] for n in E.EST_NAMES])).to_csv(os.path.join(out, "estimands_point.csv"), index=False)

    # ---------------------------------------------------------------- T2 dependence
    rows_diag, acf_rows, pw_choice = [], [], {}
    row_series = {"median_spread_session": e.spread_session.to_numpy(float), "median_spread_calendar": e.spread_calendar.to_numpy(float),
                  "mean_log_ratio": np.where(est.valid, est.loglr, np.nan)}
    for name in SERIES:
        x = per[name].to_numpy(float)
        x = x[np.isfinite(x)]
        pw = D.politis_white(x)
        pw_choice[name] = pw
        lb10, lb20 = D.ljung_box(x, 10), D.ljung_box(x, 20)
        nw_rule, nw8 = D.newey_west(x), D.newey_west(x, 8)
        r = row_series[name]
        ok = np.isfinite(r)
        icc_e = D.icc_oneway(r[ok], e.expiry.to_numpy()[ok])
        icc_d = D.icc_oneway(r[ok], e.day.to_numpy()[ok])
        rows_diag.append(dict(series=name, n_expiries=len(x), icc_by_expiry=icc_e["icc"], design_effect_by_expiry=icc_e["design_effect"], n_effective_by_expiry=icc_e["n_effective"],
                              icc_by_date=icc_d["icc"], design_effect_by_date=icc_d["design_effect"], n_effective_by_date=icc_d["n_effective"],
                              acf_lag1=D.acf(x, 1)[1], ljung_box_q10=lb10[0], ljung_box_p10=lb10[1], ljung_box_q20=lb20[0], ljung_box_p20=lb20[1],
                              pw_m_hat=pw["m_hat"], pw_block_stationary=pw["b_stationary"], pw_block_circular=pw["b_circular"],
                              nw_lags_rule=nw_rule["lags"], mean=nw_rule["mean"], iid_se=nw_rule["iid_se"], nw_se_rule=nw_rule["nw_se"], nw_variance_inflation_rule=nw_rule["variance_inflation"],
                              nw_se_lag8=nw8["nw_se"], nw_variance_inflation_lag8=nw8["variance_inflation"]))
        a = D.acf(x, 12)
        for k in range(1, 13):
            acf_rows.append(dict(series=name, lag=k, acf=a[k], approx_95_band=1.96 / math.sqrt(len(x))))
    pd.DataFrame(rows_diag).to_csv(os.path.join(out, "dependence_diagnostics.csv"), index=False)
    pd.DataFrame(acf_rows).to_csv(os.path.join(out, "acf_expiry_series.csv"), index=False)
    b_star = int(max(round(v["b_circular"]) for v in pw_choice.values()))             # pre-specified rule: the largest rounded PW circular length
    win = D.expiry_windows(e)
    ov = D.window_overlap(win)
    ov = ov.merge(win[["expiry", "start", "end", "n_rows"]], on="expiry")
    ov.to_csv(os.path.join(out, "window_overlap.csv"), index=False)
    ov_sum = dict(n_expiries=len(ov), mean_window_days=float(ov.length_days.mean()), mean_overlap_with_next_days=float(ov.overlap_with_next_days.mean()),
                  mean_overlap_with_next_share=float(ov.overlap_with_next_share.mean()), median_n_overlapping_expiries=float(ov.n_overlapping_expiries.median()),
                  max_n_overlapping_expiries=int(ov.n_overlapping_expiries.max()))

    # ---------------------------------------------------------------- T1 uncertainty: method comparison
    ci_rows: List[dict] = []
    date_est = E.build(e, cluster_col="day")
    row_est = E.build(e, row_level=True)
    methods = [("row_iid_NAIVE_reference", row_est, "iid", None), ("date_block_stage2c_method", date_est, "moving_block", 10),
               ("expiry_iid_cluster", est, "iid", None), (f"expiry_moving_block_b{b_star}", est, "moving_block", b_star),
               (f"expiry_stationary_mean_b{b_star}", est, "stationary", b_star)]
    for j, (name, ev, scheme, block) in enumerate(methods):
        ci_rows += _interval_rows(ev, name, scheme, block, headline_reps, BASE_SEED + j, theta)
    ci = pd.DataFrame(ci_rows)
    ci["null_value"] = ci.estimand.map(NULL)
    ci["null_inside_interval"] = ((ci.ci_lo <= ci.null_value) & (ci.null_value <= ci.ci_hi)).astype(object).where(ci.ci_lo.notna(), None)
    ci.to_csv(os.path.join(out, "ci_methods_comparison.csv"), index=False)

    # block-length sensitivity (all estimands, same seed across b so width differences reflect b, not noise)
    grid = []
    for b in EXPIRY_BLOCK_GRID:
        grid += _interval_rows(est, f"expiry_moving_block_b{b}", "moving_block", b, grid_reps, BASE_SEED + 101, theta)
    for b in DATE_BLOCK_GRID:
        grid += _interval_rows(date_est, f"date_block_b{b}", "moving_block", b, grid_reps, BASE_SEED + 102, theta)
    pd.DataFrame(grid).to_csv(os.path.join(out, "block_length_sensitivity.csv"), index=False)

    # cross-check of the standard error of the mean per-expiry series: iid / Newey-West / block bootstrap
    xs = []
    for name in SERIES:
        x = per[name].to_numpy(float)
        x = x[np.isfinite(x)]
        nw = D.newey_west(x)
        r = dict(series=name, n=len(x), mean=float(x.mean()), se_iid_formula=nw["iid_se"], se_newey_west=nw["nw_se"])
        for sch, blk in (("iid", None), ("moving_block", b_star), ("stationary", b_star)):
            r[f"se_bootstrap_{sch}"] = float(np.std(C.bootstrap_series(x, sch, blk, grid_reps, BASE_SEED + 7), ddof=1))
        xs.append(r)
    pd.DataFrame(xs).to_csv(os.path.join(out, "mean_se_crosscheck.csv"), index=False)

    # non-overlapping subsample (outcome-free rule) -------------------------------------------------------
    kept, dropped = D.nonoverlap_subsample(e)
    dropped.to_csv(os.path.join(out, "nonoverlap_subsample_dropped.csv"), index=False)
    kept[["expiry", "day", "time", "T_days", "target_start_ts", "target_end_ts"]].to_csv(os.path.join(out, "nonoverlap_subsample_selection.csv"), index=False)
    sub_est = E.build(kept.reset_index(drop=True))
    sub_theta = sub_est.point()
    sub_rows = []
    for j, (name, scheme, block) in enumerate((("expiry_iid_cluster", "iid", None), (f"expiry_moving_block_b{b_star}", "moving_block", b_star))):
        sub_rows += _interval_rows(sub_est, name, scheme, block, headline_reps, BASE_SEED + 50 + j, sub_theta)
    sub = pd.DataFrame(sub_rows)
    sub["n_rows"] = len(kept)
    sub["full_sample_estimate"] = sub.estimand.map(dict(zip(E.EST_NAMES, theta)))
    sub.to_csv(os.path.join(out, "nonoverlap_subsample_estimates.csv"), index=False)

    # ---------------------------------------------------------------- T3 ratio inference
    lr = per.mean_log_ratio.to_numpy(float)
    med_r = per.median_ratio.to_numpy(float)
    st = R.sign_test(med_r, center=1.0)
    t_rows = [dict(test="sign_test_expiry_median_ratio_gt_1", assumption="independent expiries", n=st["n"], statistic=st["share_positive"], block=None, p_two_sided=st["p_two_sided"])]
    for b in (1, b_star, 8):
        sf = R.sign_flip_test(lr, block=b, reps=20000, seed=BASE_SEED + 300 + b)
        t_rows.append(dict(test="sign_flip_permutation_mean_log_ratio", assumption=("independent expiries" if b == 1 else f"dependence shorter than {b} expiries; symmetry under H0"),
                           n=sf["n"], statistic=sf["statistic"], block=b, p_two_sided=sf["p_two_sided"]))
    for L in (D.newey_west(lr)["lags"], 8):
        h = R.hac_t(lr, L)
        t_rows.append(dict(test="hac_t_mean_log_ratio", assumption="Bartlett kernel, stationary series", n=len(lr), statistic=h["t"], block=h["lags"], p_two_sided=None))
    t_rows.append(dict(test="rows_with_zero_realized_variance_excluded", assumption="", n=int((~est.valid).sum()), statistic=None, block=None, p_two_sided=None))
    pd.DataFrame(t_rows).to_csv(os.path.join(out, "ratio_inference.csv"), index=False)

    # ---------------------------------------------------------------- T4 influence
    loo = I.leave_one_out(est.cs, est)
    js = I.jackknife_se(theta, loo)
    bjs = I.block_jackknife_se(est.cs, est, b_star)
    loo_df = pd.DataFrame(loo, columns=E.EST_NAMES)
    loo_df.insert(0, "expiry", est.cs.labels)
    loo_df.to_csv(os.path.join(out, "influence_leave_one_expiry_out.csv"), index=False)
    se_boot = {m: ci[(ci.method == m)].set_index("estimand").boot_se for m in ci.method.unique()}
    inf = []
    for i, n in enumerate(E.EST_NAMES):
        d = loo[:, i] - theta[i]
        top = np.argsort(-np.abs(d), kind="stable")[:3]
        inf.append(dict(estimand=n, estimate=theta[i], jackknife_se=js["se"][i], jackknife_bias=js["bias"][i], block_jackknife_se=bjs[i], block_jackknife_block=b_star,
                        boot_se_expiry_iid=se_boot["expiry_iid_cluster"][n], boot_se_expiry_block=se_boot[f"expiry_moving_block_b{b_star}"][n],
                        loo_min=loo[:, i].min(), loo_max=loo[:, i].max(), max_abs_change=np.abs(d).max(),
                        most_influential_expiries=";".join(f"{est.cs.labels[t]}({d[t]:+.4f})" for t in top)))
    pd.DataFrame(inf).to_csv(os.path.join(out, "influence_summary.csv"), index=False)
    years = np.array([s[:4] for s in est.cs.labels])
    lo = I.leave_one_group_out(est.cs, est, years)
    pd.DataFrame([dict(left_out_expiry_year=y, n_expiries_removed=cnt, **dict(zip(E.EST_NAMES, v))) for y, (cnt, v) in lo.items()]).to_csv(os.path.join(out, "leave_one_year_out.csv"), index=False)
    # each year alone (for context)
    yr = []
    for y in sorted(set(years)):
        try:
            sub_e = E.Estimands(est.cs.subset(years == y, min_clusters=1))
            yr.append(dict(expiry_year=y, n_expiries=int((years == y).sum()), **dict(zip(E.EST_NAMES, sub_e.point()))))
        except Exception as ex:                                       # pragma: no cover
            yr.append(dict(expiry_year=y, error=str(ex)))
    pd.DataFrame(yr).to_csv(os.path.join(out, "by_expiry_year_point.csv"), index=False)
    tk = I.topk_removal(est.cs, est, loo, theta, TOPK, 200, BASE_SEED + 9)
    tk_rows = []
    for k, v in tk.items():
        for i, n in enumerate(E.EST_NAMES):
            tk_rows.append(dict(k_removed=k, estimand=n, full_estimate=theta[i], after_removing_top_k_influential=v["top"][i],
                                change=v["top"][i] - theta[i], random_k_p05=v["rnd_p05"][i], random_k_p50=v["rnd_p50"][i], random_k_p95=v["rnd_p95"][i]))
    pd.DataFrame(tk_rows).to_csv(os.path.join(out, "topk_removal.csv"), index=False)
    tw = []
    for name in SERIES:
        x = per[name].to_numpy(float)
        x = x[np.isfinite(x)]
        f = (lambda v: math.exp(v)) if name == "mean_log_ratio" else (lambda v: v)
        tw.append(dict(series=name, scale="geometric (exp of mean ln r)" if name == "mean_log_ratio" else "vol points", n_expiries=len(x), mean=f(x.mean()), median=f(float(np.median(x))),
                       trimmed_mean_10pct=f(I.trimmed_mean(x, 0.10)), winsorized_mean_5pct=f(I.winsorized_mean(x, 0.05))))
    pd.DataFrame(tw).to_csv(os.path.join(out, "trimmed_winsorized.csv"), index=False)

    meta = dict(git_commit=_git(), aligned=aligned, n_rows=len(e), n_expiries=est.cs.n_clusters, n_dates=int(e.day.nunique()), base_seed=BASE_SEED, headline_reps=headline_reps, grid_reps=grid_reps,
                block_length_rule="largest rounded Politis-White circular length over the three per-expiry series", block_length_selected=b_star,
                politis_white={k: {kk: (float(vv) if not isinstance(vv, int) else vv) for kk, vv in v.items()} for k, v in pw_choice.items()},
                window_overlap_summary=ov_sum, n_subsample_kept=len(kept), n_subsample_dropped=len(dropped), seconds=round(time.time() - t0, 1),
                started=datetime.now().isoformat(timespec="seconds"), numpy=np.__version__, pandas=pd.__version__, min_clusters=C.MIN_CLUSTERS)
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", default="research_output/stage2c/annualization_audit/aligned_observations.csv")
    ap.add_argument("--stage2c", default="research_output/stage2c")
    ap.add_argument("--out", default="research_output/stage2d/gate1")
    ap.add_argument("--reps", type=int, default=HEADLINE_REPS)
    ap.add_argument("--grid-reps", type=int, default=GRID_REPS)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.aligned, a.stage2c, a.out, a.reps, a.grid_reps), indent=1, default=str))


if __name__ == "__main__":
    main()
