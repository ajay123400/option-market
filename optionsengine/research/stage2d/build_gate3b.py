"""Stage 2D Gate 3b builder: T11 Tier A (full-pipeline recovery) and Tier B (synthetic coverage / dependence / size-power study). Writes ONLY under a `stage2d` directory.

python -m optionsengine.research.stage2d.build_gate3b --out research_output/stage2d/gate3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import build_annualization_audit as baa, build_iv_rv as biv, build_surface as bsf
from . import build_gate1 as g1, clusters as C, coverage_study as CS, estimands as E, synth_pipeline as SPL, synth_process as SP

SEED = 20260101
TIER_A_WORLDS = ((1, 1.00), (2, 1.30), (3, 0.80))


def _write(df: pd.DataFrame, out: str, name: str) -> None:
    df.to_csv(os.path.join(out, name), index=False)


# ------------------------------------------------------------------------------ Tier A
def truth_aligned(truth: pd.DataFrame) -> pd.DataFrame:
    """The generator's own rows in the aligned format (oracle IV and realised variance), for estimand-level comparison."""
    d = pd.DataFrame(dict(expiry=truth.expiry, day=truth.day, time=truth.time, T_days=truth.T_years * 365.0, iv_pct=100 * truth.iv_true, rv_total_variance=truth.V_true,
                          implied_total_variance=truth.W_true))
    rs = 100 * np.sqrt(252.0 * truth.V_true / truth.session_equivalents)
    rc = 100 * np.sqrt(365.0 * truth.V_true / truth.span_days)
    d["spread_session"], d["spread_calendar"] = d.iv_pct - rs, d.iv_pct - rc
    d["total_variance_ratio_valid"] = truth.V_true > 0
    d["total_variance_ratio"] = truth.W_true / truth.V_true
    return d


def run_real_stack(root: str, pdir: str, work: str, workers: int = 2) -> pd.DataFrame:
    """The REAL, unmodified Stage 2A -> 2C -> annualization-audit builders on a synthetic world; returns the EXP/hybrid aligned rows."""
    s2a, s2c = os.path.join(work, "s2a"), os.path.join(work, "s2c")
    bsf.build(root, s2a, workers=workers)
    biv.build(s2a, os.path.join(root, "NIFTY50_1m.parquet"), s2c, pdir, reps=20, reps_secondary=20)
    baa.build(s2c, reps=20, reps_secondary=20)
    al = pd.read_csv(os.path.join(s2c, "annualization_audit", "aligned_observations.csv"))
    return al[(al.horizon == "EXP") & (al.measure == "hybrid")].reset_index(drop=True)


def run_tier_a(out: str, worlds: Sequence[Tuple[int, float]] = TIER_A_WORLDS, boot_reps: int = 1000, workers: int = 2) -> dict:
    summ, rows_all, est_rows = [], [], []
    for seed, c in worlds:
        tmp = tempfile.mkdtemp(prefix="tierA_")
        try:
            dgp = SPL.TierADGP(c=c)
            root, pdir = os.path.join(tmp, "root"), os.path.join(tmp, "pcp")
            truth = SPL.generate_world(dgp, seed, root, pdir)
            e = run_real_stack(root, pdir, tmp, workers)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        rec = SPL.recovery_table(e, truth, c)
        rec.insert(0, "world_seed", seed)
        rows_all.append(rec)
        est_p = E.build(e)
        t_al = truth_aligned(truth[truth.obs_id.isin(e.obs_id)])
        est_t = E.build(t_al)
        pp, tt = est_p.point(), est_t.point()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dr = C.bootstrap(est_p.cs, est_p.evaluate, "moving_block", 5, boot_reps, SEED + seed, k=len(E.EST_NAMES))
        lo, hi = C.percentile_ci(dr)
        i3 = E.EST_NAMES.index("R3_ratio_of_summed_total_variance")
        se = float(np.nanstd(dr[:, i3], ddof=1))
        for i, n in enumerate(E.EST_NAMES):
            est_rows.append(dict(world_seed=seed, c=c, estimand=n, pipeline=pp[i], truth_rows=tt[i], difference=pp[i] - tt[i], ci_lo=lo[i], ci_hi=hi[i], n_rows=len(e), n_expiries=est_p.cs.n_clusters))
        r3 = pp[i3]
        summ.append(dict(world_seed=seed, c=c, n_truth_rows=len(truth), n_pipeline_rows=len(e), n_matched=len(rec), max_abs_V_rel_error=float(rec.V_rel_error.abs().max()),
                         max_abs_iv_error_vol_pts=float(rec.iv_error.abs().max()), median_abs_iv_error_vol_pts=float(rec.iv_error.abs().median()), max_abs_spread_cal_error=float(rec.spread_cal_error.abs().max()),
                         max_abs_spread_sess_error=float((rec.spread_sess_pipeline - rec.spread_sess_true).abs().max()), max_abs_ratio_error=float(rec.ratio_error.abs().max()),
                         max_abs_annualization_effect_error=float(((rec.spread_sess_pipeline - rec.spread_cal_pipeline) - (rec.spread_sess_true - rec.spread_cal_true)).abs().max()),
                         max_abs_estimand_difference=float(np.nanmax(np.abs(pp - tt))), R3_pipeline=float(r3), R3_oracle_rows=float(tt[i3]), R3_bootstrap_se=se, R3_minus_c_in_se=float((r3 - c) / se), R3_ci_contains_c=bool(lo[i3] <= c <= hi[i3])))
    sim, theo = SPL.expected_variance_check(SPL.TierADGP())
    m, se_m, c0 = SPL.oracle_r3_across_worlds(SPL.TierADGP(c=1.0), 200)
    gen = pd.DataFrame([dict(check="Monte-Carlo mean of the generator's snapshot-session remainder variance vs the closed-form expectation (20,000 sessions)", simulated=sim, closed_form=theo, relative_difference=sim / theo - 1, mc_se=np.nan),
                        dict(check="generator-level unbiasedness: mean over 200 worlds (c = 1) of the oracle ratio of sums sum(W)/sum(V) vs c", simulated=m, closed_form=c0, relative_difference=m / c0 - 1, mc_se=se_m)])
    _write(pd.DataFrame(summ), out, "t11a_recovery_summary.csv")
    _write(pd.concat(rows_all, ignore_index=True), out, "t11a_recovery_rows.csv")
    _write(pd.DataFrame(est_rows), out, "t11a_estimand_recovery.csv")
    _write(gen, out, "t11a_generator_check.csv")
    return dict(worlds=len(worlds), rows=int(sum(s["n_matched"] for s in summ)))


# ------------------------------------------------------------------------------ Tier B
def run_tier_b(aligned: str, out: str, n_hist: int, reps: int, wald_reps: int, truth_expiries: int, calib_hist: int, workers: int, only: Optional[Sequence[str]] = None) -> dict:
    al = pd.read_csv(aligned)
    e = al[(al.horizon == "EXP") & (al.measure == "hybrid")].reset_index(drop=True)
    obs = SP.moments_from_aligned(e)
    base = SP.DGP()
    sims = pd.DataFrame([SP.moments(SP.simulate(base, 260, 5000 + i), base.c) for i in range(calib_hist)])
    calib = pd.DataFrame(dict(moment=list(obs), observed=[obs[k] for k in obs], simulated_mean=[sims[k].mean() for k in obs], simulated_sd=[sims[k].std(ddof=1) for k in obs],
                              simulated_p05=[sims[k].quantile(.05) for k in obs], simulated_p95=[sims[k].quantile(.95) for k in obs]))
    calib["observed_inside_simulated_5_95"] = (calib.observed >= calib.simulated_p05) & (calib.observed <= calib.simulated_p95)
    calib["relative_difference"] = (calib.simulated_mean - calib.observed) / calib.observed.abs().replace(0, np.nan)
    _write(calib, out, "t11b_calibration.csv")
    truth_rows, cov, con, wal, dep = [], [], [], [], []
    hist_iv, hist_con, hist_wald = [], [], []
    for sc in CS.scenarios():
        if only and sc.name not in only:
            continue
        truth = SP.truth_estimates(sc.dgp, truth_expiries, 987654)
        for k, v in truth.items():
            truth_rows.append(dict(scenario=sc.name, estimand=k, population_value=v, truth_expiries=truth_expiries))
        res = CS.run_scenario(sc, n_hist, reps, wald_reps, SEED + 17, workers)
        cov.append(CS.coverage_rows(sc, res, truth))
        hist_iv.append(CS.history_interval_rows(sc, res))
        hist_con.append(CS.history_contrast_rows(sc, res))
        hist_wald.append(CS.history_wald_rows(sc, res))
        if sc.contrast:
            td = {"null": {n: 0.0 for n in E.EST_NAMES}, "shift": CS.truth_difference(sc.dgp, sc.shift_factor, truth_expiries)}
            parts = []
            for kind in sc.contrast:
                sub = replace_kind(sc, kind)
                parts.append(CS.contrast_rows(sub, res, td[kind]))
            con.append(pd.concat(parts, ignore_index=True))
        if sc.wald:
            wal.append(CS.wald_rows(sc, res))
        dep.append(dict(scenario=sc.name, description=sc.description, n_histories=len(res), mean_n_rows=float(np.mean([r["n_rows"] for r in res])), mean_icc_by_expiry=float(np.mean([r["icc"] for r in res])),
                        mean_acf_lag1_expiry_median_spread=float(np.mean([r["acf1"] for r in res])), sd_acf_lag1=float(np.std([r["acf1"] for r in res], ddof=1))))
    _write(pd.DataFrame(truth_rows), out, "t11b_truth.csv")
    _write(pd.concat(hist_iv, ignore_index=True), out, "t11b_history_intervals.csv")
    _write(pd.concat(hist_con, ignore_index=True), out, "t11b_history_contrasts.csv")
    _write(pd.concat([h for h in hist_wald if len(h)], ignore_index=True), out, "t11b_history_wald.csv")
    _write(pd.concat(cov, ignore_index=True), out, "t11b_coverage.csv")
    if con:
        _write(pd.concat(con, ignore_index=True), out, "t11b_split_contrast.csv")
    if wal:
        _write(pd.concat(wal, ignore_index=True), out, "t11b_wald.csv")
    _write(pd.DataFrame(dep), out, "t11b_dependence.csv")
    return dict(scenarios=len(dep), n_hist=n_hist)


def replace_kind(sc: CS.Scenario, kind: str) -> CS.Scenario:
    from dataclasses import replace
    return replace(sc, contrast=(kind,))


def build(aligned: str, stage2c: str, out: str, n_hist: int = 300, reps: int = 500, wald_reps: int = 300, truth_expiries: int = 40000, calib_hist: int = 100, workers: int = 4,
          tier_a: bool = True, tier_b: bool = True, only: Optional[Sequence[str]] = None, worlds: Sequence[Tuple[int, float]] = TIER_A_WORLDS) -> dict:
    if "stage2d" not in os.path.normpath(out).split(os.sep):
        raise ValueError("Stage 2D outputs must live under a 'stage2d' directory (existing research artifacts are never overwritten)")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    g1.load_baseline(aligned, stage2c)
    ra = run_tier_a(out, worlds, workers=min(workers, 2)) if tier_a else {}
    rb = run_tier_b(aligned, out, n_hist, reps, wald_reps, truth_expiries, calib_hist, workers, only) if tier_b else {}
    meta = dict(git_commit=g1._git(), tier_a=ra, tier_b=rb, n_hist=n_hist, bootstrap_reps=reps, wald_reps=wald_reps, truth_expiries=truth_expiries, calibration_histories=calib_hist, base_seed=SEED,
                seconds=round(time.time() - t0, 1), started=datetime.now().isoformat(timespec="seconds"), numpy=np.__version__, pandas=pd.__version__, plan="GATE3B_IMPLEMENTATION_PLAN.md")
    with open(os.path.join(out, "run_metadata_3b.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", default="research_output/stage2c/annualization_audit/aligned_observations.csv")
    ap.add_argument("--stage2c", default="research_output/stage2c")
    ap.add_argument("--out", default="research_output/stage2d/gate3")
    ap.add_argument("--hist", type=int, default=300)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-tier-a", action="store_true")
    ap.add_argument("--skip-tier-b", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.aligned, a.stage2c, a.out, a.hist, a.reps, workers=a.workers, tier_a=not a.skip_tier_a, tier_b=not a.skip_tier_b, only=a.only.split(",") if a.only else None), indent=1, default=str))


if __name__ == "__main__":
    main()
