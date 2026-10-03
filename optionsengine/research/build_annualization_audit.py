"""Build the Stage 2C annualization-alignment audit tables (measurement only).

Reads  <stage2c>/iv_rv_observations.csv  (never modified) and writes ONLY into <stage2c>/annualization_audit/ :
  aligned_observations.csv        per-observation aligned columns (large, gitignored)
  aligned_sample_observations.csv small sample of the above
  original_vs_aligned.csv         session-basis vs calendar-aligned spread by dimension (overall, DTE, time, split, expiry day, weekend)
  annualization_effect.csv        how much of the session-basis median spread is the annualization effect (per horizon/measure/split)
  expiry_total_variance.csv       PRIMARY diagnostic: implied total variance (IV^2*T) vs realized total variance, expiry-aligned sample
  fixed_horizon_calendar_span.csv sessions vs elapsed calendar days for F1/F5/F10/F20 and the effect on annualized RV
  audit_metadata.json

Usage: python -m optionsengine.research.build_annualization_audit --stage2c research_output/stage2c
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from . import annualization_audit as aa
from .iv_rv_stats import block_bootstrap, block_length

DIMENSIONS = {"overall": None, "dte_bucket": "dte_bucket", "snapshot_time": "time", "split": "split", "expiry_day": "expiry_day",
              "weekend_in_window": "weekend_in_window"}
TARGETS = [("EXP", "hybrid"), ("F1", "hybrid"), ("F5", "hybrid"), ("F10", "hybrid"), ("F20", "hybrid"),
           ("F1", "close_to_close"), ("F5", "close_to_close"), ("F10", "close_to_close"), ("F20", "close_to_close")]


def _med_ci(g: pd.DataFrame, col: str, horizon: str, reps: int) -> tuple:
    return block_bootstrap(g[col].to_numpy(dtype=float), g["day"].to_numpy(), {"median": np.median}, block_length(horizon), reps)["median"]


def comparison_row(g: pd.DataFrame, horizon: str, ci_reps: int = 0) -> dict:
    s, c, e = g.spread_session.to_numpy(float), g.spread_calendar.to_numpy(float), g.annualization_effect.to_numpy(float)
    ms, mc = float(np.median(s)), float(np.median(c))
    row = dict(n_obs=len(g), unique_dates=int(g.day.nunique()), unique_expiries=int(g.expiry.nunique()),
               rv_session_median=float(g.rv_session_pct.median()), rv_calendar_median=float(g.rv_calendar_pct.median()),
               spread_session_median=ms, spread_calendar_median=mc, median_difference_calendar_minus_session=mc - ms,
               spread_session_mean=float(s.mean()), spread_calendar_mean=float(c.mean()), mean_difference_calendar_minus_session=float(c.mean() - s.mean()),
               frac_iv_gt_rv_session=float((s > 0).mean()), frac_iv_gt_rv_calendar=float((c > 0).mean()),
               annualization_effect_median=float(np.median(e)), kappa_median=float(g.kappa.median()),
               share_of_session_median_spread_explained_by_annualization=(float((ms - mc) / ms) if ms != 0 else math.nan))
    if horizon == "EXP":
        pos = g.total_variance_ratio_valid
        row.update(tv_ratio_median=float(g.loc[pos, "total_variance_ratio"].median()) if pos.any() else math.nan,
                   frac_implied_tv_gt_realized_tv=float((g.total_variance_diff > 0).mean()))
    if ci_reps and len(g) >= 20:
        for name, col in (("spread_session", "spread_session"), ("spread_calendar", "spread_calendar"), ("annualization_effect", "annualization_effect")):
            lo, hi = _med_ci(g, col, horizon, ci_reps)
            row[f"{name}_median_ci_lo"], row[f"{name}_median_ci_hi"] = lo, hi
        row["ci_method"] = f"moving block bootstrap over dates, block={block_length(horizon)}, reps={ci_reps}"
    return row


def original_vs_aligned(al: pd.DataFrame, reps: int, reps_secondary: int) -> pd.DataFrame:
    rows: List[dict] = []
    for (h, m) in TARGETS:
        base = al[(al.horizon == h) & (al.measure == m)]
        for dim, col in DIMENSIONS.items():
            groups = [("all", base)] if col is None else [(str(k), g) for k, g in base.groupby(col, observed=True)]
            for lvl, g in groups:
                if len(g) == 0:
                    continue
                ci = reps if dim in ("overall", "split") else 0
                rows.append(dict(horizon=h, measure=m, dimension=dim, level=lvl, **comparison_row(g, h, ci)))
    return pd.DataFrame(rows)


def annualization_effect(al: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, m) in TARGETS:
        base = al[(al.horizon == h) & (al.measure == m)]
        for split, g in [("all", base)] + [(k, x) for k, x in base.groupby("split")]:
            e = g.annualization_effect.to_numpy(float)
            ms, mc = float(g.spread_session.median()), float(g.spread_calendar.median())
            q = np.percentile(g.kappa, [5, 25, 50, 75, 95])
            rows.append(dict(horizon=h, measure=m, split=split, n_obs=len(g),
                             spread_session_median=ms, spread_calendar_median=mc, residual_spread_median_calendar_basis=mc,
                             annualization_effect_median_of_pairs=float(np.median(e)), annualization_effect_mean=float(e.mean()),
                             difference_of_medians=ms - mc,
                             share_explained_difference_of_medians=((ms - mc) / ms if ms else math.nan),
                             share_explained_median_effect_over_median_spread=(float(np.median(e)) / ms if ms else math.nan),
                             kappa_p05=q[0], kappa_p25=q[1], kappa_p50=q[2], kappa_p75=q[3], kappa_p95=q[4],
                             frac_kappa_lt_1=float((g.kappa < 1).mean()),
                             identity_max_abs_error=float(np.nanmax(np.abs(g.annualization_effect - (g.rv_calendar_pct * (1 - np.sqrt(g.kappa))))))))
    return pd.DataFrame(rows)


def _tv_row(g: pd.DataFrame) -> dict:
    pos = g.total_variance_ratio_valid
    r = g.loc[pos, "total_variance_ratio"]
    vol_eq = 100.0 * np.sqrt(g.implied_total_variance / g.T_years) - 100.0 * np.sqrt(g.rv_total_variance / g.T_years)       # = IV - RV on the option's T clock
    return dict(n_obs=len(g), unique_dates=int(g.day.nunique()), unique_expiries=int(g.expiry.nunique()),
                implied_tv_median=float(g.implied_total_variance.median()), realized_tv_median=float(g.rv_total_variance.median()),
                tv_diff_median=float(g.total_variance_diff.median()), tv_diff_mean=float(g.total_variance_diff.mean()),
                tv_diff_p05=float(g.total_variance_diff.quantile(.05)), tv_diff_p95=float(g.total_variance_diff.quantile(.95)),
                ratio_n_valid=int(pos.sum()), ratio_median=float(r.median()) if len(r) else math.nan,
                ratio_geometric_mean=float(np.exp(np.log(r).mean())) if len(r) else math.nan,
                ratio_p05=float(r.quantile(.05)) if len(r) else math.nan, ratio_p95=float(r.quantile(.95)) if len(r) else math.nan,
                frac_implied_gt_realized=float((g.total_variance_diff > 0).mean()),
                vol_space_equivalent_diff_median=float(vol_eq.median()), vol_space_equivalent_diff_mean=float(vol_eq.mean()),
                spread_session_median=float(g.spread_session.median()))


def expiry_total_variance(al: pd.DataFrame, reps: int, reps_secondary: int) -> pd.DataFrame:
    rows: List[dict] = []
    for m in ("hybrid", "intraday"):
        base = al[(al.horizon == "EXP") & (al.measure == m)]
        role = aa.EXP_TOTAL_VARIANCE[("EXP", m)]
        for dim, col in DIMENSIONS.items():
            if dim == "expiry_day":
                continue                                                # no expiry-day EXP target exists by construction
            groups = [("all", base)] if col is None else [(str(k), g) for k, g in base.groupby(col, observed=True)]
            for lvl, g in groups:
                row = dict(measure=m, comparability=role, dimension=dim, level=lvl, **_tv_row(g))
                if dim in ("overall", "split") and len(g) >= 20:
                    lo, hi = block_bootstrap(g.total_variance_diff.to_numpy(float), g.day.to_numpy(), {"median": np.median}, 10, reps)["median"]
                    rl, rh = block_bootstrap(np.log(g.loc[g.total_variance_ratio_valid, "total_variance_ratio"]).to_numpy(float),
                                             g.loc[g.total_variance_ratio_valid, "day"].to_numpy(), {"mean": np.mean}, 10, reps)["mean"]
                    row.update(tv_diff_median_ci_lo=lo, tv_diff_median_ci_hi=hi, ratio_geometric_mean_ci_lo=math.exp(rl), ratio_geometric_mean_ci_hi=math.exp(rh),
                               ci_method=f"moving block bootstrap over dates, block=10, reps={reps}")
                rows.append(row)
        # sensitivity: drop expiries whose T runs to 15:40 while spot stops at 15:30
        g = base[~base.expiry_after_close_change]
        rows.append(dict(measure=m, comparability=role, dimension="sensitivity", level="excluding expiries on/after 2026-08-03 (T-span mismatch)", **_tv_row(g)))
    return pd.DataFrame(rows)


def fixed_horizon_span(al: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, m) in TARGETS:
        if h == "EXP":
            continue
        base = al[(al.horizon == h) & (al.measure == m)].copy()
        base["span_bucket_days"] = base.span_days.round(1)
        for lvl, g in [("all", base)] + [(f"{k:g}", x) for k, x in base.groupby("span_bucket_days")]:
            rows.append(dict(horizon=h, measure=m, sessions=aa_sessions(h), elapsed_calendar_days=lvl, n_obs=len(g),
                             share_of_horizon=len(g) / len(base), rv_session_median=float(g.rv_session_pct.median()),
                             rv_calendar_median=float(g.rv_calendar_pct.median()), kappa_median=float(g.kappa.median()),
                             annualization_effect_median=float(g.annualization_effect.median()),
                             spread_session_median=float(g.spread_session.median()), spread_calendar_median=float(g.spread_calendar.median())))
    return pd.DataFrame(rows)


def aa_sessions(h: str) -> int:
    return {"F1": 1, "F5": 5, "F10": 10, "F20": 20}[h]


def build(stage2c: str, reps: int = 1000, reps_secondary: int = 300) -> dict:
    t0 = datetime.now()
    out = os.path.join(stage2c, "annualization_audit")
    os.makedirs(out, exist_ok=True)
    long = pd.read_csv(os.path.join(stage2c, "iv_rv_observations.csv"))
    assert (long.population == "primary").all()
    al = aa.build_aligned(long)
    al.to_csv(os.path.join(out, "aligned_observations.csv"), index=False)
    sample = al.groupby(["horizon", "measure"], group_keys=False).apply(lambda g: g.sample(min(len(g), 12), random_state=7))
    sample.to_csv(os.path.join(out, "aligned_sample_observations.csv"), index=False)
    ova = original_vs_aligned(al, reps, reps_secondary)
    eff = annualization_effect(al)
    tv = expiry_total_variance(al, reps, reps_secondary)
    fh = fixed_horizon_span(al)
    for name, df in (("original_vs_aligned", ova), ("annualization_effect", eff), ("expiry_total_variance", tv), ("fixed_horizon_calendar_span", fh)):
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    exp = al[(al.horizon == "EXP") & (al.measure == "hybrid")]
    meta = dict(source=os.path.join(stage2c, "iv_rv_observations.csv"), n_aligned_rows=len(al), n_exp_hybrid=len(exp),
                T_convention="ACT/365 seconds/(365*86400) from observation (bar start+60s) to expiry 15:30 IST (15:40 from 2026-08-03)",
                session_annualization_days=aa.SESSION_DAYS, calendar_annualization_days=aa.CALENDAR_DAYS,
                max_abs_t_minus_span_minutes_pre_change=float(exp.loc[~exp.expiry_after_close_change, "t_minus_span_minutes"].abs().max()),
                n_exp_rows_post_change=int(exp.expiry_after_close_change.sum()),
                bootstrap_reps=reps, seconds=(datetime.now() - t0).total_seconds(), pandas=pd.__version__)
    with open(os.path.join(out, "audit_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2c", default="research_output/stage2c")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args(argv)
    print(json.dumps(build(a.stage2c, a.reps), indent=1))


if __name__ == "__main__":
    main()
