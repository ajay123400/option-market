"""Stage 2C driver: Stage 2A ATM implied volatility vs SUBSEQUENT realized volatility (descriptive measurement only).

    python -m optionsengine.research.build_iv_rv --stage2a research_output/stage2a/full \
        --spot data/hist1m/NIFTY50_1m.parquet --out research_output/stage2c

Inputs are read-only: Stage 2A smiles.csv / points.csv, the spot file and calendar evidence. Stage 2A and Stage 2B logic
are imported, never modified. No signal, strategy, threshold search or P&L is produced.

Populations (never mixed): primary = forward OK + fresh + reliable strict ATM IV interpolated from reliable bracketing
strikes; loose = Stage 2A's labelled loose ATM IV where the strict one is unavailable (0 rows in this dataset for <=14 DTE).
Primary research universe: T_days <= 14 (Stage 2A strike-universe bias beyond 14 days); >14 DTE appears only in counts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time as _time
from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import iv_rv, iv_rv_stats as stats
from . import realized_vol as rv
from . import session_calendar as sc
from . import spot_quality as sq

DTE_MAX = 14.0
DTE_BUCKETS = [(0.0, 1.0, "<=1d"), (1.0, 3.0, "1-3d"), (3.0, 7.0, "3-7d"), (7.0, DTE_MAX, "7-14d")]
RES_SENSITIVE_VOLPTS = 0.5       # exploratory a-priori label: ATM tick half-width >= half of Stage 2A's 1-vol-point limit
SPLIT_DATE = "2025-01-01"
A = rv.DEFAULT_ANNUALIZATION_DAYS


def dte_bucket(t_days: float) -> str:
    for lo, hi, name in DTE_BUCKETS:
        if lo < t_days <= hi or (lo == 0.0 and t_days <= hi):
            return name
    return ">14d"


def _git() -> Optional[str]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "optionsengine"], stderr=subprocess.DEVNULL, text=True).strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return None


def atm_halfwidth(smiles_row, pts_by_smile: Dict[str, pd.DataFrame]) -> float:
    """Tick half-width of the ATM IV (vol points): the Stage 2A interpolation weights applied to the two bracketing
    points' (iv_high - iv_low)/2. NaN if the bracketing points are not in points.csv."""
    p = pts_by_smile.get(smiles_row.smile_id)
    if p is None or pd.isna(smiles_row.atm_k_low) or pd.isna(smiles_row.atm_k_high):
        return math.nan
    lo = p[p.strike == smiles_row.atm_k_low]
    hi = p[p.strike == smiles_row.atm_k_high]
    if lo.empty or hi.empty:
        return math.nan
    lo, hi = lo.iloc[0], hi.iloc[0]
    w = (0.0 - lo.log_moneyness) / (hi.log_moneyness - lo.log_moneyness)
    hw = lambda r: 0.5 * (r.iv_high - r.iv_low) * 100.0
    return float((1 - w) * hw(lo) + w * hw(hi))


def load_inputs(stage2a: str, spot: str, participant_dir: str):
    smiles = pd.read_csv(os.path.join(stage2a, "smiles.csv"))
    points = pd.read_csv(os.path.join(stage2a, "points.csv"),
                         usecols=["smile_id", "strike", "kind", "log_moneyness", "iv", "iv_low", "iv_high", "reliable",
                                  "resolution_limited", "used", "exclusion", "forward_status", "T_days", "time", "day", "split"])
    df = pd.read_parquet(spot, columns=["ts", "open", "high", "low", "close"])
    cal = sc.SessionCalendar(sc.participant_oi_dates(participant_dir))
    sessions = sq.assess(df, cal)
    daily = rv.daily_table(sessions)
    rolling = rv.rolling_table(daily)
    return smiles, points, sessions, daily, rolling


def observations(smiles: pd.DataFrame, points: pd.DataFrame, idx: iv_rv.SessionIndex, rolling: pd.DataFrame) -> pd.DataFrame:
    """Observation table (one row per snapshot-expiry observation, populations kept separate)."""
    sm = smiles.copy()
    ok = sm[(sm.forward_status == "ok") & (sm.T_days <= DTE_MAX)].copy()
    prim = ok[ok.atm_iv.notna()].assign(population="primary", iv_pct=lambda d: 100.0 * d.atm_iv)
    loose = ok[ok.atm_iv.isna() & ok.atm_loose_iv.notna()].assign(population="loose", iv_pct=lambda d: 100.0 * d.atm_loose_iv)
    obs = pd.concat([prim, loose], ignore_index=True)
    pts = {sid: g for sid, g in points[points.smile_id.isin(obs.smile_id)].groupby("smile_id")}
    r20 = rolling[(rolling.measure == "hybrid") & (rolling.window_sessions == 20) & (rolling.method == "strict")]
    r20 = {date.fromisoformat(a): b for a, b in zip(r20.session_date, r20.rv_ann_pct) if pd.notna(b)}
    obs["atm_halfwidth_volpts"] = [atm_halfwidth(r, pts) for r in obs.itertuples()]
    obs["resolution_sensitive"] = obs.atm_halfwidth_volpts >= RES_SENSITIVE_VOLPTS
    obs["dte_bucket"] = obs.T_days.map(dte_bucket)
    obs["recent_rv20_pct"] = [iv_rv.recent_rv_before(r20, idx.chain_dates, date.fromisoformat(d)) for d in obs.day]
    obs["observation_ts_ist"] = [iv_rv.observation_ts(date.fromisoformat(d), t) for d, t in zip(obs.day, obs.time)]
    # consistency of the Stage 2A spot with the spot file at the snapshot bar
    sp_mis = 0
    for r in obs.itertuples():
        s = idx.by_date.get(date.fromisoformat(r.day))
        if s is None or not np.isfinite(s.close[iv_rv.slot_of(r.time)]) or abs(s.close[iv_rv.slot_of(r.time)] - r.spot) > 1e-6:
            sp_mis += 1
    obs.attrs["spot_mismatch"] = sp_mis
    return obs


def long_table(obs: pd.DataFrame, idx: iv_rv.SessionIndex) -> pd.DataFrame:
    rows = []
    for o in obs.itertuples():
        d = date.fromisoformat(o.day)
        e = date.fromisoformat(o.expiry)
        base = dict(obs_id=o.smile_id, population=o.population, split=o.split, day=o.day, time=o.time,
                    observation_ts_ist=o.observation_ts_ist, expiry=o.expiry, expiry_day=bool(o.expiry_day), T_days=o.T_days,
                    dte_bucket=o.dte_bucket, spot=o.spot, forward=o.forward_used, iv_pct=o.iv_pct,
                    atm_halfwidth_volpts=o.atm_halfwidth_volpts, resolution_sensitive=bool(o.resolution_sensitive),
                    recent_rv20_pct=o.recent_rv20_pct,
                    # IV annualizes by CALENDAR time, RV by sessions: windows containing a weekend/holiday differ in basis
                    snapshot_to_expiry_has_weekend=bool(d.weekday() + (e - d).days >= 5))
        tasks = [(h, m, iv_rv.forward_sessions_target(idx, d, k, m)) for h, k in iv_rv.HORIZONS.items() for m in iv_rv.FIXED_MEASURES]
        tasks += [("EXP", m, iv_rv.expiry_aligned_target(idx, d, o.time, e, m)) for m in iv_rv.EXPIRY_MEASURES]
        for h, m, t in tasks:
            r = dict(base, horizon=h, measure=m, target_available=t.available, unavailable_reason=t.reason,
                     future_rv_pct=t.rv_pct, n_valid_sessions=t.n_sessions, n_returns=t.n_returns,
                     session_equivalents=t.session_equivalents, complete=t.available, target_start_ts=t.start_ts,
                     target_end_ts=t.end_ts, includes_overnight=t.includes_overnight,
                     includes_partial_first_session=t.includes_partial_first_session,
                     annualization="sqrt(252 * mean variance per session-equivalent), percent", rv_total_variance=t.total_variance)
            if t.available:
                diff = o.iv_pct - t.rv_pct
                r.update(iv_minus_rv=diff, iv_over_rv=(o.iv_pct / t.rv_pct) if t.rv_pct > 0 else math.nan,
                         ratio_valid=bool(t.rv_pct > 0), abs_err=abs(diff), sq_err=diff * diff)
                if h == "EXP":
                    ivtv = (o.iv_pct / 100.0) ** 2 * o.T_days / 365.0
                    r.update(iv_total_variance=ivtv, total_variance_diff=ivtv - t.total_variance,
                             time_basis_ratio=(o.T_days / 365.0) / (t.session_equivalents / A))
                    # same total variance annualized on the option's own CALENDAR clock (T_days/365): a convention-free alternative
                    # to the session-based annualization used everywhere else; descriptive only
                    rcal = 100.0 * math.sqrt(t.total_variance / (o.T_days / 365.0))
                    r.update(rv_calendar_basis_pct=rcal, iv_minus_rv_calendar_basis=o.iv_pct - rcal)
            rows.append(r)
    return pd.DataFrame(rows)


def _label(values: pd.Series, cuts: list, labels: list) -> pd.Series:
    """Right-closed bins (-inf, c1], (c1, c2], ... robust to tied cut-offs (pd.cut refuses non-increasing bins)."""
    v = values.to_numpy(dtype=float)
    idx = np.searchsorted(np.asarray(cuts, dtype=float), v, side="left")
    out = np.array(labels, dtype=object)[idx]
    out[~np.isfinite(v)] = None
    return pd.Series(out, index=values.index)


def add_regimes(long: pd.DataFrame, cutoffs_out: dict) -> pd.DataFrame:
    """Quartile / tercile labels. ALL cut-offs are computed on DEVELOPMENT primary data only and applied unchanged to holdout."""
    long = long.copy()
    prim_dev = long[(long.population == "primary") & (long.split == "dev")]
    uniq = prim_dev.drop_duplicates("obs_id")
    q = [float(uniq.iv_pct.quantile(p)) for p in (0.25, 0.5, 0.75)]
    cutoffs_out["iv_quartiles_pct"] = q
    long["iv_quartile"] = _label(long.iv_pct, q, ["Q1 low", "Q2", "Q3", "Q4 high"])
    rq = [float(uniq.recent_rv20_pct.dropna().quantile(p)) for p in (1 / 3, 2 / 3)]
    cutoffs_out["recent_rv20_terciles_pct"] = rq
    long["recent_rv_regime"] = _label(long.recent_rv20_pct, rq, ["low-vol", "mid-vol", "high-vol"])
    long.loc[long.recent_rv20_pct.isna(), "recent_rv_regime"] = "unknown"
    long["future_rv_quartile"] = None
    cutoffs_out["future_rv_quartiles_pct"] = {}
    for (h, m), g in long[long.target_available].groupby(["horizon", "measure"]):
        dev = g[(g.population == "primary") & (g.split == "dev")].future_rv_pct
        cuts = [float(dev.quantile(p)) for p in (0.25, 0.5, 0.75)]
        cutoffs_out["future_rv_quartiles_pct"][f"{h}|{m}"] = cuts
        sel = long.index[(long.horizon == h) & (long.measure == m) & long.target_available]
        long.loc[sel, "future_rv_quartile"] = _label(long.loc[sel, "future_rv_pct"], cuts, ["Q1 low", "Q2", "Q3", "Q4 high"])
    return long


def per_date_features(idx: iv_rv.SessionIndex, rolling: pd.DataFrame) -> pd.DataFrame:
    """Spot-only features per regular session date, used for the sampling-bias audit: recent RV known before the date and
    the subsequent hybrid F5 target (outcome, used ONLY to describe selection, never as a feature)."""
    r20 = rolling[(rolling.measure == "hybrid") & (rolling.window_sessions == 20) & (rolling.method == "strict")]
    r20 = {date.fromisoformat(a): b for a, b in zip(r20.session_date, r20.rv_ann_pct) if pd.notna(b)}
    rows = []
    for d in idx.chain_dates:
        t5 = iv_rv.forward_sessions_target(idx, d, 5, "hybrid")
        rows.append(dict(day=d.isoformat(), recent_rv20_pct=iv_rv.recent_rv_before(r20, idx.chain_dates, d),
                         f5_hybrid_rv_pct=t5.rv_pct if t5.available else math.nan))
    return pd.DataFrame(rows)


def coverage_tables(smiles: pd.DataFrame, obs_long: pd.DataFrame, feats: pd.DataFrame, points: pd.DataFrame):
    sm = smiles.merge(feats, on="day", how="left")
    sm["bucket"] = sm.T_days.map(lambda t: dte_bucket(t) if pd.notna(t) else "n/a")
    stages = [
        ("S0 all attempted snapshot-expiry groups (Stage 2A)", sm),
        ("S1 option bars and spot bar present (forward attempted)", sm[sm.forward_status.notna()]),
        ("S2 forward status OK", sm[sm.forward_status == "ok"]),
        ("S3 forward OK and <=14 calendar days to expiry", sm[(sm.forward_status == "ok") & (sm.T_days <= DTE_MAX)]),
        ("S4 primary: reliable strict ATM IV (final IV observations)", sm[(sm.forward_status == "ok") & (sm.T_days <= DTE_MAX) & sm.atm_iv.notna()]),
        ("S4b loose-only: loose ATM IV where strict unavailable", sm[(sm.forward_status == "ok") & (sm.T_days <= DTE_MAX) & sm.atm_iv.isna() & sm.atm_loose_iv.notna()]),
        ("DIAG forward OK and >14 DTE (excluded from primary statistics)", sm[(sm.forward_status == "ok") & (sm.T_days > DTE_MAX)]),
        ("DIAG forward OK, >14 DTE, strict ATM available (excluded)", sm[(sm.forward_status == "ok") & (sm.T_days > DTE_MAX) & sm.atm_iv.notna()]),
    ]
    rows = []
    for name, g in stages:
        for split_name, gg in (("all", g), ("dev", g[g.split == "dev"]), ("holdout", g[g.split == "holdout"])):
            rows.append(_composition(name, split_name, gg))
    prim = obs_long[obs_long.population == "primary"]
    for (h, m), g in prim.groupby(["horizon", "measure"]):
        a = g[g.target_available]
        for split_name, gg in (("all", a), ("dev", a[a.split == "dev"]), ("holdout", a[a.split == "holdout"])):
            tot = {"all": g, "dev": g[g.split == "dev"], "holdout": g[g.split == "holdout"]}[split_name]
            r = _composition(f"S5 final IV-vs-RV sample: {h} / {m}", split_name, gg.drop_duplicates("obs_id").rename(columns={"future_rv_pct": "f_rv"}).assign(
                recent_rv20_pct=gg.drop_duplicates("obs_id").recent_rv20_pct, f5_hybrid_rv_pct=np.nan, bucket=gg.drop_duplicates("obs_id").dte_bucket,
                expiry_day=gg.drop_duplicates("obs_id").expiry_day))
            r["share_of_primary_with_valid_target"] = round(len(gg) / max(len(tot), 1), 4)
            rows.append(r)
    return pd.DataFrame(rows)


def _composition(stage: str, split: str, g: pd.DataFrame) -> dict:
    n = len(g)
    row = dict(stage=stage, split=split, n_obs=n, unique_dates=int(g["day"].nunique()) if n else 0,
               unique_expiries=int(g["expiry"].nunique()) if n else 0)
    if n:
        row.update(share_expiry_day=round(float(g.expiry_day.astype(bool).mean()), 4), median_T_days=round(float(g.T_days.median()), 3) if "T_days" in g else None,
                   share_10_00=round(float((g.time == "10:00").mean()), 4), share_13_00=round(float((g.time == "13:00").mean()), 4),
                   share_15_00=round(float((g.time == "15:00").mean()), 4),
                   share_year_2021_22=round(float(g.day.str[:4].isin(["2021", "2022"]).mean()), 4),
                   share_year_2025_26=round(float(g.day.str[:4].isin(["2025", "2026"]).mean()), 4),
                   median_recent_rv20_pct=round(float(g.recent_rv20_pct.median()), 3) if g.recent_rv20_pct.notna().any() else None,
                   median_f5_hybrid_rv_pct=round(float(g.f5_hybrid_rv_pct.median()), 3) if "f5_hybrid_rv_pct" in g and g.f5_hybrid_rv_pct.notna().any() else None)
    return row


def resolution_tables(smiles: pd.DataFrame, points: pd.DataFrame, obs_long: pd.DataFrame):
    """(a) data lost to resolution limits among <=14 DTE, forward-OK groups; (b) descriptive table of the resolution-limited
    OTM POINTS (wing/away-from-ATM observations, NOT comparable with ATM IV; shown only as a description)."""
    ok_ids = smiles[(smiles.forward_status == "ok") & (smiles.T_days <= DTE_MAX)][["smile_id", "expiry_day", "time", "split", "T_days", "atm_iv"]]
    p = points.merge(ok_ids[["smile_id", "expiry_day", "split"]].rename(columns={"split": "split_s"}), on="smile_id", how="inner")
    fresh = p[p.iv.notna()]
    lim = fresh[fresh.resolution_limited == True]  # noqa: E712
    g = fresh.assign(res=fresh.resolution_limited.astype(bool), bucket=fresh.T_days.map(dte_bucket))
    loss = g.groupby(["expiry_day", "time"]).agg(converged_otm_points=("iv", "size"), resolution_limited_points=("res", "sum")).reset_index()
    loss["share_resolution_limited"] = (loss.resolution_limited_points / loss.converged_otm_points).round(4)
    atm = ok_ids.groupby(["expiry_day", "time"]).agg(snapshots=("smile_id", "size"), with_strict_atm_iv=("atm_iv", lambda x: int(x.notna().sum()))).reset_index()
    loss = loss.merge(atm, on=["expiry_day", "time"], how="outer")
    # descriptive table of resolution-limited points with the targets of their snapshot
    tg = obs_long[(obs_long.population == "primary") & obs_long.target_available][["obs_id", "horizon", "measure", "future_rv_pct"]]
    lp = lim.merge(tg, left_on="smile_id", right_on="obs_id", how="inner")
    lp["point_iv_pct"] = 100 * lp.iv
    lp["half_width_volpts"] = 100 * 0.5 * (lp.iv_high - lp.iv_low)
    lp["point_iv_minus_rv"] = lp.point_iv_pct - lp.future_rv_pct
    desc = lp.groupby(["expiry_day", "time", "horizon", "measure"]).agg(
        n_points=("point_iv_pct", "size"), snapshots=("smile_id", "nunique"), point_iv_median=("point_iv_pct", "median"),
        half_width_median_volpts=("half_width_volpts", "median"), half_width_p90_volpts=("half_width_volpts", lambda x: x.quantile(.9)),
        future_rv_median=("future_rv_pct", "median"), point_iv_minus_rv_median=("point_iv_minus_rv", "median")).reset_index()
    return loss, desc


def total_variance_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Expiry-aligned target compared in TOTAL variance (IV^2 * T vs the sum of squared returns): no annualization convention,
    so the calendar-time (IV) vs session-time (RV) basis difference does not enter."""
    e = long[(long.horizon == "EXP") & long.target_available & (long.population == "primary")].copy()
    rows = []
    for (split_name, measure), g in pd.concat([e.assign(split_group="all"), e.assign(split_group=e.split)]).groupby(["split_group", "measure"]):
        pos = g[g.rv_total_variance > 0]
        rows.append(dict(split_group=split_name, measure=measure, n_obs=len(g), unique_dates=int(g.day.nunique()),
                         iv_total_variance_median=float(g.iv_total_variance.median()), rv_total_variance_median=float(g.rv_total_variance.median()),
                         diff_median=float(g.total_variance_diff.median()), diff_mean=float(g.total_variance_diff.mean()),
                         ratio_median=float((pos.iv_total_variance / pos.rv_total_variance).median()) if len(pos) else math.nan,
                         frac_iv_gt_rv=float((g.total_variance_diff > 0).mean()),
                         rv_calendar_basis_median=float(g.rv_calendar_basis_pct.median()),
                         iv_minus_rv_calendar_basis_median=float(g.iv_minus_rv_calendar_basis.median()),
                         iv_minus_rv_calendar_basis_mean=float(g.iv_minus_rv_calendar_basis.mean()),
                         frac_iv_gt_rv_calendar_basis=float((g.iv_minus_rv_calendar_basis > 0).mean()),
                         time_basis_ratio_median=float(g.time_basis_ratio.median())))
    return pd.DataFrame(rows)


def dev_vs_holdout(hz: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side development vs holdout of the headline statistics, with a plain overlap flag of the two bootstrap intervals."""
    p = hz[hz.population == "primary"].set_index(["horizon", "measure", "split_group"])
    rows = []
    for (h, m) in sorted({(a, b) for a, b, _ in p.index}):
        if (h, m, "dev") not in p.index or (h, m, "holdout") not in p.index:
            continue
        d, o = p.loc[(h, m, "dev")], p.loc[(h, m, "holdout")]
        rows.append(dict(horizon=h, measure=m, n_dev=int(d.n_obs), n_holdout=int(o.n_obs),
                         diff_median_dev=d.diff_median, diff_median_holdout=o.diff_median,
                         median_ci_overlap=bool(max(d.diff_median_ci_lo, o.diff_median_ci_lo) <= min(d.diff_median_ci_hi, o.diff_median_ci_hi)),
                         frac_gt_dev=d.frac_iv_gt_rv, frac_gt_holdout=o.frac_iv_gt_rv,
                         frac_ci_overlap=bool(max(d.frac_iv_gt_rv_ci_lo, o.frac_iv_gt_rv_ci_lo) <= min(d.frac_iv_gt_rv_ci_hi, o.frac_iv_gt_rv_ci_hi)),
                         corr_spearman_dev=d.corr_spearman, corr_spearman_holdout=o.corr_spearman,
                         same_sign_of_median_diff=bool(np.sign(d.diff_median) == np.sign(o.diff_median))))
    return pd.DataFrame(rows)


def build(stage2a: str, spot: str, out: str, participant_dir: str = "data/participant_oi", reps: int = 1000,
          reps_secondary: int = 300) -> dict:
    t0 = _time.time()
    os.makedirs(out, exist_ok=True)
    smiles, points, sessions, daily, rolling = load_inputs(stage2a, spot, participant_dir)
    idx = iv_rv.SessionIndex(sessions, daily)
    obs = observations(smiles, points, idx, rolling)
    long = long_table(obs, idx)
    cutoffs: dict = {}
    long = add_regimes(long, cutoffs)
    long.to_csv(os.path.join(out, "iv_rv_observations.csv"), index=False)

    prim_av = long[(long.population == "primary") & long.target_available]
    # ---- summaries ---------------------------------------------------------------------------------------------
    sp = {"all": prim_av}
    allsplit = pd.concat([prim_av.assign(split_group="all"), prim_av.assign(split_group=prim_av.split)], ignore_index=True)
    hz = stats.summarize(allsplit.assign(population="primary"), ["population", "split_group", "horizon", "measure"], ci=True, reps=reps)
    loose_av = long[(long.population == "loose") & long.target_available]
    if len(loose_av):
        hz = pd.concat([hz, stats.summarize(loose_av.assign(split_group="all"), ["population", "split_group", "horizon", "measure"], ci=True, reps=reps)])
    hz.to_csv(os.path.join(out, "summary_by_horizon.csv"), index=False)
    dev_vs_holdout(hz).to_csv(os.path.join(out, "dev_vs_holdout.csv"), index=False)
    total_variance_summary(long).to_csv(os.path.join(out, "summary_expiry_total_variance.csv"), index=False)
    stats.summarize(allsplit, ["split_group", "dte_bucket", "horizon", "measure"], ci=True, reps=reps_secondary).to_csv(os.path.join(out, "summary_by_dte.csv"), index=False)
    stats.summarize(allsplit, ["split_group", "time", "horizon", "measure"], ci=True, reps=reps_secondary).to_csv(os.path.join(out, "summary_by_snapshot_time.csv"), index=False)
    dims = []
    for dim in ("iv_quartile", "future_rv_quartile", "recent_rv_regime", "expiry_day", "resolution_sensitive", "snapshot_to_expiry_has_weekend"):
        s = stats.summarize(allsplit, ["split_group", dim, "horizon", "measure"])
        s.insert(0, "dimension", dim)
        dims.append(s.rename(columns={dim: "level"}))
    pd.concat(dims, ignore_index=True).to_csv(os.path.join(out, "summary_by_iv_regime.csv"), index=False)

    feats = per_date_features(idx, rolling)
    cov = coverage_tables(smiles, long, feats, points)
    cov.to_csv(os.path.join(out, "data_coverage.csv"), index=False)
    avail = long.groupby(["population", "horizon", "measure", "target_available", "unavailable_reason"], dropna=False).size().reset_index(name="n_obs")
    avail.to_csv(os.path.join(out, "target_availability.csv"), index=False)
    loss, desc = resolution_tables(smiles, points, long)
    loss.to_csv(os.path.join(out, "resolution_data_loss.csv"), index=False)
    desc.to_csv(os.path.join(out, "summary_resolution_limited.csv"), index=False)
    # small committable sample
    keep = long[(long.population == "primary")].drop_duplicates(["obs_id", "horizon", "measure"])
    samp = keep[(keep.measure == "hybrid") | (keep.horizon == "EXP")]
    samp = samp.sample(min(120, len(samp)), random_state=3).sort_values(["day", "time", "horizon"])
    samp.to_csv(os.path.join(out, "sample_observations.csv"), index=False)
    with open(os.path.join(out, "regime_cutoffs.json"), "w") as f:
        json.dump(cutoffs, f, indent=1)
    meta = dict(git_commit=_git(), stage2a_dir=stage2a, spot=spot, n_smile_rows=len(smiles), n_observations=len(obs),
                n_primary=int((obs.population == "primary").sum()), n_loose=int((obs.population == "loose").sum()),
                n_long_rows=len(long), spot_mismatch_rows=obs.attrs.get("spot_mismatch"), dte_max=DTE_MAX, split_date=SPLIT_DATE,
                annualization_days=A, resolution_sensitive_volpts=RES_SENSITIVE_VOLPTS, bootstrap_reps=reps, bootstrap_reps_secondary=reps_secondary,
                seconds=round(_time.time() - t0, 1), started=datetime.fromtimestamp(t0).isoformat(), pandas=pd.__version__)
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2a", default="research_output/stage2a/full")
    ap.add_argument("--spot", default="data/hist1m/NIFTY50_1m.parquet")
    ap.add_argument("--out", default="research_output/stage2c")
    ap.add_argument("--participant-dir", default="data/participant_oi")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-reps-secondary", type=int, default=300, help="replications for the DTE / snapshot-time tables")
    a = ap.parse_args(argv)
    meta = build(a.stage2a, a.spot, a.out, a.participant_dir, a.bootstrap_reps, a.bootstrap_reps_secondary)
    print(json.dumps({k: meta[k] for k in ("n_observations", "n_primary", "n_loose", "n_long_rows", "spot_mismatch_rows", "seconds", "git_commit")}))


if __name__ == "__main__":
    main()
