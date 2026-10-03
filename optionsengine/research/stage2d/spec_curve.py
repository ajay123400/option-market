"""Specification curve for the expiry-aligned findings (Stage 2D, T7/T8). One row per (specification, estimand): estimate and expiry-block interval.

No hypothesis tests and no selection: every specification listed in the pre-registration is reported. Variant pipelines (risk-free rate, forward gate, stale-quote age,
DTE cap) re-run the committed Stage 2A/2C/audit builders into a Stage 2D directory (cached; no existing artifact is touched); arithmetic/RV variants transform the
stored baseline columns or use `alt_rv`.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import build_annualization_audit as baa, build_iv_rv as biv
from . import alt_rv, clusters as C, estimands as E

NULLS = {n: (0.0 if n.startswith("S") else 1.0) for n in E.EST_NAMES}
BASELINE = "baseline"


@dataclass
class Spec:
    name: str
    dimension: str
    value: str
    df: pd.DataFrame
    note: str = ""


def estimate_spec(spec: Spec, block: int = 5, reps: int = 5000, seed: int = 20260101) -> pd.DataFrame:
    est = E.build(spec.df)
    th = est.point()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dr = C.bootstrap(est.cs, est.evaluate, "moving_block", block, reps, seed, k=len(E.EST_NAMES))
        lo, hi = C.percentile_ci(dr)
    rows = []
    for i, n in enumerate(E.EST_NAMES):
        ok = np.isfinite(lo[i]) and np.isfinite(hi[i])
        rows.append(dict(spec=spec.name, dimension=spec.dimension, value=spec.value, note=spec.note, estimand=n, estimate=th[i], ci_lo=lo[i], ci_hi=hi[i],
                         null_value=NULLS[n], null_inside=(bool(lo[i] <= NULLS[n] <= hi[i]) if ok else None), n_rows=len(spec.df), n_expiries=est.cs.n_clusters,
                         block=block, reps=reps))
    return pd.DataFrame(rows)


def exp_hybrid(aligned: pd.DataFrame, measure: str = "hybrid") -> pd.DataFrame:
    return aligned[(aligned.horizon == "EXP") & (aligned.measure == measure)].reset_index(drop=True)


# ------------------------------------------------------------------------------ variant pipelines (cached)
def _run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True, stdout=subprocess.DEVNULL)


def stage2a_rate_variant(root: str, out_dir: str, rate: float, workers: int = 4) -> str:
    if not os.path.exists(os.path.join(out_dir, "smiles.csv")):
        if "stage2d" not in os.path.normpath(out_dir).split(os.sep):
            raise ValueError("variant outputs must live under a stage2d directory")
        _run([sys.executable, "-m", "optionsengine.research.build_surface", "--root", root, "--out", out_dir, "--rate", str(rate), "--workers", str(workers)])
    return out_dir


def aligned_from_stage2a(stage2a_dir: str, out_dir: str, spot: str, participant_dir: str, dte_max: Optional[float] = None) -> pd.DataFrame:
    """Stage 2C builder + annualization audit on a Stage 2A directory, written under out_dir (a stage2d path). Cached."""
    if "stage2d" not in os.path.normpath(out_dir).split(os.sep):
        raise ValueError("variant outputs must live under a stage2d directory")
    path = os.path.join(out_dir, "annualization_audit", "aligned_observations.csv")
    if not os.path.exists(path):
        old = biv.DTE_MAX
        try:
            if dte_max is not None:
                biv.DTE_MAX = dte_max                                   # runtime override only; no file edited
            biv.build(stage2a_dir, spot, out_dir, participant_dir, reps=20, reps_secondary=20)
        finally:
            biv.DTE_MAX = old
        baa.build(out_dir, reps=20, reps_secondary=20)
    return pd.read_csv(path)


# ------------------------------------------------------------------------------ derived specifications
def with_sampling(df: pd.DataFrame, W: alt_rv.WindowVariance, k: int, weighting: str = "uniform") -> pd.DataFrame:
    V, S = [], []
    for r in df.itertuples():
        d, e = date.fromisoformat(r.day), date.fromisoformat(r.expiry)
        v = W.variance(d, r.time, e, k)
        s = W.session_equivalents(d, r.time, e, weighting)
        V.append(np.nan if v is None else v)
        S.append(np.nan if s is None else s)
    V, S = np.array(V), np.array(S)
    keep = np.isfinite(V) & np.isfinite(S)
    return alt_rv.recompute(df[keep].reset_index(drop=True), V[keep], S[keep])


def build_specs(base_all: pd.DataFrame, W: alt_rv.WindowVariance, variants: Dict[str, pd.DataFrame], rates: Dict[str, pd.DataFrame]) -> List[Spec]:
    """base_all = full aligned table of the baseline (EXP hybrid + intraday rows). variants: name -> aligned EXP/hybrid df. rates: '5.5'/'6.5'/'7.5' -> aligned EXP/hybrid df."""
    base = exp_hybrid(base_all)
    specs = [Spec(BASELINE, "baseline", "r=6.5%, A=252, 365, 1-min, uniform, <=14 DTE, 5 bps, 5 min", base)]
    # --- T7 arithmetic
    for A in (250, 256):
        specs.append(Spec(f"session_days_{A}", "annualization_session_days", str(A), alt_rv.recompute(base, session_days=float(A)), "S1_session scales by sqrt(A/252); calendar basis and ratios unchanged"))
    specs.append(Spec("calendar_days_365.25", "annualization_calendar_days", "365.25", alt_rv.recompute(base, calendar_days=365.25), "IV rescaled consistently (T uses 365.25); IV^2*T invariant"))
    # --- T7 RV sampling and partial-session weighting
    if W is not None:
        for k in (5, 15):
            specs.append(Spec(f"rv_sampling_{k}min", "rv_sampling", f"{k}-minute", with_sampling(base, W, k)))
        specs.append(Spec("partial_session_profile", "partial_session_weighting", "dev intraday variance profile", with_sampling(base, W, 1, "profile"), "affects the session-basis RV only"))
    intraday = exp_hybrid(base_all, "intraday")
    if len(intraday):
        specs.append(Spec("rv_intraday_only", "rv_measure", "intraday-only (subset, overnight excluded)", intraday, "not interval-complete: calendar-basis RV undefined"))
    # --- T7 rate
    for r in ("5.5", "7.5"):
        if r in rates:
            specs.append(Spec(f"rate_{r}pct", "risk_free_rate", f"{r}%", rates[r]))
    # --- T7 factorial rate x sampling
    for r in ("5.5", "6.5", "7.5"):
        src = base if r == "6.5" else rates.get(r)
        if src is None:
            continue
        for k in (1, 5, 15):
            if k > 1 and W is None:
                continue
            d = src if k == 1 else with_sampling(src, W, k)
            specs.append(Spec(f"factorial_rate{r}_k{k}", "factorial_rate_x_sampling", f"r={r}%, {k}-min", d))
    # --- T8 filters via re-run pipelines
    for name, dim, val in (("fwd_3bps", "forward_gate", "3 bps"), ("fwd_8bps", "forward_gate", "8 bps"), ("fwd_12bps", "forward_gate", "12 bps"),
                           ("age_2min", "stale_quote_age", "2 min"), ("age_15min", "stale_quote_age", "15 min"), ("dte_le21", "dte_cap", "<= 21 d (biased strike coverage beyond 14 d)")):
        if name in variants:
            specs.append(Spec(name, dim, val, variants[name], "strike coverage beyond 14 DTE is biased (Stage 2A)" if name == "dte_le21" else ""))
    # --- T8 filters via row selection on the baseline
    specs.append(Spec("dte_le7", "dte_cap", "<= 7 d", base[base.T_days <= 7].reset_index(drop=True)))
    for y in ("2021", "2026"):
        specs.append(Spec(f"drop_expiry_year_{y}", "exclude_year", y, base[~base.expiry.str.startswith(y)].reset_index(drop=True)))
    for t in ("10:00", "13:00", "15:00"):
        specs.append(Spec(f"drop_time_{t.replace(':', '')}", "exclude_snapshot_time", t, base[base.time != t].reset_index(drop=True)))
    return specs


def robustness_reading(curve: pd.DataFrame, estimands: Sequence[str] = ("S1_calendar_all_obs_median", "R1_geometric_mean_all_obs")) -> pd.DataFrame:
    """Pre-declared descriptive summary: min/max estimate over all specifications, share excluding the null, sign check, widest departure."""
    rows = []
    for n in estimands:
        c = curve[(curve.estimand == n) & curve.estimate.notna()]
        base = float(c[c.spec == BASELINE].estimate.iloc[0])
        null = NULLS[n]
        sign_ok = bool(((c.estimate - null) * (base - null) > 0).all())
        dep = (c.estimate - base).abs()
        worst = c.loc[dep.idxmax()]
        known = c[c.null_inside.notna()]
        rows.append(dict(estimand=n, n_specs=len(c), baseline=base, min_estimate=float(c.estimate.min()), max_estimate=float(c.estimate.max()),
                         share_intervals_excluding_null=float((~known.null_inside.astype(bool)).mean()), sign_same_as_baseline_in_every_spec=sign_ok,
                         widest_departure_spec=worst.spec, widest_departure=float(worst.estimate - base),
                         reading=("not sensitive to the tested conventions (sign unchanged in every specification)" if sign_ok else "sign changes in some specification: see curve")))
    return pd.DataFrame(rows)
