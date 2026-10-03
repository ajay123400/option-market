"""Dev-vs-holdout contrasts (Stage 2D, T5): block-bootstrap differences, post-stratification on the IV quartile, Holm adjustment.

Design (frozen in research_output/stage2d/gate2/PREREGISTRATION.md): clusters = expiries within a split (an expiry straddling the split date contributes one
cluster to each split); the expiry moving-block bootstrap runs INDEPENDENTLY in the two splits (adjacent at the boundary: ignored); the difference is
holdout - dev. P-values are percentile-bootstrap two-sided p = 2 min(P(d* <= 0), P(d* >= 0)) (floored at 1/(R+1)) -- approximate, descriptive.
The holdout has already been displayed in Stage 2C and Gate 1: results are 'previously viewed', never 'out-of-sample'.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import clusters as C, estimands as E


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def boot_p(diff_draws: np.ndarray) -> float:
    d = diff_draws[np.isfinite(diff_draws)]
    r = len(d)
    lo = (np.sum(d <= 0) + 1) / (r + 1)
    hi = (np.sum(d >= 0) + 1) / (r + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def straddling_expiries(df: pd.DataFrame, split_col: str = "split") -> List[str]:
    n = df.groupby("expiry")[split_col].nunique()
    return sorted(n[n > 1].index)


def split_contrast(dev: pd.DataFrame, hold: pd.DataFrame, block: int = 5, reps: int = 5000, seed: int = 20260101) -> pd.DataFrame:
    """Holdout - dev for the nine Gate 1 estimands with independent expiry-block bootstraps in the two splits."""
    ed, eh = E.build(dev), E.build(hold)
    td, th = ed.point(), eh.point()
    dd = C.bootstrap(ed.cs, ed.evaluate, "moving_block", block, reps, seed, k=len(E.EST_NAMES))
    dh = C.bootstrap(eh.cs, eh.evaluate, "moving_block", block, reps, seed + 1, k=len(E.EST_NAMES))
    diff = dh - dd
    lo, hi = C.percentile_ci(diff)
    rows = []
    for i, n in enumerate(E.EST_NAMES):
        rows.append(dict(estimand=n, dev_estimate=td[i], holdout_estimate=th[i], difference_holdout_minus_dev=th[i] - td[i], ci_lo=lo[i], ci_hi=hi[i],
                         p_two_sided_bootstrap=boot_p(diff[:, i]), dev_clusters=ed.cs.n_clusters, holdout_clusters=eh.cs.n_clusters,
                         dev_ci_lo=np.nanpercentile(dd[:, i], 2.5), dev_ci_hi=np.nanpercentile(dd[:, i], 97.5),
                         holdout_ci_lo=np.nanpercentile(dh[:, i], 2.5), holdout_ci_hi=np.nanpercentile(dh[:, i], 97.5)))
    out = pd.DataFrame(rows)
    out["block"] = block
    out["reps"] = reps
    return out


# ------------------------------------------------------------------------------ post-stratification on the IV quartile
def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    return float(x[np.searchsorted(cw, 0.5 * cw[-1])])


PS_NAMES = ["S1_calendar_ps", "S1_session_ps", "R1_geometric_mean_ps"]


class PostStratified:
    """Estimates re-weighted so that the IV-quartile composition equals `target` (shares by level code), recomputed inside every bootstrap draw."""

    def __init__(self, cs: C.ClusterSet, n_levels: int, target: np.ndarray):
        self.cs = cs
        self.q = cs.cols["q"].astype(int)
        self.sc, self.ss, self.loglr, self.valid = cs.cols["sc"], cs.cols["ss"], cs.cols["loglr"], cs.cols["valid"]
        self.n_levels = n_levels
        self.target = np.asarray(target, float)

    def evaluate(self, rows: np.ndarray, seq: np.ndarray) -> np.ndarray:
        q = self.q[rows]
        cnt = np.bincount(q, minlength=self.n_levels).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            wl = np.where(cnt > 0, self.target / cnt, 0.0)
        w = wl[q]
        out = np.full(3, np.nan)
        out[0] = weighted_median(self.sc[rows], w)
        out[1] = weighted_median(self.ss[rows], w)
        ok = self.valid[rows]
        if ok.any():
            out[2] = math.exp(float(np.sum(w[ok] * self.loglr[rows][ok]) / np.sum(w[ok])))
        return out

    def point(self) -> np.ndarray:
        seq = np.arange(self.cs.n_clusters)
        return self.evaluate(self.cs.rows_for(seq), seq)


def _ps_clusterset(df: pd.DataFrame, levels: Sequence[str], level_col: str) -> C.ClusterSet:
    cols = E.cluster_columns(df)
    cols["q"] = df[level_col].map({l: i for i, l in enumerate(levels)}).to_numpy()
    return C.ClusterSet(df.expiry.to_numpy(), cols)


def post_stratified_contrast(dev: pd.DataFrame, hold: pd.DataFrame, level_col: str = "iv_quartile", block: int = 5, reps: int = 5000, seed: int = 20260101):
    """Holdout re-weighted to the DEV composition (fixed at the dev point estimate) minus dev (same weights: identity at the point estimate)."""
    levels = sorted(set(dev[level_col]) | set(hold[level_col]))
    dev_share = np.array([(dev[level_col] == l).mean() for l in levels])
    cd, ch = _ps_clusterset(dev, levels, level_col), _ps_clusterset(hold, levels, level_col)
    pd_, ph_ = PostStratified(cd, len(levels), dev_share), PostStratified(ch, len(levels), dev_share)
    dd = C.bootstrap(cd, pd_.evaluate, "moving_block", block, reps, seed + 20, k=3)
    dh = C.bootstrap(ch, ph_.evaluate, "moving_block", block, reps, seed + 21, k=3)
    diff = dh - dd
    lo, hi = C.percentile_ci(diff)
    td, th = pd_.point(), ph_.point()
    rows = [dict(estimand=n, dev_estimate=td[i], holdout_reweighted_to_dev_composition=th[i], difference=th[i] - td[i], ci_lo=lo[i], ci_hi=hi[i],
                 p_two_sided_bootstrap=boot_p(diff[:, i]), weights_fixed_at="dev point composition", block=block, reps=reps) for i, n in enumerate(PS_NAMES)]
    comp = pd.DataFrame(dict(level=levels, dev_share=dev_share, holdout_share=[(hold[level_col] == l).mean() for l in levels]))
    return pd.DataFrame(rows), comp


def within_level_contrasts(dev: pd.DataFrame, hold: pd.DataFrame, level_col: str = "iv_quartile", block: int = 5, reps: int = 2000, seed: int = 20260101) -> pd.DataFrame:
    """Within each level: dev and holdout S1_cal / R1 with their own intervals and the difference (point only when a split has < 30 expiries in the level)."""
    rows = []
    for j, l in enumerate(sorted(set(dev[level_col]) | set(hold[level_col]))):
        d, h = dev[dev[level_col] == l], hold[hold[level_col] == l]
        row = dict(level=l, dev_rows=len(d), holdout_rows=len(h), dev_expiries=d.expiry.nunique(), holdout_expiries=h.expiry.nunique())
        try:
            ed, eh = E.build(d), E.build(h)
            dd = C.bootstrap(ed.cs, ed.evaluate, "moving_block", block, reps, seed + 40 + j, k=9)
            dh = C.bootstrap(eh.cs, eh.evaluate, "moving_block", block, reps, seed + 60 + j, k=9)
            diff = dh - dd
            lo, hi = C.percentile_ci(diff)
            for name, i in (("S1_calendar", 2), ("R1_geometric_mean", 4)):
                row.update({f"{name}_dev": ed.point()[i], f"{name}_holdout": eh.point()[i], f"{name}_diff": eh.point()[i] - ed.point()[i],
                            f"{name}_diff_ci_lo": lo[i], f"{name}_diff_ci_hi": hi[i]})
        except C.ClusterError as ex:
            ed_ = E.build(d, min_clusters=1) if len(d) else None
            eh_ = E.build(h, min_clusters=1) if len(h) else None
            for name, i in (("S1_calendar", 2), ("R1_geometric_mean", 4)):
                a = ed_.point()[i] if ed_ else np.nan
                b = eh_.point()[i] if eh_ else np.nan
                row.update({f"{name}_dev": a, f"{name}_holdout": b, f"{name}_diff": b - a, f"{name}_diff_ci_lo": np.nan, f"{name}_diff_ci_hi": np.nan})
            row["note"] = f"point estimates only: {ex}"
        rows.append(row)
    return pd.DataFrame(rows)
