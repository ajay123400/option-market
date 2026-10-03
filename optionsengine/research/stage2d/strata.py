"""Strata estimates, joint-bootstrap heterogeneity tests and a descriptive Mincer-Zarnowitz regression (Stage 2D, T6).

All resampling is over WHOLE EXPIRIES (clusters) in calendar order with circular moving blocks, applied to the full sample; a stratum's estimate in a draw uses
only that stratum's rows of the drawn expiries. Because all strata of a dimension are re-estimated from the SAME draws, the bootstrap covariance between strata
(strata share expiries for DTE / time / IV quartile / regime / weekday dimensions) is respected. Strata with fewer than MIN_CLUSTERS expiries get point estimates only.

Heterogeneity test (H0: all strata share one value of the estimand): contrasts d = theta_l - theta_ref for the other levels; Wald W = d' S^+ d with S the bootstrap
covariance of the contrasts; the null distribution is the bootstrap distribution of the CENTRED contrasts, p = (1 + #{W* >= W}) / (1 + R). Descriptive; levels
that are not regimes ('unknown') are excluded by the caller.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import clusters as C, estimands as E

STAT_NAMES = ["S1_calendar", "S1_session", "R1_geometric_mean", "R2_median_ratio"]


class Aligned:
    """Rows sorted by expiry; every stratum view shares the same cluster index space (clusters may be empty in a view)."""

    def __init__(self, df: pd.DataFrame, extra: Optional[Dict[str, np.ndarray]] = None, min_clusters: int = C.MIN_CLUSTERS):
        cols = E.cluster_columns(df)
        if extra:
            cols.update(extra)
        labels = df.expiry.to_numpy()
        self.labels, codes = np.unique(labels, return_inverse=True)
        self.n_clusters = len(self.labels)
        if self.n_clusters < min_clusters:
            raise C.ClusterError(f"only {self.n_clusters} clusters (< {min_clusters})")
        order = np.argsort(codes, kind="stable")
        self.codes = codes[order]
        self.cols = {k: np.asarray(v)[order] for k, v in cols.items()}
        self.df = df.iloc[order].reset_index(drop=True)
        self.min_clusters = min_clusters

    def view(self, mask: np.ndarray) -> "View":
        return View(self, np.asarray(mask, bool))


class View:
    def __init__(self, al: Aligned, mask: np.ndarray):
        self.al = al
        self.cnt = np.bincount(al.codes[mask], minlength=al.n_clusters)
        self.off = np.concatenate([[0], np.cumsum(self.cnt)[:-1]]).astype(np.int64)
        self.cols = {k: v[mask] for k, v in al.cols.items()}
        self.n_clusters_nonempty = int((self.cnt > 0).sum())
        self.n_rows = int(mask.sum())

    def rows_for(self, seq: np.ndarray) -> np.ndarray:
        ln = self.cnt[seq]
        tot = int(ln.sum())
        return np.repeat(self.off[seq] - np.cumsum(ln) + ln, ln) + np.arange(tot)

    def stats(self, seq: np.ndarray) -> np.ndarray:
        rows = self.rows_for(seq)
        out = np.full(len(STAT_NAMES), np.nan)
        if len(rows) == 0:
            return out
        out[0] = np.median(self.cols["sc"][rows])
        out[1] = np.median(self.cols["ss"][rows])
        ok = self.cols["valid"][rows]
        if ok.any():
            out[2] = math.exp(float(self.cols["loglr"][rows][ok].mean()))
            out[3] = float(np.median(self.cols["ratio"][rows][ok]))
        return out


def joint_draws(al: Aligned, views: Sequence[View], block: int, reps: int, seed: int) -> np.ndarray:
    """reps x L x 4 array of stratum statistics from the SAME cluster resamples."""
    rng = np.random.default_rng(seed)
    out = np.empty((reps, len(views), len(STAT_NAMES)))
    for r in range(reps):
        seq = C.draw_clusters("moving_block", rng, al.n_clusters, block)
        for j, v in enumerate(views):
            out[r, j] = v.stats(seq)
    return out


def point_stats(view: View) -> np.ndarray:
    return view.stats(np.arange(view.al.n_clusters))


def strata_table(al: Aligned, dim: str, levels: Sequence, masks: Sequence[np.ndarray], block: int = 5, reps: int = 5000, seed: int = 20260101,
                 min_clusters: int = C.MIN_CLUSTERS) -> Tuple[pd.DataFrame, np.ndarray, List[View]]:
    views = [al.view(m) for m in masks]
    draws = joint_draws(al, views, block, reps, seed)
    rows = []
    for j, (lvl, v) in enumerate(zip(levels, views)):
        pt = point_stats(v)
        ok = v.n_clusters_nonempty >= min_clusters
        for i, n in enumerate(STAT_NAMES):
            d = draws[:, j, i]
            lo, hi = (np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)) if ok else (np.nan, np.nan)
            rows.append(dict(dimension=dim, level=str(lvl), statistic=n, estimate=pt[i], ci_lo=lo, ci_hi=hi, n_rows=v.n_rows, n_expiries=v.n_clusters_nonempty,
                             ci_note=("" if ok else f"point only: {v.n_clusters_nonempty} expiries < {min_clusters}"), block=block, reps=reps))
    return pd.DataFrame(rows), draws, views


def wald_heterogeneity(draws: np.ndarray, point: np.ndarray, stat_index: int, usable: Sequence[int]) -> Dict[str, float]:
    """Bootstrap Wald test of equal strata for statistic `stat_index` over the usable level indices (>= 2)."""
    usable = list(usable)
    if len(usable) < 2:
        return dict(wald=np.nan, df=0, p_value=np.nan, n_levels=len(usable), note="fewer than 2 usable levels")
    ref, others = usable[0], usable[1:]
    d_pt = np.array([point[j, stat_index] - point[ref, stat_index] for j in others])
    D = np.stack([draws[:, j, stat_index] - draws[:, ref, stat_index] for j in others], axis=1)
    good = np.isfinite(D).all(axis=1)
    D = D[good]
    mu = D.mean(axis=0)
    S = np.atleast_2d(np.cov(D, rowvar=False))
    Sp = np.linalg.pinv(S)
    W = float(d_pt @ Sp @ d_pt)
    Dc = D - mu
    Ws = np.einsum("ri,ij,rj->r", Dc, Sp, Dc)
    p = float((1 + np.sum(Ws >= W - 1e-12)) / (1 + len(Ws)))
    return dict(wald=W, df=len(others), p_value=p, n_levels=len(usable), n_valid_draws=int(good.sum()), note="")


def heterogeneity_table(dim: str, levels: Sequence, draws: np.ndarray, views: Sequence[View], min_clusters: int = C.MIN_CLUSTERS,
                        stat_names: Sequence[str] = ("S1_calendar", "R1_geometric_mean"), exclude_levels: Sequence = ()) -> pd.DataFrame:
    usable = [j for j, (l, v) in enumerate(zip(levels, views)) if v.n_clusters_nonempty >= min_clusters and l not in exclude_levels]
    point = np.stack([point_stats(v) for v in views])
    rows = []
    for n in stat_names:
        r = wald_heterogeneity(draws, point, STAT_NAMES.index(n), usable)
        rows.append(dict(dimension=dim, statistic=n, usable_levels=";".join(str(levels[j]) for j in usable), **r))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ Mincer-Zarnowitz (descriptive)
def ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx == 0:
        return math.nan, math.nan
    b = float(((x - xm) * (y - ym)).sum() / sxx)
    return float(ym - b * xm), b


def mincer_zarnowitz(df: pd.DataFrame, x_col: str = "iv_pct", y_col: str = "rv_calendar_pct", log: bool = False, block: int = 5, reps: int = 2000,
                     seed: int = 20260101, min_clusters: int = C.MIN_CLUSTERS) -> Dict[str, float]:
    """RV = a + b IV over rows (optionally on logs); CI by whole-expiry block bootstrap. Errors-in-variables (noisy IV) attenuates b: caveat, not corrected."""
    d = df[[x_col, y_col, "expiry"]].dropna()
    if log:
        d = d[(d[x_col] > 0) & (d[y_col] > 0)]
        d = d.assign(**{x_col: np.log(d[x_col]), y_col: np.log(d[y_col])})
    cs = C.ClusterSet(d.expiry.to_numpy(), {"x": d[x_col].to_numpy(float), "y": d[y_col].to_numpy(float)}, min_clusters)

    def ev(rows, seq):
        a, b = ols(cs.cols["x"][rows], cs.cols["y"][rows])
        return np.array([a, b])

    seq = np.arange(cs.n_clusters)
    a0, b0 = ev(cs.rows_for(seq), seq)
    dr = C.bootstrap(cs, ev, "moving_block", block, reps, seed, k=2)
    lo, hi = C.percentile_ci(dr)
    return dict(n_rows=len(d), n_expiries=cs.n_clusters, intercept=a0, slope=b0, intercept_ci_lo=lo[0], intercept_ci_hi=hi[0], slope_ci_lo=lo[1], slope_ci_hi=hi[1],
                scale="log" if log else "level", block=block, reps=reps)
