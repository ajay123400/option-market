"""Estimands for the expiry-aligned sample (Stage 2D, T1/T3). Each is reported separately; none is preferred.

Spread (vol points), for the session basis and the calendar basis:
  S1  all-observation median of spread                      (the Stage 2C statistic; rows with many snapshots weigh more)
  S2  median of per-expiry medians of spread                (each expiry counts once)
Total-variance ratio r = W_imp / V over rows with V > 0:
  R1  geometric mean of r over all rows                     exp(mean ln r)
  R1e expiry-weighted geometric mean                        exp(mean over expiries of the per-expiry mean ln r)
  R2  median of r over all rows
  R2e median of the per-expiry median r
  R3  ratio of summed total variances                       sum(W_imp) / sum(V) over all rows
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .clusters import ClusterSet

EST_NAMES = [
    "S1_session_all_obs_median",
    "S2_session_median_of_expiry_medians",
    "S1_calendar_all_obs_median",
    "S2_calendar_median_of_expiry_medians",
    "R1_geometric_mean_all_obs",
    "R1e_geometric_mean_expiry_weighted",
    "R2_median_ratio_all_obs",
    "R2e_median_of_expiry_median_ratios",
    "R3_ratio_of_summed_total_variance",
]
CLUSTER_LEVEL = {1, 3, 5, 7}                                      # indices that need per-cluster aggregates
REQUIRED = ["spread_session", "spread_calendar", "implied_total_variance", "rv_total_variance", "total_variance_ratio_valid"]


def cluster_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    for c in REQUIRED:
        if c not in df.columns:
            raise KeyError(f"missing column {c}")
    valid = df["total_variance_ratio_valid"].to_numpy(dtype=bool) & (df.rv_total_variance.to_numpy(float) > 0)
    w = df.implied_total_variance.to_numpy(float)
    v = df.rv_total_variance.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, w / np.where(valid, v, 1.0), np.nan)
    return dict(ss=df.spread_session.to_numpy(float), sc=df.spread_calendar.to_numpy(float), valid=valid,
                ratio=ratio, loglr=np.where(valid, np.log(np.where(valid, ratio, 1.0)), np.nan), w=w, v=v)


class Estimands:
    """Evaluator bound to a ClusterSet. `evaluate(rows, seq)` returns the 9 estimands for a resample (rows: sorted-row indices, seq: cluster indices)."""

    def __init__(self, cs: ClusterSet, cluster_level: bool = True):
        """cluster_level=False (row-level or snapshot-date clusters) leaves the per-expiry estimands S2/R1e/R2e undefined (NaN)."""
        self.cs = cs
        c = cs.cols
        self.ss, self.sc, self.valid, self.ratio, self.loglr, self.w, self.v = (c[k] for k in ("ss", "sc", "valid", "ratio", "loglr", "w", "v"))
        if cs.row_level or not cluster_level:
            self.cl = None
            return
        n = cs.n_clusters
        self.cl_med_s = np.full(n, np.nan); self.cl_med_c = np.full(n, np.nan); self.cl_mean_lr = np.full(n, np.nan); self.cl_med_r = np.full(n, np.nan)
        for i in range(n):
            sl = slice(cs.offsets[i], cs.offsets[i] + cs.counts[i])
            self.cl_med_s[i] = np.median(self.ss[sl]); self.cl_med_c[i] = np.median(self.sc[sl])
            ok = self.valid[sl]
            if ok.any():
                self.cl_mean_lr[i] = self.loglr[sl][ok].mean(); self.cl_med_r[i] = np.median(self.ratio[sl][ok])
        self.cl = True

    def evaluate(self, rows: np.ndarray, seq: np.ndarray) -> np.ndarray:
        out = np.full(len(EST_NAMES), np.nan)
        out[0] = np.median(self.ss[rows])
        out[2] = np.median(self.sc[rows])
        ok = self.valid[rows]
        if ok.any():
            out[4] = math.exp(self.loglr[rows][ok].mean())
            out[6] = np.median(self.ratio[rows][ok])
        out[8] = self.w[rows].sum() / self.v[rows].sum()
        if self.cl:
            out[1] = np.nanmedian(self.cl_med_s[seq]); out[3] = np.nanmedian(self.cl_med_c[seq])
            lr = self.cl_mean_lr[seq]
            if np.isfinite(lr).any():
                out[5] = math.exp(np.nanmean(lr))
            out[7] = np.nanmedian(self.cl_med_r[seq]) if np.isfinite(self.cl_med_r[seq]).any() else np.nan
        return out

    def point(self) -> np.ndarray:
        seq = np.arange(self.cs.n_clusters)
        return self.evaluate(self.cs.rows_for(seq), seq)

    def per_cluster_table(self) -> pd.DataFrame:
        return pd.DataFrame(dict(label=self.cs.labels, n_rows=self.cs.counts, median_spread_session=self.cl_med_s, median_spread_calendar=self.cl_med_c,
                                 mean_log_ratio=self.cl_mean_lr, median_ratio=self.cl_med_r))


def build(df: pd.DataFrame, cluster_col: str = "expiry", min_clusters: int = 30, row_level: bool = False) -> Estimands:
    """Estimands bound to clusters = `cluster_col`. The per-expiry estimands exist only when cluster_col is 'expiry'."""
    cols = cluster_columns(df)
    labels = np.arange(len(df)) if row_level else df[cluster_col].to_numpy()
    if row_level:
        labels = np.char.zfill(labels.astype(str), 9)
    return Estimands(ClusterSet(labels, cols, min_clusters, row_level), cluster_level=(cluster_col == "expiry" and not row_level))
