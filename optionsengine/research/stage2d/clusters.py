"""Cluster / block resampling machinery (Stage 2D, T1).

Rows are grouped into ORDERED clusters (expiries, or snapshot dates). A bootstrap replicate is a sequence of cluster indices; all rows of a drawn
cluster are kept together, so the within-cluster dependence (all snapshots of one expiry share one realized path) is preserved.

Schemes (cluster order = ascending label, i.e. calendar order for ISO date labels):
  iid           clusters drawn independently with replacement
  moving_block  circular moving blocks of `block` consecutive clusters (Kunsch 1989 / Liu-Singh 1992), truncated to n clusters
  stationary    circular blocks of geometric length with mean `block` (Politis-Romano 1994)

Failure policy: fewer than MIN_CLUSTERS clusters, a block longer than the number of clusters, or a block < 1 raises ClusterError; no interval is
ever returned from a degenerate design.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

MIN_CLUSTERS = 30
SCHEMES = ("iid", "moving_block", "stationary")


class ClusterError(ValueError):
    """Raised when the cluster design cannot support a resampling interval."""


class ClusterSet:
    """Rows sorted by cluster label with CSR offsets. `columns` are float/bool arrays aligned with the input rows."""

    def __init__(self, labels: Sequence, columns: Mapping[str, np.ndarray], min_clusters: int = MIN_CLUSTERS, row_level: bool = False):
        labels = np.asarray(labels)
        n = len(labels)
        if n == 0:
            raise ClusterError("no rows")
        codes, uniq = _factorize_sorted(labels)
        order = np.argsort(codes, kind="stable")
        self.labels = uniq
        self.n_clusters = len(uniq)
        self.n_rows = n
        self.row_level = row_level
        self.counts = np.bincount(codes, minlength=len(uniq))
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(np.int64)
        self.cols: Dict[str, np.ndarray] = {k: np.asarray(v)[order] for k, v in columns.items()}
        self.row_order = order                                    # original row position of each sorted row
        self.cluster_of_row = codes[order]
        if self.n_clusters < min_clusters:
            raise ClusterError(f"only {self.n_clusters} clusters (< {min_clusters}): no resampling interval is produced")

    def rows_for(self, seq: np.ndarray) -> np.ndarray:
        """Row indices (into the sorted arrays) of the clusters in `seq` (with repeats)."""
        ln = self.counts[seq]
        tot = int(ln.sum())
        start = np.repeat(self.offsets[seq] - np.cumsum(ln) + ln, ln)
        return start + np.arange(tot)

    def subset(self, cluster_mask: np.ndarray, min_clusters: int = MIN_CLUSTERS) -> "ClusterSet":
        keep_rows = np.asarray(cluster_mask)[self.cluster_of_row]
        return ClusterSet(self.labels[self.cluster_of_row][keep_rows], {k: v[keep_rows] for k, v in self.cols.items()}, min_clusters, self.row_level)


def _factorize_sorted(labels: np.ndarray):
    uniq, codes = np.unique(labels, return_inverse=True)
    return codes, uniq


def draw_clusters(scheme: str, rng: np.random.Generator, n: int, block: Optional[int] = None) -> np.ndarray:
    if scheme == "iid":
        return rng.integers(0, n, size=n)
    if block is None or block < 1:
        raise ClusterError("block length must be >= 1")
    if block > n:
        raise ClusterError(f"block length {block} exceeds the number of clusters {n}")
    if scheme == "moving_block":
        nb = math.ceil(n / block)
        starts = rng.integers(0, n, size=nb)
        return ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
    if scheme == "stationary":
        p = 1.0 / block
        new = rng.random(n) < p
        new[0] = True
        first = np.flatnonzero(new)
        block_id = np.cumsum(new) - 1
        starts = rng.integers(0, n, size=len(first))
        pos = np.arange(n) - first[block_id]
        return (starts[block_id] + pos) % n
    raise ClusterError(f"unknown scheme {scheme!r}")


def bootstrap(cs: ClusterSet, evaluate, scheme: str, block: Optional[int] = None, reps: int = 1000, seed: int = 20260101, k: int = 1) -> np.ndarray:
    """reps x k matrix of `evaluate(rows, seq)` over cluster resamples. Deterministic for a given seed."""
    if scheme not in SCHEMES:
        raise ClusterError(f"unknown scheme {scheme!r}")
    rng = np.random.default_rng(seed)
    out = np.empty((reps, k))
    for r in range(reps):
        seq = draw_clusters(scheme, rng, cs.n_clusters, block)
        out[r] = evaluate(cs.rows_for(seq), seq)
    return out


def percentile_ci(draws: np.ndarray, level: float = 0.95):
    a = (1 - level) / 2 * 100
    return np.nanpercentile(draws, a, axis=0), np.nanpercentile(draws, 100 - a, axis=0)


def bootstrap_series(x: np.ndarray, scheme: str, block: Optional[int] = None, reps: int = 1000, seed: int = 20260101, stat=np.mean, min_clusters: int = MIN_CLUSTERS) -> np.ndarray:
    """Bootstrap of a statistic of an ORDERED 1-D series (one value per cluster), same schemes and guards as `bootstrap`."""
    x = np.asarray(x, float)
    if len(x) < min_clusters:
        raise ClusterError(f"only {len(x)} clusters (< {min_clusters})")
    rng = np.random.default_rng(seed)
    return np.array([stat(x[draw_clusters(scheme, rng, len(x), block)]) for _ in range(reps)])
