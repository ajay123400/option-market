"""Influence and jackknife analysis (Stage 2D, T4). Descriptive; the influence ranking is NOT an exclusion rule.

* leave-one-cluster-out estimates and the delete-1 jackknife standard error
* delete-block (moving-block) jackknife (Kunsch 1989): replicates delete b consecutive clusters, var = (n-b)/(b N) * sum (theta_j - mean)^2, N = n-b+1
* leave-one-group-out (e.g. calendar year of expiry)
* removal of the k most influential clusters (ranked by their leave-one-out effect on the SAME estimand) versus removal of k random clusters
* trimmed and winsorized means of a per-cluster series
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .clusters import ClusterError, ClusterSet
from .estimands import Estimands


def _eval_clusters(cs: ClusterSet, est: Estimands, keep: np.ndarray) -> np.ndarray:
    seq = np.flatnonzero(keep)
    return est.evaluate(cs.rows_for(seq), seq)


def leave_one_out(cs: ClusterSet, est: Estimands) -> np.ndarray:
    n = cs.n_clusters
    out = np.empty((n, len(est.point())))
    for i in range(n):
        keep = np.ones(n, bool)
        keep[i] = False
        out[i] = _eval_clusters(cs, est, keep)
    return out


def jackknife_se(theta_full: np.ndarray, loo: np.ndarray) -> Dict[str, np.ndarray]:
    n = loo.shape[0]
    mean = loo.mean(axis=0)
    se = np.sqrt((n - 1) / n * ((loo - mean) ** 2).sum(axis=0))
    bias = (n - 1) * (mean - theta_full)
    return dict(se=se, bias=bias)


def block_jackknife_se(cs: ClusterSet, est: Estimands, block: int) -> np.ndarray:
    n = cs.n_clusters
    if block < 1 or block >= n:
        raise ClusterError("block must be in [1, n_clusters)")
    reps = []
    for j in range(n - block + 1):
        keep = np.ones(n, bool)
        keep[j: j + block] = False
        reps.append(_eval_clusters(cs, est, keep))
    reps = np.array(reps)
    N = len(reps)
    return np.sqrt((n - block) / (block * N) * ((reps - reps.mean(axis=0)) ** 2).sum(axis=0))


def leave_one_group_out(cs: ClusterSet, est: Estimands, group_of_cluster: Sequence) -> Dict[str, np.ndarray]:
    g = np.asarray(group_of_cluster)
    out = {}
    for u in sorted(set(g.tolist())):
        keep = g != u
        if keep.sum() < 1:
            continue
        out[u] = (int((g == u).sum()), _eval_clusters(cs, est, keep))
    return out


def topk_removal(cs: ClusterSet, est: Estimands, loo: np.ndarray, theta: np.ndarray, ks: Sequence[int], random_draws: int = 200, seed: int = 20260101) -> Dict[int, Dict[str, np.ndarray]]:
    """For each k: estimate after removing the k clusters whose removal moves each estimand most (per estimand), and the 5/50/95 percentiles of the
    estimate after removing k RANDOM clusters."""
    n = cs.n_clusters
    k_est = len(theta)
    rng = np.random.default_rng(seed)
    res: Dict[int, Dict[str, np.ndarray]] = {}
    infl = np.abs(loo - theta)                                       # n x k_est
    for k in ks:
        if k >= n:
            raise ClusterError("k must be < number of clusters")
        top = np.empty(k_est)
        for e in range(k_est):
            idx = np.argsort(-infl[:, e], kind="stable")[:k]
            keep = np.ones(n, bool)
            keep[idx] = False
            top[e] = _eval_clusters(cs, est, keep)[e]
        rnd = np.empty((random_draws, k_est))
        for r in range(random_draws):
            keep = np.ones(n, bool)
            keep[rng.choice(n, size=k, replace=False)] = False
            rnd[r] = _eval_clusters(cs, est, keep)
        res[k] = dict(top=top, rnd_p05=np.nanpercentile(rnd, 5, axis=0), rnd_p50=np.nanpercentile(rnd, 50, axis=0), rnd_p95=np.nanpercentile(rnd, 95, axis=0))
    return res


def trimmed_mean(x: np.ndarray, prop: float) -> float:
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    k = int(np.floor(prop * len(x)))
    return float(x[k: len(x) - k].mean())


def winsorized_mean(x: np.ndarray, prop: float) -> float:
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    k = int(np.floor(prop * len(x)))
    if k:
        x = x.copy()
        x[:k] = x[k]
        x[len(x) - k:] = x[len(x) - k - 1]
    return float(x.mean())
