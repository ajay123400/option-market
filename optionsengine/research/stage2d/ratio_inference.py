"""Total-variance-ratio inference (Stage 2D, T3): exact sign test, block sign-flip permutation test, HAC t statistic.

H0 for all three: the per-expiry total-variance ratio is centred on 1 (ln r centred on 0). A small p-value says the historical
difference is unlikely under that null given the stated dependence handling; it does not say the difference is tradable, stable
out of sample, or free of selection effects (those are other Stage 2D gates).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .clusters import ClusterError


def sign_test(x: np.ndarray, center: float = 0.0) -> Dict[str, float]:
    """Exact two-sided binomial sign test on x - center (ties dropped). ASSUMES independent expiries (stated limitation)."""
    d = np.asarray(x, float) - center
    d = d[d != 0]
    n = len(d)
    if n == 0:
        raise ValueError("no non-tied values")
    k = int((d > 0).sum())
    kk = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(kk, n + 1)) / 2.0 ** n
    return dict(n=n, n_positive=k, share_positive=k / n, p_two_sided=min(1.0, 2 * tail))


def sign_flip_test(x: np.ndarray, block: int = 1, reps: int = 10000, seed: int = 20260101, min_blocks: int = 10) -> Dict[str, float]:
    """Permutation test of E[x] = 0 by flipping the sign of whole blocks of consecutive expiries (fixed partition into ceil(n/block) blocks).
    Assumes x is symmetric about 0 under H0 and that dependence is shorter than `block` (block = 1 assumes independence)."""
    x = np.asarray(x, float)
    n = len(x)
    if block < 1:
        raise ClusterError("block must be >= 1")
    nb = math.ceil(n / block)
    if nb < min_blocks:
        raise ClusterError(f"only {nb} sign-flip blocks (< {min_blocks})")
    bid = np.arange(n) // block
    bsum = np.bincount(bid, weights=x, minlength=nb)
    obs = abs(bsum.sum()) / n
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(reps, nb))
    stat = np.abs(signs @ bsum) / n
    return dict(n=n, block=block, n_blocks=nb, statistic=float(x.mean()), p_two_sided=float((1 + np.sum(stat >= obs - 1e-15)) / (1 + reps)), reps=reps)


def hac_t(x: np.ndarray, lags: int) -> Dict[str, float]:
    from .dependence import newey_west
    r = newey_west(x, lags)
    return dict(lags=r["lags"], mean=r["mean"], se=r["nw_se"], t=r["mean"] / r["nw_se"] if r["nw_se"] > 0 else math.nan)
