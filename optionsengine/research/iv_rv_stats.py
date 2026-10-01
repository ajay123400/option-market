"""Descriptive statistics and date-block bootstrap for Stage 2C (IV minus subsequent realized volatility).

Everything here is DESCRIPTIVE. There is no ranking of horizons, no threshold search and no performance measure.

Uncertainty: observations on the same date (3 snapshots x several expiries) and on neighbouring dates share
overlapping future windows, so rows are NOT independent. Confidence intervals therefore come from a moving
(circular) BLOCK bootstrap over snapshot DATES: dates are resampled in blocks of `block` consecutive dates
(block = max(5, horizon in sessions), 10 for the expiry-aligned target), all rows of a drawn date are kept
together, and the statistic is recomputed (1,000 replications, fixed seed). The intervals are 2.5/97.5 percentiles.
They quantify sampling uncertainty of the historical sample under serial dependence of the stated block length;
they do not make the sample representative of the whole NIFTY options market.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def block_length(horizon: str) -> int:
    return {"F1": 5, "F5": 5, "F10": 10, "F20": 20, "EXP": 10}.get(horizon, 5)


def _csr(dates: np.ndarray):
    """Rows sorted by date: returns (order, offsets, counts, n_dates)."""
    codes, uniq = pd.factorize(dates, sort=True)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(uniq))
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return order, offsets, counts, len(uniq)


def block_bootstrap(values: np.ndarray, dates: np.ndarray, stats: Dict[str, callable], block: int,
                    reps: int = 1000, seed: int = 20260101) -> Dict[str, tuple]:
    """Moving circular block bootstrap over dates. Returns {stat: (lo, hi)} (2.5 / 97.5 percentiles)."""
    n = len(values)
    if n < 20:
        return {k: (math.nan, math.nan) for k in stats}
    order, offsets, counts, nd = _csr(np.asarray(dates))
    vals = np.asarray(values, dtype=float)[order]
    rng = np.random.default_rng(seed)
    nb = math.ceil(nd / block)
    out = {k: np.empty(reps) for k in stats}
    ar = np.arange(block)
    for r in range(reps):
        starts = rng.integers(0, nd, size=nb)
        seq = ((starts[:, None] + ar) % nd).ravel()[:nd]
        ln = counts[seq]
        tot = int(ln.sum())
        idx = np.repeat(offsets[seq] - np.cumsum(ln) + ln, ln) + np.arange(tot)
        sample = vals[idx]
        for k, fn in stats.items():
            out[k][r] = fn(sample)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def describe(g: pd.DataFrame, ci_horizon: Optional[str] = None, reps: int = 1000) -> dict:
    """One row of descriptive statistics for a group of observations (columns: iv_pct, future_rv_pct, iv_minus_rv, iv_over_rv,
    abs_err, sq_err, day, expiry, time)."""
    n = len(g)
    d = g["iv_minus_rv"].to_numpy(dtype=float)
    row = dict(n_obs=n, unique_dates=int(g["day"].nunique()), unique_expiries=int(g["expiry"].nunique()))
    if n == 0:
        return row
    per_date = g.groupby("day").size()
    row.update(snapshots_per_date_median=float(per_date.median()), snapshots_per_date_max=int(per_date.max()),
               iv_mean=float(g.iv_pct.mean()), iv_median=float(g.iv_pct.median()),
               rv_mean=float(g.future_rv_pct.mean()), rv_median=float(g.future_rv_pct.median()),
               diff_mean=float(d.mean()), diff_median=float(np.median(d)), diff_std=float(d.std(ddof=1)) if n > 1 else math.nan)
    for p in PCTS:
        row[f"diff_p{p:02d}"] = float(np.percentile(d, p))
    ratio = g.loc[g.future_rv_pct > 0, "iv_over_rv"]
    row.update(ratio_n_valid=int(len(ratio)), ratio_median=float(ratio.median()) if len(ratio) else math.nan,
               ratio_mean=float(ratio.mean()) if len(ratio) else math.nan,
               mae=float(g.abs_err.mean()), mse=float(g.sq_err.mean()), rmse=float(math.sqrt(g.sq_err.mean())),
               frac_iv_gt_rv=float((d > 0).mean()), frac_iv_lt_rv=float((d < 0).mean()), frac_iv_eq_rv=float((d == 0).mean()))
    if n > 2 and g.iv_pct.nunique() > 1 and g.future_rv_pct.nunique() > 1:
        row["corr_pearson"] = float(g.iv_pct.corr(g.future_rv_pct))
        row["corr_spearman"] = float(g.iv_pct.corr(g.future_rv_pct, method="spearman"))
    else:
        row["corr_pearson"] = row["corr_spearman"] = math.nan
    if ci_horizon is not None:
        ci = block_bootstrap(d, g["day"].to_numpy(), {"mean": np.mean, "median": np.median, "frac_gt": lambda x: float((x > 0).mean())},
                             block_length(ci_horizon), reps)
        row.update(diff_mean_ci_lo=ci["mean"][0], diff_mean_ci_hi=ci["mean"][1], diff_median_ci_lo=ci["median"][0],
                   diff_median_ci_hi=ci["median"][1], frac_iv_gt_rv_ci_lo=ci["frac_gt"][0], frac_iv_gt_rv_ci_hi=ci["frac_gt"][1],
                   ci_method=f"moving block bootstrap over dates, block={block_length(ci_horizon)}, reps={reps}")
    return row


def summarize(df: pd.DataFrame, keys: Sequence[str], ci: bool = False, reps: int = 1000) -> pd.DataFrame:
    rows: List[dict] = []
    for vals, g in df.groupby(list(keys), observed=True, dropna=False):
        vals = vals if isinstance(vals, tuple) else (vals,)
        base = dict(zip(keys, vals))
        horizon = base.get("horizon") if ci else None
        rows.append({**base, **describe(g, ci_horizon=horizon, reps=reps)})
    return pd.DataFrame(rows)
