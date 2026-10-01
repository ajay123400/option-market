"""Dependence and block-length diagnostics (Stage 2D, T2). Pure numpy; no scipy.

* acf / Ljung-Box (chi-square survival function implemented here, validated against scipy in the validation script)
* one-way ANOVA ICC for rows nested in clusters, design effect and effective sample size
* Politis-White (2004) automatic block length with the Patton-Politis-White (2009) correction (stationary and circular bootstrap)
* Newey-West (Bartlett) long-run variance of a mean
* window-overlap map between consecutive expiries and a pre-specified, outcome-free non-overlapping subsample rule
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------ autocorrelation
def acvf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    d = x - x.mean()
    return np.array([float((d[: n - k] * d[k:]).sum() / n) for k in range(nlags + 1)])


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    g = acvf(x, nlags)
    return g / g[0]


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (series / Lentz continued fraction)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi2_sf(q: float, df: int) -> float:
    return _gammaincc(df / 2.0, q / 2.0)


def ljung_box(x: np.ndarray, lags: int) -> Tuple[float, float]:
    n = len(x)
    r = acf(x, lags)[1:]
    q = n * (n + 2) * float(np.sum(r ** 2 / (n - np.arange(1, lags + 1))))
    return q, chi2_sf(q, lags)


# ------------------------------------------------------------------------------ clustering diagnostics
def icc_oneway(values: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """One-way ANOVA ICC(1) for unbalanced groups, design effect 1 + (mean cluster size - 1) ICC and effective n."""
    values = np.asarray(values, float)
    uniq, codes = np.unique(groups, return_inverse=True)
    k, n = len(uniq), len(values)
    if k < 2:
        raise ValueError("need >= 2 groups")
    cnt = np.bincount(codes, minlength=k).astype(float)
    sums = np.bincount(codes, weights=values, minlength=k)
    gm = values.mean()
    ssb = float((cnt * (sums / cnt - gm) ** 2).sum())
    ssw = float(((values - (sums / cnt)[codes]) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - float((cnt ** 2).sum()) / n) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    deff = 1 + (n / k - 1) * max(icc, 0.0)
    return dict(icc=float(icc), n_rows=n, n_clusters=k, mean_cluster_size=n / k, design_effect=float(deff), n_effective=float(n / deff))


# ------------------------------------------------------------------------------ block length
def politis_white(x: np.ndarray) -> Dict[str, float]:
    """Politis-White (2004) block length with the Patton-Politis-White (2009) correction. Returns stationary- and circular-bootstrap lengths."""
    x = np.asarray(x, float)
    n = len(x)
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    mmax = int(math.ceil(math.sqrt(n))) + kn
    bmax = math.ceil(min(3 * math.sqrt(n), n / 3))
    g = acvf(x, mmax)
    rho = g[1:] / g[0]
    crit = 2.0 * math.sqrt(math.log10(n) / n)
    insig = [int(np.sum(np.abs(rho[j: j + kn]) < crit)) for j in range(mmax - kn + 1)]
    hit = [j for j, v in enumerate(insig) if v == kn]
    if hit:
        mhat = hit[0] + 1
    else:
        sig = np.flatnonzero(np.abs(rho) > crit) + 1
        mhat = int(sig.max()) if len(sig) else 1
    M = min(2 * mhat, mmax)
    ks = np.arange(-M, M + 1)
    t = np.abs(ks) / M
    lam = np.where(t <= 0.5, 1.0, 2 * (1 - t))
    gam = np.array([g[abs(int(k))] for k in ks])
    G = float(np.sum(lam * np.abs(ks) * gam))
    g0 = float(np.sum(lam * gam))
    b_sb = ((2 * G * G) / (2 * g0 * g0)) ** (1 / 3) * n ** (1 / 3) if g0 else 1.0
    b_cb = ((2 * G * G) / ((4 / 3) * g0 * g0)) ** (1 / 3) * n ** (1 / 3) if g0 else 1.0
    return dict(n=n, m_hat=mhat, M=M, G=G, g0=g0, b_stationary=float(min(max(b_sb, 1.0), bmax)), b_circular=float(min(max(b_cb, 1.0), bmax)), b_max=float(bmax))


# ------------------------------------------------------------------------------ HAC
def newey_west(x: np.ndarray, lags: Optional[int] = None) -> Dict[str, float]:
    """Bartlett long-run variance of the series and the implied standard error of its mean. lags default = floor(4 (n/100)^(2/9))."""
    x = np.asarray(x, float)
    n = len(x)
    L = int(math.floor(4 * (n / 100.0) ** (2 / 9))) if lags is None else int(lags)
    g = acvf(x, L)
    lrv = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    se = math.sqrt(max(lrv, 0.0) / n)
    return dict(lags=L, mean=float(x.mean()), iid_se=float(math.sqrt(g[0] / n)), nw_se=float(se), long_run_variance=float(lrv),
                variance_inflation=float(lrv / g[0]) if g[0] else math.nan)


# ------------------------------------------------------------------------------ windows
def expiry_windows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per expiry: window = [first snapshot observation instant, expiry-session close], in calendar order."""
    start = pd.to_datetime(df.target_start_ts, utc=True)
    end = pd.to_datetime(df.target_end_ts, utc=True)
    g = pd.DataFrame(dict(expiry=df.expiry.to_numpy(), start=start.to_numpy(), end=end.to_numpy())).groupby("expiry").agg(start=("start", "min"), end=("end", "max"), n_rows=("start", "size"))
    g = g.sort_values("end").reset_index()
    g["length_days"] = (g.end - g.start).dt.total_seconds() / 86400
    return g


def window_overlap(win: pd.DataFrame) -> pd.DataFrame:
    """For each expiry: overlap (days, share of its window) with the NEXT expiry's window, and how many other expiries' windows overlap it."""
    s = win.start.to_numpy("datetime64[ns]").astype("int64") / 86400e9
    e = win.end.to_numpy("datetime64[ns]").astype("int64") / 86400e9
    n = len(win)
    out = []
    for i in range(n):
        nxt = max(0.0, min(e[i], e[i + 1]) - max(s[i], s[i + 1])) if i + 1 < n else 0.0
        others = int(sum(1 for j in range(n) if j != i and min(e[i], e[j]) > max(s[i], s[j])))
        out.append(dict(expiry=win.expiry.iloc[i], length_days=e[i] - s[i], overlap_with_next_days=nxt,
                        overlap_with_next_share=nxt / (e[i] - s[i]) if e[i] > s[i] else 0.0, n_overlapping_expiries=others))
    return pd.DataFrame(out)


def nonoverlap_subsample(df: pd.DataFrame, time: str = "10:00", t_max_days: float = 6.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-specified, outcome-free rule: per expiry keep the `time` snapshot with the largest T_days <= t_max_days; walk expiries in calendar order and
    drop an expiry whose window starts before the previously KEPT window ends. Returns (kept rows, dropped table with reason)."""
    d = df[(df.time == time) & (df.T_days <= t_max_days)].copy()
    d = d.sort_values(["expiry", "T_days", "day"], ascending=[True, False, True]).groupby("expiry", as_index=False).head(1)
    d["wstart"] = pd.to_datetime(d.target_start_ts, utc=True)
    d["wend"] = pd.to_datetime(d.target_end_ts, utc=True)
    d = d.sort_values("wend")
    kept, dropped, last_end = [], [], None
    for r in d.itertuples():
        if last_end is not None and r.wstart < last_end:
            dropped.append(dict(expiry=r.expiry, reason="window overlaps previously kept window"))
            continue
        kept.append(r.Index)
        last_end = r.wend
    missing = sorted(set(df.expiry) - set(d.expiry))
    dropped += [dict(expiry=e, reason=f"no {time} snapshot with T_days <= {t_max_days}") for e in missing]
    return d.loc[kept].drop(columns=["wstart", "wend"]), pd.DataFrame(dropped, columns=["expiry", "reason"])
