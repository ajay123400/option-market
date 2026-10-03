"""T11 Tier B synthetic process (Stage 2D, Gate 3b). Measurement/validation only.

A calibrated, known-truth data-generating process at the DAILY level that produces rows with exactly the columns the Gate 1/2 statistics consume.
It is deliberately richer than an AR(1): regime shifts, heavy tails, jumps, an intraday variance profile, overnight/weekend variance, holidays,
weekly expiries whose <= 14-day windows OVERLAP (cross-expiry dependence arises mechanically from shared vol states and shared future sessions).

Process (all per session d; annualised vol levels use the 252-session convention):
* vol STATE s_d: K-state Markov chain  P = stay*I + (1 - stay - eps)*N + eps*U   (N: nearest-neighbour kernel, U: uniform jumps = rare regime shifts to any state, incl. a crisis state);
  daily variance v_s = (level_s/100)^2 / 252.
* intraday variance V_d = v_s (1 - omega) Z_d  +  1{jump} v_s xi_d,   Z lognormal(mean 1, sigma_z),  xi lognormal(mean jump_mean, sigma_jump)    (heavy right tail)
* overnight/gap variance O_d = v_s omega (1 + weekend_extra * extra_d) Zo_d,  Zo lognormal(mean 1, sigma_zo)       (extra_d = non-trading days before d)
* intraday variance is split into 4 segments (pre-10:00, 10-13, 13-15, 15-15:30) by Dirichlet(seg_conc * seg_mean) shares  (so 10:00/13:00/15:00 snapshot remainders are nested)
* weekly Thursday expiries, snapshots at 10:00 / 13:00 / 15:00 on each session within 14 calendar days before the expiry (expiry day excluded: no expiry-aligned target)
* realised total variance of the window  V = remainder of the snapshot session + sum over later sessions (O + V_d)             [hybrid, exactly the Stage 2C construction]
* implied total variance  W = c * F * exp(m_d + eta - (sigma_m^2 + sigma_eta^2)/2),  F = E[V | state at t] (EXACT conditional expectation from the Markov chain),
  m_d a persistent AR(1) market-wide mispricing shared by every row of a day, eta ~ N(0, sigma_eta^2) per row
  => the TRUTH is known: the population value of every estimand is obtained from a very long simulation of the same process (`truth_estimates`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import estimands as E

SLOT_MIN = {0: 10 * 60 + 1, 1: 13 * 60 + 1, 2: 15 * 60 + 1}           # observation instant (bar start + 60 s), minutes after midnight
SLOT_NAME = {0: "10:00", 1: "13:00", 2: "15:00"}
N_PART = {0: 329, 1: 149, 2: 29}                                       # snapshot-session remainder returns (Stage 2C)
EXPIRY_CLOSE_MIN = 15 * 60 + 30
HMAX = 14


@dataclass(frozen=True)
class DGP:
    levels: Tuple[float, ...] = (7.5, 9.5, 12.0, 16.0, 21.0, 30.0)
    pi_target: Tuple[float, ...] = (0.24, 0.30, 0.24, 0.14, 0.06, 0.02)
    stay: float = 0.86
    eps: float = 0.025
    omega: float = 0.15
    weekend_extra: float = 0.25
    sigma_z: float = 0.40
    sigma_zo: float = 0.6
    p_jump: float = 0.02
    jump_mean: float = 6.0
    jump_sigma: float = 0.8
    seg_mean: Tuple[float, ...] = (0.30, 0.34, 0.24, 0.12)
    seg_conc: float = 60.0
    c: float = 1.20
    sigma_eta: float = 0.05
    sigma_m: float = 0.24
    phi_m: float = 0.60
    holiday_prob: float = 0.02
    max_dte_days: float = 14.0


def transition_matrix(dgp: DGP) -> Tuple[np.ndarray, np.ndarray]:
    """P = (1 - eps) (stay I + (1 - stay) M) + eps 1 pi', with M the nearest-neighbour Metropolis kernel for pi = pi_target: pi is the exact stationary law."""
    K = len(dgp.levels)
    pi = np.asarray(dgp.pi_target, float)
    pi = pi / pi.sum()
    M = np.zeros((K, K))
    for i in range(K):
        for j in (i - 1, i + 1):
            if 0 <= j < K:
                M[i, j] = 0.5 * min(1.0, pi[j] / pi[i])
        M[i, i] = 1.0 - M[i].sum()
    P = (1 - dgp.eps) * (dgp.stay * np.eye(K) + (1 - dgp.stay) * M) + dgp.eps * np.outer(np.ones(K), pi)
    return P, pi


def _lognormal_mean_one(rng, sigma, size):
    return np.exp(rng.normal(0.0, sigma, size) - 0.5 * sigma * sigma)


def calendar(n_expiries: int, burn_days: int, rng: np.random.Generator, holiday_prob: float, start: date = date(2021, 1, 4)):
    """Weekdays from `start`; random weekday holidays (never an expiry Thursday). Returns dates, expiry session positions, extra non-trading days before each session."""
    first_expiry = start + timedelta(days=burn_days // 5 * 7 + 3)
    expiry_dates = [first_expiry + timedelta(days=7 * k) for k in range(n_expiries)]
    last = expiry_dates[-1]
    days, d = [], start
    ex = set(expiry_dates)
    while d <= last:
        if d.weekday() < 5 and (d in ex or rng.random() >= holiday_prob):
            days.append(d)
        d += timedelta(days=1)
    pos = {d: i for i, d in enumerate(days)}
    extra = np.zeros(len(days))
    for i in range(1, len(days)):
        extra[i] = (days[i] - days[i - 1]).days - 1
    return days, pos, [pos[e] for e in expiry_dates], expiry_dates, extra


def simulate(dgp: DGP, n_expiries: int, seed: int, burn_days: int = 60) -> pd.DataFrame:
    """One synthetic history. Columns: day, time, expiry, T_days, span_days, session_equivalents, V (realised window variance), F_eta (= F exp(eta - s^2/2); implied variance is c * F_eta)."""
    rng = np.random.default_rng(seed)
    P, pi = transition_matrix(dgp)
    K = len(dgp.levels)
    days, pos, e_pos, e_dates, extra = calendar(n_expiries, burn_days, rng, dgp.holiday_prob)
    n = len(days)
    vS = (np.asarray(dgp.levels) / 100.0) ** 2 / 252.0
    # state path
    cum = np.cumsum(P, axis=1)
    s = np.empty(n, int)
    s[0] = rng.choice(K, p=pi)
    u = rng.random(n)
    for i in range(1, n):
        s[i] = min(int(np.searchsorted(cum[s[i - 1]], u[i])), K - 1)
    v = vS[s]
    mxi = dgp.jump_mean
    mu_xi = math.log(mxi) - 0.5 * dgp.jump_sigma ** 2
    Z = _lognormal_mean_one(rng, dgp.sigma_z, n)
    jump = rng.random(n) < dgp.p_jump
    xi = np.exp(rng.normal(mu_xi, dgp.jump_sigma, n))
    Vi = v * (1 - dgp.omega) * Z + jump * v * xi
    Zo = _lognormal_mean_one(rng, dgp.sigma_zo, n)
    O = v * dgp.omega * (1 + dgp.weekend_extra * extra) * Zo
    sh = rng.dirichlet(dgp.seg_conc * np.asarray(dgp.seg_mean), n)                  # n x 4
    m = np.zeros(n)
    nm = rng.normal(0.0, 1.0, n)
    for i in range(1, n):
        m[i] = dgp.phi_m * m[i - 1] + dgp.sigma_m * math.sqrt(1 - dgp.phi_m ** 2) * nm[i]
    m[0] = dgp.sigma_m * nm[0]
    a_intra = (1 - dgp.omega) + dgp.p_jump * mxi
    g = dgp.omega * (1 + dgp.weekend_extra * extra) + a_intra                        # expected variance multiplier of a session (relative to v) given nothing else
    # expected variance of later sessions: A[s, h] = (P^h v)[s]
    A = np.empty((K, HMAX + 1))
    cur = vS.copy()
    for h in range(1, HMAX + 1):
        cur = P @ cur
        A[:, h] = cur
    # cumF[i, h] = sum_{k=1..h} A[s_i, k] * g_{i+k}
    cumF = np.zeros((n, HMAX + 1))
    for h in range(1, HMAX + 1):
        gk = np.zeros(n)
        gk[: n - h] = g[h:]
        cumF[:, h] = cumF[:, h - 1] + A[s, h] * gk
    csum = np.concatenate([[0.0], np.cumsum(O + Vi)])
    mean_after = {0: sum(dgp.seg_mean[1:]), 1: sum(dgp.seg_mean[2:]), 2: dgp.seg_mean[3]}
    real_after = {0: Vi * (sh[:, 1] + sh[:, 2] + sh[:, 3]), 1: Vi * (sh[:, 2] + sh[:, 3]), 2: Vi * sh[:, 3]}
    rows = []
    cols = {k: [] for k in ("day", "time", "expiry", "T_days", "span_days", "session_equivalents", "V", "F_eta", "expiry_idx")}
    for k, (ep, ed) in enumerate(zip(e_pos, e_dates)):
        for i in range(max(0, ep - 12), ep):
            dd = (ed - days[i]).days
            if dd > dgp.max_dte_days or dd <= 0:
                continue
            h = ep - i
            if h > HMAX:
                continue
            for slot in (0, 1, 2):
                T_days = dd + (EXPIRY_CLOSE_MIN - SLOT_MIN[slot]) / 1440.0
                V = real_after[slot][i] + (csum[ep + 1] - csum[i + 1])
                F = vS[s[i]] * mean_after[slot] * a_intra + cumF[i, h]
                eta = rng.normal(0.0, dgp.sigma_eta)
                cols["day"].append(days[i].isoformat()); cols["time"].append(SLOT_NAME[slot]); cols["expiry"].append(ed.isoformat())
                cols["T_days"].append(T_days); cols["span_days"].append(T_days); cols["session_equivalents"].append(N_PART[slot] / 375.0 + h)
                cols["V"].append(V); cols["F_eta"].append(F * math.exp(m[i] + eta - 0.5 * (dgp.sigma_m ** 2 + dgp.sigma_eta ** 2))); cols["expiry_idx"].append(k)
    return pd.DataFrame(cols)


def to_aligned(rows: pd.DataFrame, c) -> pd.DataFrame:
    """Rows in the Gate 1/2 aligned format for implied variance W = c * F_eta (c: scalar or per-row array)."""
    d = rows.copy()
    cc = np.asarray(c, float) if np.ndim(c) else np.full(len(d), float(c))
    W = cc * d.F_eta.to_numpy(float)
    V = d.V.to_numpy(float)
    T_years = d.T_days.to_numpy(float) / 365.0
    iv = 100.0 * np.sqrt(W / T_years)
    rv_s = 100.0 * np.sqrt(252.0 * V / d.session_equivalents.to_numpy(float))
    rv_c = 100.0 * np.sqrt(365.0 * V / d.span_days.to_numpy(float))
    d["iv_pct"] = iv
    d["rv_session_pct"], d["rv_calendar_pct"] = rv_s, rv_c
    d["spread_session"], d["spread_calendar"] = iv - rv_s, iv - rv_c
    d["implied_total_variance"], d["rv_total_variance"] = W, V
    d["total_variance_ratio_valid"] = V > 0
    d["total_variance_ratio"] = W / V
    d["dte_bucket"] = np.where(d.T_days <= 3, "1-3d", np.where(d.T_days <= 7, "3-7d", "7-14d"))
    return d


def assign_split(rows: pd.DataFrame, n_dev_expiries: int) -> pd.Series:
    """Dev = rows whose snapshot day precedes the first snapshot day of expiry index n_dev_expiries (a calendar split, as in the real data: expiries may straddle it)."""
    cut = rows.loc[rows.expiry_idx >= n_dev_expiries, "day"].min()
    return pd.Series(np.where(rows.day < cut, "dev", "holdout"), index=rows.index)


# ------------------------------------------------------------------------------ truth and calibration
def truth_estimates(dgp: DGP, n_expiries: int = 12000, seed: int = 987654, c=None) -> Dict[str, float]:
    """Population values of the nine Gate 1 estimands for the process: the estimands evaluated on a very long history of the same process."""
    rows = simulate(dgp, n_expiries, seed)
    est = E.build(to_aligned(rows, dgp.c if c is None else c), min_clusters=1)
    return dict(zip(E.EST_NAMES, est.point()))


def moments(rows: pd.DataFrame, c: float) -> Dict[str, float]:
    """Calibration moments of a synthetic history (comparable with `moments_from_aligned` of the observed sample)."""
    return moments_from_aligned(to_aligned(rows, c))


def moments_from_aligned(a: pd.DataFrame) -> Dict[str, float]:
    """Calibration moments of an aligned-format frame (observed Stage 2C EXP/hybrid rows or synthetic rows)."""
    from . import dependence as D
    e = E.build(a, min_clusters=1)
    per = e.per_cluster_table()
    x = per.median_spread_calendar.to_numpy(float)
    x = x[np.isfinite(x)]
    icc = D.icc_oneway(a.spread_calendar.to_numpy(float), a.expiry.to_numpy())
    r = a.total_variance_ratio.to_numpy(float)
    q = lambda s_, p: float(pd.Series(s_).quantile(p))
    return dict(n_rows=len(a), n_expiries=int(a.expiry.nunique()), rows_per_expiry=float(len(a) / a.expiry.nunique()),
                median_iv=float(a.iv_pct.median()), iv_p05=q(a.iv_pct, .05), iv_p95=q(a.iv_pct, .95),
                median_rv_calendar=float(a.rv_calendar_pct.median()), rv_p05=q(a.rv_calendar_pct, .05), rv_p95=q(a.rv_calendar_pct, .95),
                median_spread_calendar=float(a.spread_calendar.median()), spread_p05=q(a.spread_calendar, .05), spread_p25=q(a.spread_calendar, .25),
                spread_p75=q(a.spread_calendar, .75), spread_p95=q(a.spread_calendar, .95), median_spread_session=float(a.spread_session.median()),
                median_ratio=float(np.median(r)), geometric_mean_ratio=float(math.exp(np.log(r).mean())), ratio_p05=q(r, .05), ratio_p25=q(r, .25), ratio_p75=q(r, .75), ratio_p95=q(r, .95),
                frac_ratio_gt_1=float((r > 1).mean()), ln_ratio_sd=float(np.log(r).std()), ln_ratio_skew=float(pd.Series(np.log(r)).skew()),
                icc_by_expiry=icc["icc"], acf_lag1_expiry_median_spread=float(D.acf(x, 1)[1]), ljung_box_p10=float(D.ljung_box(x, 10)[1]))
