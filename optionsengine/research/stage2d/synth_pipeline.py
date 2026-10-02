"""T11 Tier A: small FULL-PIPELINE known-truth recovery (Stage 2D, Gate 3b). Measurement/validation only.

A small synthetic world is written in the exact on-disk format the real research stack reads: a 1-minute index file, one option file per weekly expiry and a participant-OI date directory
(only used as a trading-date calendar). The REAL Stage 2A (surface), Stage 2B/2C (realized variance, targets, annualization audit) code is then run on it unchanged and its output is compared
with the generator's truth.

Generator (deterministic volatility path => exact conditional expectations):
* sessions Mon-Fri with fixed holidays; per-session annualised vol level_d = base + amp sin(2 pi d / period) + step (regime shift);  v_d = (level/100)^2 / 252
* minute returns r_{d,m} ~ N(0, v_d (1 - omega) w_m) with a U-shaped intraday weight profile w_m (sum 1) plus, with probability p_jump per session, one jump of variance v_d xi (xi lognormal, mean jump_mean)
* overnight gap return ~ N(0, v_d omega (1 + weekend_extra * extra_d)); prices P_m = P_{m-1} exp(r_m), bar b has open P_b, close P_{b+1}, a 5 bp wick
* realised window variance V (hybrid) = remainder of the snapshot session after the snapshot bar + sum over later sessions of (gap^2 + sum r^2): the Stage 2C definition, computed here from the
  generated returns, NOT from the pipeline
* oracle expectation F = E[V | information at t] (closed form), implied total variance W = c F, IV_true = sqrt(W / T) with T = ACT/365 to the 15:30 expiry close
* option quotes at the 10:00 / 13:00 / 15:00 bars: Black-76 prices (r = 6.5%) on a 50-point strike grid (+-5%) from a smile IV(K) = IV_true (1 + skew x + curv x^2), x = ln K/F, forward F = S exp(r T),
  rounded to the 0.05 tick (volume > 0 at the snapshot bar => quote age 1 minute)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import quote_noise as Q

IST = timezone(timedelta(hours=5, minutes=30))
N = 375
SLOTS = {"10:00": 45, "13:00": 225, "15:00": 345}
OBS_MIN = {"10:00": 10 * 60 + 1, "13:00": 13 * 60 + 1, "15:00": 15 * 60 + 1}
R_RATE = 0.065


@dataclass(frozen=True)
class TierADGP:
    n_weeks: int = 38
    start: date = date(2024, 9, 2)
    holidays: Tuple[date, ...] = (date(2024, 10, 2), date(2024, 11, 1), date(2025, 2, 26))
    base: float = 12.0
    amp: float = 4.0
    period: float = 40.0
    step_at: int = 110
    step: float = 5.0
    omega: float = 0.15
    weekend_extra: float = 0.25
    p_jump: float = 0.02
    jump_mean: float = 6.0
    jump_sigma: float = 0.8
    c: float = 1.0
    s0: float = 22000.0
    skew: float = -0.6
    curv: float = 3.0
    strike_step: float = 50.0
    strike_halfwidth: float = 0.05
    max_dte_days: int = 13


def minute_weights() -> np.ndarray:
    m = np.arange(1, N + 1, dtype=float)
    w = 1.0 + 1.5 * np.exp(-m / 20.0) + 0.8 * np.exp(-(N - m) / 25.0)
    return w / w.sum()


def sessions(dgp: TierADGP) -> Tuple[List[date], List[date]]:
    days, d = [], dgp.start
    end = dgp.start + timedelta(weeks=dgp.n_weeks)
    hol = set(dgp.holidays)
    while d < end:
        if d.weekday() < 5 and d not in hol:
            days.append(d)
        d += timedelta(days=1)
    expiries = [d for d in days if d.weekday() == 3 and d >= dgp.start + timedelta(days=14)]
    return days, expiries


def _epoch(d: date, minute_of_day: int) -> int:
    return int(datetime(d.year, d.month, d.day, minute_of_day // 60, minute_of_day % 60, tzinfo=IST).timestamp())


def generate_world(dgp: TierADGP, seed: int, root: Optional[str], participant_dir: Optional[str], write: bool = True) -> pd.DataFrame:
    """Writes root/NIFTY50_1m.parquet, root/options/<expiry>.parquet and participant_dir/<date>.csv (unless write=False); returns the TRUTH table (one row per snapshot x expiry)."""
    rng = np.random.default_rng(seed)
    days, expiries = sessions(dgp)
    n = len(days)
    w = minute_weights()
    cw = np.cumsum(w)
    extra = np.zeros(n)
    for i in range(1, n):
        extra[i] = (days[i] - days[i - 1]).days - 1
    idx = np.arange(n)
    level = dgp.base + dgp.amp * np.sin(2 * math.pi * idx / dgp.period) + np.where(idx >= dgp.step_at, dgp.step, 0.0)
    v = (level / 100.0) ** 2 / 252.0
    mu_xi = math.log(dgp.jump_mean) - 0.5 * dgp.jump_sigma ** 2
    # returns
    R = rng.normal(0.0, 1.0, (n, N)) * np.sqrt(v[:, None] * (1 - dgp.omega) * w[None, :])
    jump = rng.random(n) < dgp.p_jump
    jm = np.minimum(np.searchsorted(cw, rng.random(n)), N - 1)
    xi = np.exp(rng.normal(mu_xi, dgp.jump_sigma, n))
    for i in np.flatnonzero(jump):
        R[i, jm[i]] += rng.normal(0.0, math.sqrt(v[i] * xi[i]))
    gap = rng.normal(0.0, 1.0, n) * np.sqrt(v * dgp.omega * (1 + dgp.weekend_extra * extra))
    gap[0] = 0.0
    # price path, bars
    P = np.empty((n, N + 1))
    prev = dgp.s0
    for i in range(n):
        P[i, 0] = prev * math.exp(gap[i])
        P[i, 1:] = P[i, 0] * np.exp(np.cumsum(R[i]))
        prev = P[i, -1]
    if write:
        os.makedirs(os.path.join(root, "options"), exist_ok=True)
        os.makedirs(participant_dir, exist_ok=True)
    ts, op, hi, lo, cl = [], [], [], [], []
    for i, d in enumerate(days if write else []):
        t0 = _epoch(d, 9 * 60 + 15)
        o, c_ = P[i, :-1], P[i, 1:]
        ts.append(t0 + 60 * np.arange(N)); op.append(o); cl.append(c_)
        hi.append(np.maximum(o, c_) * 1.00005); lo.append(np.minimum(o, c_) * 0.99995)
        with open(os.path.join(participant_dir, f"{d.isoformat()}.csv"), "w") as fh:
            fh.write("x")
    if write:
        spot = pd.DataFrame(dict(ts=np.concatenate(ts).astype("int64"), open=np.concatenate(op), high=np.concatenate(hi), low=np.concatenate(lo), close=np.concatenate(cl), volume=np.ones(n * N)))
        spot.to_parquet(os.path.join(root, "NIFTY50_1m.parquet"))
    # cumulative realised variance by session (gap^2 + sum r^2) for window sums
    sess_var = gap ** 2 + (R ** 2).sum(axis=1)
    csum = np.concatenate([[0.0], np.cumsum(sess_var)])
    pos = {d: i for i, d in enumerate(days)}
    a_intra = (1 - dgp.omega) + dgp.p_jump * dgp.jump_mean
    g = dgp.omega * (1 + dgp.weekend_extra * extra) + a_intra
    gcum = np.concatenate([[0.0], np.cumsum(v * g)])
    truth = []
    for ed in expiries:
        e = pos[ed]
        end_dt = datetime(ed.year, ed.month, ed.day, 15, 30, tzinfo=IST)
        recs = []
        for i in range(max(0, e - 12), e):
            dd = (ed - days[i]).days
            if dd > dgp.max_dte_days or dd <= 0:
                continue
            for tm, s in SLOTS.items():
                obs = datetime(days[i].year, days[i].month, days[i].day, OBS_MIN[tm] // 60, OBS_MIN[tm] % 60, tzinfo=IST)
                T = (end_dt - obs).total_seconds() / (365 * 86400)
                Vrem = float((R[i, s + 1:] ** 2).sum())                      # returns m = s+2 .. 375 (0-based s+1 ..): strictly after the snapshot bar's close
                Vlater = float(csum[e + 1] - csum[i + 1])
                V = Vrem + Vlater
                Frem = v[i] * a_intra * float(w[s + 1:].sum())
                F = Frem + float(gcum[e + 1] - gcum[i + 1])
                W = dgp.c * F
                iv = math.sqrt(W / T)
                spot_t = float(P[i, s + 1])
                recs.append(dict(obs_id=f"{days[i].isoformat()}T{tm}_{ed.isoformat()}", day=days[i].isoformat(), time=tm, expiry=ed.isoformat(), T_years=T, V_true=V, F_true=F, W_true=W, iv_true=iv,
                                 spot=spot_t, session_equivalents=(N - 1 - s) / 375.0 + (e - i), span_days=T * 365.0))
        truth.extend(recs)
        if not write:
            continue
        # option file for this expiry
        rows = []
        for rec in recs:
            F_fwd = rec["spot"] * math.exp(R_RATE * rec["T_years"])
            K0 = round(F_fwd / dgp.strike_step) * dgp.strike_step
            nk = int(dgp.strike_halfwidth * F_fwd / dgp.strike_step)
            K = K0 + dgp.strike_step * np.arange(-nk, nk + 1)
            x = np.log(K / F_fwd)
            iv_k = rec["iv_true"] * np.maximum(1 + dgp.skew * x + dgp.curv * x * x, 0.3)
            t_bar = _epoch(date.fromisoformat(rec["day"]), 9 * 60 + 15 + SLOTS[rec["time"]])
            for call, typ in ((True, "CE"), (False, "PE")):
                px = Q.b76_price(np.full(len(K), F_fwd), K, np.full(len(K), rec["T_years"]), R_RATE, iv_k, np.full(len(K), call))
                px = np.maximum(np.round(px / 0.05) * 0.05, 0.05)
                rows.append(pd.DataFrame(dict(symbol=[f"NIFTY{ed:%y%m%d}{int(k)}{typ}" for k in K], type=typ, strike=K.astype(float), ts=np.int64(t_bar), close=px, volume=np.int64(100))))
        pd.concat(rows, ignore_index=True).to_parquet(os.path.join(root, "options", f"{ed.isoformat()}.parquet"))
    return pd.DataFrame(truth)


def expected_variance_check(dgp: TierADGP, n_rep: int = 20000, seed: int = 1) -> Tuple[float, float]:
    """Monte-Carlo check of the oracle expectation for ONE session: mean of simulated intraday remainder variance vs v (1-omega + p_jump m_xi) * sum(w after)."""
    rng = np.random.default_rng(seed)
    w = minute_weights()
    cw = np.cumsum(w)
    v = (dgp.base / 100.0) ** 2 / 252.0
    s = 45
    mu_xi = math.log(dgp.jump_mean) - 0.5 * dgp.jump_sigma ** 2
    r = rng.normal(0.0, 1.0, (n_rep, N)) * np.sqrt(v * (1 - dgp.omega) * w)
    jump = rng.random(n_rep) < dgp.p_jump
    jm = np.minimum(np.searchsorted(cw, rng.random(n_rep)), N - 1)
    xi = np.exp(rng.normal(mu_xi, dgp.jump_sigma, n_rep))
    jr = rng.normal(0.0, 1.0, n_rep) * np.sqrt(v * xi) * jump
    r[np.arange(n_rep), jm] += jr
    sim = float((r[:, s + 1:] ** 2).sum(axis=1).mean())
    theo = v * ((1 - dgp.omega) + dgp.p_jump * dgp.jump_mean) * float(w[s + 1:].sum())
    return sim, theo


def recovery_table(aligned_exp: pd.DataFrame, truth: pd.DataFrame, c: float) -> pd.DataFrame:
    """Row-level comparison of the pipeline output with the generator truth."""
    a = aligned_exp.merge(truth, on="obs_id", suffixes=("", "_t"))
    rv_cal_true = 100.0 * np.sqrt(365.0 * a.V_true / a.span_days_t)
    rv_sess_true = 100.0 * np.sqrt(252.0 * a.V_true / a.session_equivalents_t)
    out = pd.DataFrame(dict(obs_id=a.obs_id, iv_pipeline=a.iv_pct, iv_true=100 * a.iv_true, iv_error=a.iv_pct - 100 * a.iv_true,
                            V_pipeline=a.rv_total_variance, V_true=a.V_true, V_rel_error=(a.rv_total_variance - a.V_true) / a.V_true,
                            spread_cal_pipeline=a.spread_calendar, spread_cal_true=100 * a.iv_true - rv_cal_true, spread_sess_pipeline=a.spread_session, spread_sess_true=100 * a.iv_true - rv_sess_true,
                            ratio_pipeline=a.total_variance_ratio, ratio_true=a.W_true / a.V_true, W_pipeline=a.implied_total_variance, W_true=a.W_true))
    out["spread_cal_error"] = out.spread_cal_pipeline - out.spread_cal_true
    out["ratio_error"] = out.ratio_pipeline - out.ratio_true
    out["c"] = c
    return out


def oracle_r3_across_worlds(dgp: TierADGP, n_worlds: int = 200, seed0: int = 9000) -> Tuple[float, float, float]:
    """Generator-level unbiasedness check (no files): mean and Monte-Carlo SE across worlds of the oracle ratio of summed total variances sum(W)/sum(V), and c.
    sum(W)/sum(V) = c sum(F)/sum(V): with F = E[V | information at t] its expectation is c up to the (small) Jensen term of a ratio of random sums."""
    vals = []
    for k in range(n_worlds):
        t = generate_world(dgp, seed0 + k, None, None, write=False)
        vals.append(float(t.W_true.sum() / t.V_true.sum()))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n_worlds)), float(dgp.c)
