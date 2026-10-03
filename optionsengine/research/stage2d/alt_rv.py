"""Alternative realized-variance constructions for the expiry-aligned target (Stage 2D, T7). Read-only use of the Stage 2B/2C session objects.

Price grid of a regular session (375 bars, bar-start timestamps): P_0 = open of bar 0, P_m = close of bar m-1 (m = 1..375). The 1-minute return m is
ln(P_m / P_{m-1}): exactly the Stage 2B intraday returns (first = open->close of bar 0, the rest close-to-close). Overnight return = ln(P_0 / previous P_375)
(hybrid only), as in Stage 2B.

Sampling step k (minutes): intraday returns on the grid m = 0, k, 2k, ... , 375 (375 is divisible by 5 and 15, so the partition is exact). The snapshot-session
remainder starts at the reference price P_{s+1} (close of the snapshot bar s, observable at the observation instant) and steps by k, ending with a shorter final
step at P_375 when (374 - s) is not a multiple of k. Everything is strictly after the observation instant. k = 1 reproduces the Stage 2C target exactly (tested).

Partial-session weight: the baseline weights the snapshot-session remainder as n_returns / 375 session-equivalents (uniform in time). The alternative uses the
development-period (<= 2024-12-31) share of intraday variance falling after the snapshot (intraday seasonality). It changes only the session-basis RV annualization.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import iv_rv, realized_vol as rv, session_calendar as sc, spot_quality as sq

N = 375


def load_index(spot: str, participant_dir: str = "data/participant_oi") -> iv_rv.SessionIndex:
    df = pd.read_parquet(spot, columns=["ts", "open", "high", "low", "close"])
    cal = sc.SessionCalendar(sc.participant_oi_dates(participant_dir))
    sessions = sq.assess(df, cal)
    return iv_rv.SessionIndex(sessions, rv.daily_table(sessions))


def price_grid(open_: np.ndarray, close: np.ndarray) -> Optional[np.ndarray]:
    if len(open_) != N or len(close) != N or not (np.isfinite(open_[0]) and np.isfinite(close).all()):
        return None
    return np.concatenate([[open_[0]], close])


def sampled_intraday_var(P: np.ndarray, k: int) -> float:
    if N % k:
        raise ValueError("k must divide 375")
    g = P[::k]
    if len(g) != N // k + 1:
        g = np.concatenate([g, [P[-1]]]) if g[-1] != P[-1] else g
    r = np.diff(np.log(g))
    return float(np.sum(r * r))


def partial_var_sampled(close: np.ndarray, s: int, k: int) -> float:
    """Remainder of the snapshot session after the reference close (bar s), stepping k minutes, last step shortened to bar 374."""
    ref = s
    idxs = list(range(ref, 374, k)) + [374]
    pr = close[idxs]
    r = np.diff(np.log(pr))
    return float(np.sum(r * r))


def dev_profile(idx: iv_rv.SessionIndex, dev_end: date) -> np.ndarray:
    """Share of intraday variance in each of the 375 return positions (1-minute returns), averaged over complete development sessions (ratio of means)."""
    acc = np.zeros(N)
    n = 0
    for s in idx.chain:
        if s.quality.day > dev_end:
            continue
        P = price_grid(s.open, s.close)
        if P is None:
            continue
        r = np.diff(np.log(P))
        acc += r * r
        n += 1
    if n == 0:
        raise ValueError("no complete development session")
    return acc / acc.sum()


def remainder_weight(profile: np.ndarray, s: int) -> float:
    """Profile share of the returns strictly after the snapshot reference close: return positions s+2 .. 375 (1-indexed), i.e. array positions s+1 .. 374."""
    return float(profile[s + 1:].sum())


class WindowVariance:
    """Expiry-aligned hybrid total variance and session-equivalents under sampling step k and the weighting choice, for (day, hh:mm, expiry) triples."""

    def __init__(self, idx: iv_rv.SessionIndex, ks: Sequence[int] = (1, 5, 15), dev_end: date = date(2024, 12, 31)):
        self.idx = idx
        self.ks = tuple(ks)
        self.profile = dev_profile(idx, dev_end)
        chain = idx.chain
        n = len(chain)
        self.cum: Dict[int, np.ndarray] = {}
        for k in self.ks:
            per = np.full(n, np.nan)
            for j, s in enumerate(chain):
                P = price_grid(s.open, s.close)
                if P is None or j == 0 or not np.isfinite(chain[j - 1].close[-1]):
                    continue
                on = math.log(P[0] / chain[j - 1].close[-1])
                per[j] = sampled_intraday_var(P, k) + on * on
            self.cum[k] = per
        self._csum = {k: np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(v), v, 0.0))]) for k, v in self.cum.items()}
        self._bad = {k: np.concatenate([[0], np.cumsum(~np.isfinite(v))]) for k, v in self.cum.items()}

    def variance(self, day: date, hhmm: str, expiry: date, k: int) -> Optional[float]:
        idx = self.idx
        p, q = idx.pos.get(day), idx.pos.get(expiry)
        if p is None or q is None or q <= p:
            return None
        s = iv_rv.slot_of(hhmm)
        sess = idx.chain[p]
        if not np.isfinite(sess.close[s:]).all():
            return None
        if self._bad[k][q + 1] - self._bad[k][p + 1] > 0:
            return None
        part = partial_var_sampled(sess.close, s, k)
        return part + float(self._csum[k][q + 1] - self._csum[k][p + 1])

    def session_equivalents(self, day: date, hhmm: str, expiry: date, weighting: str = "uniform") -> Optional[float]:
        idx = self.idx
        p, q = idx.pos.get(day), idx.pos.get(expiry)
        if p is None or q is None or q <= p:
            return None
        s = iv_rv.slot_of(hhmm)
        n_part = 374 - s
        part = n_part / N if weighting == "uniform" else remainder_weight(self.profile, s)
        return part + (q - p)


def recompute(df: pd.DataFrame, V: Optional[np.ndarray] = None, S: Optional[np.ndarray] = None, session_days: float = 252.0, calendar_days: float = 365.0) -> pd.DataFrame:
    """Rebuild the outcome-side columns from stored quantities. calendar_days != 365 rescales IV consistently (implied total variance IV^2 T is invariant:
    T = seconds / (calendar_days * 86400) changes inversely), the calendar-basis RV, and nothing else."""
    d = df.copy()
    v = d.rv_total_variance.to_numpy(float) if V is None else np.asarray(V, float)
    s = d.session_equivalents.to_numpy(float) if S is None else np.asarray(S, float)
    d["rv_total_variance"] = v
    scale = math.sqrt(calendar_days / 365.0)
    iv = d.iv_pct.to_numpy(float) * scale
    d["iv_pct"] = iv
    with np.errstate(divide="ignore", invalid="ignore"):
        rv_sess = 100.0 * np.sqrt(session_days * v / s)
        rv_cal = 100.0 * np.sqrt(calendar_days * v / d.span_days.to_numpy(float))
        pos = v > 0
        d["rv_session_pct"] = rv_sess
        d["rv_calendar_pct"] = rv_cal
        d["spread_session"] = iv - rv_sess
        d["spread_calendar"] = iv - rv_cal
        d["total_variance_ratio_valid"] = pos
        d["total_variance_ratio"] = np.where(pos, d.implied_total_variance.to_numpy(float) / np.where(pos, v, 1.0), np.nan)
    d["total_variance_diff"] = d.implied_total_variance - d.rv_total_variance
    d["session_equivalents"] = s
    return d
