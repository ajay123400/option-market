"""Future realized-volatility TARGETS for Stage 2C (measurement only; nothing here is a signal or a strategy).

LOOK-AHEAD RULES (enforced by construction and by tests)
-------------------------------------------------------
* A snapshot is taken at bar START t0 (10:00, 13:00 or 15:00 IST). Its observation point is t0 + 60 s (the bar
  has closed). Every FEATURE (ATM IV, forward, spot, recent RV, DTE) is computed only from information at or
  before that point -- Stage 2A guarantees this for IV (bars ts <= t0), `recent_rv` below uses only sessions
  that completed BEFORE the snapshot date.
* TARGETS (future RV) use only returns whose price interval lies strictly after the observation point:
    - the reference price is the close of the snapshot bar (known at t0 + 60 s); the first target return is
      ln(close[t0 + 1 min] / close[t0]), which spans [t0 + 60 s, t0 + 120 s);
    - returns inside the snapshot bar or earlier never enter a target.
  Targets are OUTCOMES; they never feed IV, snapshot selection or any feature.
* The only thing known at t about the future is the contract's fixed expiry date/time.

TARGET FAMILIES
---------------
A. FULL-SESSION targets F1, F5, F10, F20: the next k complete regular sessions AFTER the snapshot session
   (sessions t+1 .. t+k of the Stage 2B regular-session chain). The remainder of the snapshot day (everything
   after the snapshot) is NOT part of them, so a 13:00 snapshot is never compared with a same-day full-session RV.
   measure = hybrid           overnight (previous close -> open) + intraday 1-minute returns (overnight INCLUDED)
   measure = close_to_close   daily close-to-close returns (overnight INCLUDED)
   measure = intraday         1-minute returns only (overnight EXCLUDED)
   The overnight return into session t+1 starts at the close of the snapshot session (15:30), strictly after
   the snapshot. Annualization: 100*sqrt(252 * mean variance per session) -- the Stage 2B convention.
   Strict only: any missing/incomplete session or missing return => target unavailable (nothing is filled).
B. EXPIRY-ALIGNED target EXP (hybrid and intraday): realized variance from the snapshot reference price to the
   close (15:29 bar, 15:30) of the expiry-date session:
       partial remainder of the snapshot session (returns strictly after the snapshot, slots s0+1 .. 374)
     + for every regular session u with t < u <= expiry: overnight return (hybrid only) + 375 intraday returns.
   Annualized with SESSION-EQUIVALENTS = n_partial/375 + number of later sessions (uniform trading-time
   convention for the partial session; intraday seasonality makes this approximate -- see the report's
   sensitivity). Also reported as TOTAL variance (sum of squared returns) so it can be compared with
   IV^2 * T without any annualization convention.
   Not constructed (unavailable, reason recorded) when: the snapshot is on the expiry day (only a partial
   session would remain), the snapshot or expiry session is not in the regular chain, any session in the window
   is incomplete/missing, or the expiry session lies beyond the end of the spot data. Special sessions are not in
   the chain: they neither count nor break it (their price moves lie inside the overnight returns).
   The option's expiry instant for T is 15:40 IST from 2026-08-03 but the index (spot) ends at 15:30 every day,
   so for post-change expiries the last 10 minutes of T cannot be matched by any spot return.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .spot_quality import N_REGULAR, OPEN_MINUTE, SessionData

IST = timezone(timedelta(hours=5, minutes=30))
HORIZONS = {"F1": 1, "F5": 5, "F10": 10, "F20": 20}
FIXED_MEASURES = ("hybrid", "close_to_close", "intraday")
EXPIRY_MEASURES = ("hybrid", "intraday")
VAR_COLUMN = {"hybrid": "hybrid_variance", "close_to_close": "cc_variance", "intraday": "intraday_variance"}
ANNUALIZATION_DAYS = 252.0


def slot_of(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    slot = h * 60 + m - OPEN_MINUTE
    if not 0 <= slot < N_REGULAR:
        raise ValueError(f"{hhmm} is outside the regular session")
    return slot


def _ts(day: date, hhmm: str) -> str:
    return f"{day.isoformat()}T{hhmm}:00+05:30"


def observation_ts(day: date, hhmm: str) -> str:
    """Snapshot bar START + 60 s: the earliest instant at which the snapshot bar's close is known."""
    h, m = (int(x) for x in hhmm.split(":"))
    t = datetime(day.year, day.month, day.day, h, m, tzinfo=IST) + timedelta(seconds=60)
    return t.isoformat()


@dataclass(frozen=True)
class Target:
    available: bool
    reason: Optional[str]
    rv_pct: Optional[float] = None
    total_variance: Optional[float] = None
    n_sessions: int = 0
    n_returns: int = 0
    session_equivalents: Optional[float] = None
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    includes_overnight: Optional[bool] = None
    includes_partial_first_session: bool = False


def _unavail(reason: str, overnight: Optional[bool] = None) -> Target:
    return Target(False, reason, includes_overnight=overnight)


class SessionIndex:
    """Regular-session chain + per-session variances (from the Stage 2B daily table)."""

    def __init__(self, sessions: Sequence[SessionData], daily: pd.DataFrame):
        self.by_date: Dict[date, SessionData] = {s.quality.day: s for s in sessions}
        self.chain: List[SessionData] = [s for s in sessions if s.quality.in_sequence]
        self.chain_dates: List[date] = [s.quality.day for s in self.chain]
        self.pos: Dict[date, int] = {d: i for i, d in enumerate(self.chain_dates)}
        d = daily.set_index("session_date")
        self.var: Dict[str, Dict[date, float]] = {}
        for measure, col in VAR_COLUMN.items():
            ser = d[col] if col in d.columns else pd.Series(dtype=float)
            self.var[measure] = {date.fromisoformat(k): float(v) for k, v in ser.items() if pd.notna(v)}
        self.last_day = self.chain_dates[-1] if self.chain_dates else None


def _ann(total_var: float, sessions_equiv: float, A: float) -> float:
    return 100.0 * math.sqrt(A * total_var / sessions_equiv)


def forward_sessions_target(idx: SessionIndex, snap_day: date, k: int, measure: str, A: float = ANNUALIZATION_DAYS) -> Target:
    """Family A: the next k complete regular sessions strictly after the snapshot session."""
    overnight = measure != "intraday"
    if snap_day not in idx.pos:
        return _unavail("snapshot_session_not_regular", overnight)
    p = idx.pos[snap_day]
    if p + k >= len(idx.chain):
        return _unavail("beyond_data_end", overnight)
    days = idx.chain_dates[p + 1:p + 1 + k]
    vals = []
    for d in days:
        v = idx.var[measure].get(d)
        if v is None:
            return _unavail("incomplete_or_missing_session_in_window", overnight)
        vals.append(v)
    start = _ts(snap_day, "15:30") if overnight else _ts(days[0], "09:15")
    n_ret = k * (N_REGULAR if measure != "close_to_close" else 1) + (k if measure == "hybrid" else 0)
    tv = float(np.sum(vals))
    return Target(True, None, _ann(tv, k, A), tv, k, n_ret, float(k), start, _ts(days[-1], "15:30"), overnight, False)


def partial_returns(session: SessionData, slot0: int) -> Optional[np.ndarray]:
    """Minute returns strictly after the snapshot bar `slot0`: ln(close_i / close_{i-1}), i = slot0+1 .. 374.
    Strict: every close from slot0 to 374 must be valid, else None (nothing is filled)."""
    c = session.close[slot0:]
    if not np.isfinite(c).all():
        return None
    return np.log(c[1:] / c[:-1])


def expiry_aligned_target(idx: SessionIndex, snap_day: date, hhmm: str, expiry_day: date, measure: str,
                          A: float = ANNUALIZATION_DAYS) -> Target:
    """Family B: remaining life from the snapshot reference price to the close of the expiry-date session."""
    overnight = measure == "hybrid"
    if expiry_day <= snap_day:
        return _unavail("expiry_day_partial_session_only" if expiry_day == snap_day else "expiry_before_snapshot", overnight)
    if snap_day not in idx.pos:
        return _unavail("snapshot_session_not_regular", overnight)
    if expiry_day not in idx.pos:
        if idx.last_day is not None and expiry_day > idx.last_day:
            return _unavail("beyond_data_end", overnight)
        return _unavail("expiry_session_not_in_regular_chain", overnight)
    p, q = idx.pos[snap_day], idx.pos[expiry_day]
    slot0 = slot_of(hhmm)
    part = partial_returns(idx.by_date[snap_day], slot0)
    if part is None:
        return _unavail("snapshot_session_bars_missing_after_snapshot", overnight)
    total = float(np.sum(part ** 2))
    n_ret = len(part)
    for d in idx.chain_dates[p + 1:q + 1]:
        v = idx.var[measure].get(d)
        if v is None:
            return _unavail("incomplete_or_missing_session_in_window", overnight)
        total += v
        n_ret += N_REGULAR + (1 if overnight else 0)
    n_later = q - p
    equiv = len(part) / N_REGULAR + n_later
    return Target(True, None, _ann(total, equiv, A), total, n_later, n_ret, equiv, observation_ts(snap_day, hhmm),
                  _ts(expiry_day, "15:30"), overnight, True)


def recent_rv_before(rolling: pd.DataFrame, chain_dates: Sequence[date], snap_day: date) -> Optional[float]:
    """Strict hybrid 20-session RV as of the last session that COMPLETED before the snapshot date (known at t)."""
    i = bisect_right(chain_dates, snap_day) - 1
    if i >= 0 and chain_dates[i] == snap_day:
        i -= 1
    if i < 0:
        return None
    v = rolling.get(chain_dates[i])
    return None if v is None or not np.isfinite(v) else float(v)
