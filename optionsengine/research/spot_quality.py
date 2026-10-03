"""Data-quality layer for the 1-minute spot dataset (Stage 2B). NO realized-volatility maths here.

Conventions (verified on the dataset, see research_output/stage2b/REPORT.md):
* `ts` = epoch seconds UTC, bar START. IST = UTC+05:30. Session date = IST date of ts.
* Regular session = bars starting 09:15 .. 15:29 IST = 375 bars; the 15:29 bar closes at 15:30 (index close).
* Bars outside that window (pre-open 09:08-09:14, Muhurat evenings, ...) are COUNTED but not used for
  regular-session statistics.

What is checked per IST date (all counts recorded, nothing silently dropped):
  out_of_order       records whose ts is lower than the previous record's (raw file order)
  duplicate_exact    repeated ts with identical OHLC  -> collapsed to one bar, counted
  duplicate_conflict repeated ts with different OHLC  -> that minute is treated as MISSING
  invalid_ohlc       non-finite/<=0 price, high<low, open or close outside [low, high] -> minute treated as MISSING
  missing_minutes    expected regular minutes with no valid bar (never forward-filled)
Session types: regular | special_weekend | special_offhours (no regular-window bars) |
special_short (weekday, fewer than `special_max_bars` regular-window bars, e.g. Muhurat) |
missing_session (calendar expected a trading date but there are no bars at all).
A session is `complete` only if it is `regular`, has all 375 valid bars and no out-of-order or
conflicting-duplicate records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .session_calendar import SessionCalendar

IST_OFFSET_SECONDS = 19800
OPEN_MINUTE = 9 * 60 + 15
N_REGULAR = 375                      # 09:15 .. 15:29
SPECIAL_MAX_BARS = 150               # fewer regular-window bars than this on a weekday => special_short
EPOCH = date(1970, 1, 1)


def minute_label(m: Optional[int]) -> Optional[str]:
    return None if m is None else f"{m // 60:02d}:{m % 60:02d}"


@dataclass
class SessionData:
    """Valid regular-window bars of one date on the 375-minute grid (NaN where missing/invalid)."""
    day: date
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quality: "SessionQuality"


@dataclass
class SessionQuality:
    day: date
    session_type: str
    expected_trading_date: Optional[bool]
    calendar_source: str
    n_records: int = 0
    n_regular_window: int = 0
    n_outside_window: int = 0
    n_out_of_order: int = 0
    n_duplicate_exact: int = 0
    n_duplicate_conflict: int = 0
    n_invalid_ohlc: int = 0
    n_valid_bars: int = 0
    n_missing_minutes: int = N_REGULAR
    longest_gap_minutes: int = 0
    first_bar: Optional[str] = None
    last_bar: Optional[str] = None
    complete: bool = False
    in_sequence: bool = False          # part of the regular-session chain used for close-to-close/rolling
    reasons: List[str] = field(default_factory=list)


def _valid_mask(o, h, l, c) -> np.ndarray:
    ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    with np.errstate(invalid="ignore"):
        ok &= (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l) & (o >= l) & (o <= h) & (c >= l) & (c <= h)
    return ok


def assess(df: pd.DataFrame, calendar: SessionCalendar, special_max_bars: int = SPECIAL_MAX_BARS) -> List[SessionData]:
    """Split `df` (columns ts, open, high, low, close in RAW file order) into IST-date sessions and run every check.
    Returns one SessionData per date with bars, plus `missing_session` entries for expected dates without bars,
    ordered by date. Marks which sessions form the regular chain (`in_sequence`)."""
    ts_all = df["ts"].to_numpy(dtype="int64")
    o_all, h_all = df["open"].to_numpy(dtype="float64"), df["high"].to_numpy(dtype="float64")
    l_all, c_all = df["low"].to_numpy(dtype="float64"), df["close"].to_numpy(dtype="float64")
    ist = ts_all + IST_OFFSET_SECONDS
    day_no = ist // 86400
    minute_of_day = (ist % 86400) // 60
    sessions: Dict[date, SessionData] = {}
    order_break = np.zeros(len(df), dtype=bool)
    if len(df) > 1:
        order_break[1:] = ts_all[1:] < ts_all[:-1]

    for d_no in np.unique(day_no):
        idx = np.nonzero(day_no == d_no)[0]
        d = EPOCH + timedelta(days=int(d_no))
        exp, src = calendar.expected_trading_date(d)
        q = SessionQuality(d, "regular", exp, src, n_records=len(idx))
        q.n_out_of_order = int(order_break[idx].sum())
        ts, mod = ts_all[idx], minute_of_day[idx]
        sort = np.argsort(ts, kind="stable")                       # sorted copy for processing; the flag above keeps the evidence
        idx, ts, mod = idx[sort], ts[sort], mod[sort]
        # ---- duplicates (same ts) --------------------------------------------------
        keep = np.ones(len(idx), dtype=bool)
        conflict_minutes = set()
        if len(idx) > 1:
            same = ts[1:] == ts[:-1]
            for j in np.nonzero(same)[0] + 1:
                exact = (o_all[idx[j]] == o_all[idx[j - 1]] and h_all[idx[j]] == h_all[idx[j - 1]] and
                         l_all[idx[j]] == l_all[idx[j - 1]] and c_all[idx[j]] == c_all[idx[j - 1]])
                keep[j] = False
                if exact:
                    q.n_duplicate_exact += 1
                else:
                    q.n_duplicate_conflict += 1
                    conflict_minutes.add(int(mod[j]))
        idx, mod = idx[keep], mod[keep]
        # ---- window split ------------------------------------------------------------
        in_win = (mod >= OPEN_MINUTE) & (mod < OPEN_MINUTE + N_REGULAR)
        q.n_outside_window = int((~in_win).sum())
        q.n_regular_window = int(in_win.sum())
        widx, wmod = idx[in_win], mod[in_win]
        o, h, l, c = (np.full(N_REGULAR, np.nan) for _ in range(4))
        slot = (wmod - OPEN_MINUTE).astype(int)
        valid = _valid_mask(o_all[widx], h_all[widx], l_all[widx], c_all[widx])
        q.n_invalid_ohlc = int((~valid).sum())
        for m in conflict_minutes:                                  # conflicting duplicates: minute is not trustworthy
            if OPEN_MINUTE <= m < OPEN_MINUTE + N_REGULAR:
                valid[slot == (m - OPEN_MINUTE)] = False
        o[slot[valid]], h[slot[valid]] = o_all[widx][valid], h_all[widx][valid]
        l[slot[valid]], c[slot[valid]] = l_all[widx][valid], c_all[widx][valid]
        present = np.isfinite(c)
        q.n_valid_bars = int(present.sum())
        q.n_missing_minutes = N_REGULAR - q.n_valid_bars
        if q.n_valid_bars:
            q.first_bar, q.last_bar = minute_label(OPEN_MINUTE + int(np.argmax(present))), minute_label(OPEN_MINUTE + N_REGULAR - 1 - int(np.argmax(present[::-1])))
        # longest run of consecutive missing minutes inside the window
        run = best = 0
        for pr in present:
            run = 0 if pr else run + 1
            best = max(best, run)
        q.longest_gap_minutes = best
        # ---- session type ---------------------------------------------------------------
        if d.weekday() >= 5:
            q.session_type = "special_weekend"
        elif q.n_regular_window == 0:
            q.session_type = "special_offhours"
        elif q.n_regular_window < special_max_bars:
            q.session_type = "special_short"
        else:
            q.session_type = "regular"
        q.in_sequence = q.session_type == "regular"
        if q.session_type != "regular":
            q.reasons.append(q.session_type)
        if q.n_outside_window:
            q.reasons.append("bars_outside_regular_window")
        for cond, name in ((q.n_out_of_order, "out_of_order"), (q.n_duplicate_exact, "duplicate_exact"),
                           (q.n_duplicate_conflict, "duplicate_conflict"), (q.n_invalid_ohlc, "invalid_ohlc")):
            if cond:
                q.reasons.append(name)
        if q.session_type == "regular" and q.n_missing_minutes:
            q.reasons.append("missing_minutes")
        q.complete = (q.session_type == "regular" and q.n_valid_bars == N_REGULAR and q.n_out_of_order == 0
                      and q.n_duplicate_conflict == 0)
        sessions[d] = SessionData(d, o, h, l, c, q)

    # ---- calendar: expected trading dates with no bars at all ---------------------------------
    if sessions:
        lo, hi = min(sessions), max(sessions)
        d = lo
        while d <= hi:
            if d not in sessions:
                exp, src = calendar.expected_trading_date(d)
                if exp:
                    q = SessionQuality(d, "missing_session", True, src, reasons=["missing_session"])
                    q.in_sequence = True       # a regular session was due: the chain is broken here (not skipped)
                    nan = np.full(N_REGULAR, np.nan)
                    sessions[d] = SessionData(d, nan, nan.copy(), nan.copy(), nan.copy(), q)
            d += timedelta(days=1)
    # calendar conflict: a 'regular' session on a date the calendar does not expect (holiday)
    for s in sessions.values():
        if s.quality.session_type == "regular" and s.quality.expected_trading_date is False:
            s.quality.reasons.append("calendar_conflict_unexpected_date")
    return [sessions[k] for k in sorted(sessions)]
