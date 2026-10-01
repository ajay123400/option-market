"""Annualization-alignment audit for Stage 2C (measurement only; nothing here is a signal or a strategy).

QUESTION
--------
Stage 2C compared an implied volatility (IV) annualized on CALENDAR time with a realized volatility (RV) annualized on a
252-SESSION year. How much of the observed "IV minus future RV" spread is produced mechanically by those two clocks?
This module builds an aligned comparison next to the original one. It reads the existing Stage 2C observation table
(`iv_rv_observations.csv`) and never changes it; it needs no new market data and no new IV.

CONVENTIONS (exact)
-------------------
* IV time:  T_years = (expiry instant - observation instant).total_seconds() / (365 * 86400)      [ACT/365, Phase 1
  `analytics.time_to_expiry_years`; the observation instant is bar START + 60 s; the expiry instant is 15:30 IST, or
  15:40 IST for expiries on/after 2026-08-03, `sessions.NSE_FNO_CLOSE_SCHEDULE`]. `T_days` in the tables is T_years * 365.
  Not ACT/365.25, not trading-day time.
* Implied total variance over the option's remaining life:      W_imp = (IV/100)^2 * T_years.
* Realized total variance over a window:                       V = sum of squared 1-minute (and, for `hybrid`, overnight)
  log returns inside the window (Stage 2B/2C definition, nothing filled, nothing re-estimated).
* Session-basis RV (ORIGINAL):   RV_sess = 100 * sqrt(252 * V / session_equivalents)         session_equivalents = n_partial/375 + sessions
* Calendar-basis RV (ALIGNED):   RV_cal  = 100 * sqrt(365 * V / span_days)                   span_days = (window end - window start) in days
  (span_days / 365 is the ACT/365 year fraction of the window's own elapsed calendar time).
* Exact relation:   RV_sess / RV_cal = sqrt(kappa),   kappa = (span_days/365) / (session_equivalents/252).
  kappa is "years of calendar time per year of session time" for that window. kappa = 1 means the two clocks agree.
  Hence  spread_sess - spread_cal = RV_cal - RV_sess = RV_cal * (1 - sqrt(kappa))   (per observation, exact).
* For the expiry-aligned target the window is [observation instant, expiry-session close 15:30]. Before 2026-08-03 its span
  equals T exactly, so  W_imp / V = (IV / RV_cal)^2  exactly. For expiries on/after 2026-08-03 the option's T runs to 15:40
  while spot stops at 15:30; the 10 unmatched minutes are reported (`t_minus_span_minutes`) and a sensitivity excludes those rows.

WHAT IS ALIGNED
---------------
* EXP / hybrid: W_imp versus V over the SAME interval: mathematically aligned (total variance is the primary diagnostic).
* EXP / intraday: V excludes every overnight/weekend/holiday interval, so it does not cover the interval T. Total-variance
  ratio is reported as a SUBSET quantity only; no calendar-basis RV is defined for it.
* F1/F5/F10/F20 hybrid and close_to_close: the RV window (from the snapshot-session close to the k-th session close) has its own
  calendar span; RV_cal annualizes it on that span. The IV is still the IV of an option expiring at T (a different horizon);
  the comparison of an annualized IV with an annualized RV over a different horizon stays a conventional comparison, and
  it implicitly assumes a flat term structure. Fixed-horizon targets are never merged with each other or with EXP.
* F intraday: not interval-complete (no overnight) -> no calendar-basis RV.

LOOK-AHEAD
----------
Everything here is an arithmetic transformation of (a) IV, T and the expiry date, all known at the snapshot, and (b) the
outcome V and the window timestamps from Stage 2C. No future spot value can reach IV, T, DTE or filtering. T and W_imp use
only the observation instant and the contract's expiry date/time (known contract metadata).
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..analytics import expiry_at_close, time_to_expiry_years

SESSION_DAYS = 252.0
CALENDAR_DAYS = 365.0
SECONDS_PER_DAY = 86400.0
INTERVAL_COMPLETE = {("EXP", "hybrid"), ("F1", "hybrid"), ("F5", "hybrid"), ("F10", "hybrid"), ("F20", "hybrid"),
                     ("F1", "close_to_close"), ("F5", "close_to_close"), ("F10", "close_to_close"), ("F20", "close_to_close")}
EXP_TOTAL_VARIANCE = {("EXP", "hybrid"): "aligned (same interval)", ("EXP", "intraday"): "subset only (overnight excluded)"}


# ------------------------------------------------------------------------------ pure conversions
def span_days(start: datetime, end: datetime) -> float:
    """Elapsed calendar days between two aware instants."""
    return (end - start).total_seconds() / SECONDS_PER_DAY


def act365_years(start: datetime, end: datetime) -> float:
    """ACT/365 year fraction (same arithmetic as the Phase 1 time-to-expiry; never negative)."""
    return max((end - start).total_seconds(), 0.0) / (CALENDAR_DAYS * SECONDS_PER_DAY)


def implied_total_variance(iv_pct: float, t_years: float) -> float:
    return (iv_pct / 100.0) ** 2 * t_years


def calendar_rv_pct(total_variance: float, span_in_days: float, calendar_days: float = CALENDAR_DAYS) -> float:
    """100 * sqrt(V / (span/365))."""
    return 100.0 * math.sqrt(total_variance / (span_in_days / calendar_days))


def session_rv_pct(total_variance: float, session_equivalents: float, session_days: float = SESSION_DAYS) -> float:
    return 100.0 * math.sqrt(session_days * total_variance / session_equivalents)


def kappa(span_in_days: float, session_equivalents: float) -> float:
    """(calendar years of the window) / (session years of the window); RV_sess = RV_cal * sqrt(kappa)."""
    return (span_in_days / CALENDAR_DAYS) / (session_equivalents / SESSION_DAYS)


def expected_t_years(observation_ts: datetime, expiry: date) -> float:
    """T exactly as Stage 2A/Phase 1 define it (ACT/365 to the scheduled expiry instant)."""
    return time_to_expiry_years(observation_ts, expiry_at_close(expiry))


# ------------------------------------------------------------------------------ table construction
def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def build_aligned(long: pd.DataFrame) -> pd.DataFrame:
    """One row per AVAILABLE (observation, horizon, measure) with the aligned columns. Input = Stage 2C long table (primary, available)."""
    a = long[long.target_available.astype(bool)].copy()
    obs_ts, start, end = _ts(a.observation_ts_ist), _ts(a.target_start_ts), _ts(a.target_end_ts)
    a["span_days"] = (end - start).dt.total_seconds() / SECONDS_PER_DAY
    key = list(zip(a.horizon, a.measure))
    a["interval_complete"] = [k in INTERVAL_COMPLETE for k in key]
    a["T_years"] = a.T_days / CALENDAR_DAYS
    a["kappa"] = np.where(a.interval_complete, (a.span_days / CALENDAR_DAYS) / (a.session_equivalents / SESSION_DAYS), np.nan)
    v = a.rv_total_variance.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cal = 100.0 * np.sqrt(v / (a.span_days.to_numpy() / CALENDAR_DAYS))
    a["rv_session_pct"] = a.future_rv_pct
    a["rv_calendar_pct"] = np.where(a.interval_complete, cal, np.nan)
    a["spread_session"] = a.iv_pct - a.rv_session_pct
    a["spread_calendar"] = a.iv_pct - a.rv_calendar_pct
    a["annualization_effect"] = a.spread_session - a.spread_calendar              # = RV_cal - RV_sess
    # implied variance scaled to the target window's own span (flat-term-structure assumption; F-targets) or to the option's T (EXP)
    is_exp = a.horizon.eq("EXP")
    a["implied_total_variance"] = np.where(is_exp, (a.iv_pct / 100.0) ** 2 * a.T_years, (a.iv_pct / 100.0) ** 2 * a.span_days / CALENDAR_DAYS)
    a["implied_variance_basis"] = np.where(is_exp, "option_T_actual_remaining_life", "window_span_flat_term_structure_assumed")
    a["total_variance_diff"] = a.implied_total_variance - a.rv_total_variance
    pos = a.rv_total_variance > 0
    a["total_variance_ratio"] = np.where(pos, a.implied_total_variance / a.rv_total_variance.where(pos, np.nan), np.nan)
    a["total_variance_ratio_valid"] = pos
    a["t_minus_span_minutes"] = np.where(is_exp, (a.T_years * CALENDAR_DAYS - a.span_days) * 1440.0, np.nan)
    a["expiry_after_close_change"] = a.expiry >= "2026-08-03"
    a["total_variance_basis"] = [EXP_TOTAL_VARIANCE.get(k, "window_span_only") for k in key]
    d0 = start.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    d1 = end.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
    a["weekend_in_window"] = (d0.dt.weekday + (d1 - d0).dt.days) >= 5               # the window reaches a Saturday
    keep = ["obs_id", "split", "day", "time", "expiry", "expiry_day", "dte_bucket", "T_days", "T_years", "iv_pct", "horizon", "measure",
            "interval_complete", "session_equivalents", "span_days", "kappa", "n_returns", "rv_total_variance", "rv_session_pct", "rv_calendar_pct",
            "spread_session", "spread_calendar", "annualization_effect", "implied_total_variance", "implied_variance_basis", "total_variance_diff",
            "total_variance_ratio", "total_variance_ratio_valid", "t_minus_span_minutes", "expiry_after_close_change", "total_variance_basis",
            "weekend_in_window", "recent_rv_regime", "iv_quartile", "target_start_ts", "target_end_ts"]
    keep = [c for c in keep if c in a.columns]
    return a[keep].reset_index(drop=True)
