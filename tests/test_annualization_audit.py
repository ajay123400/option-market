"""Annualization-alignment audit: time conversion, window boundaries, identities and look-ahead protection."""
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from optionsengine.research import annualization_audit as aa, iv_rv
from tests.test_iv_rv import WD, alt, build, const_specs, make_index

IST = timezone(timedelta(hours=5, minutes=30))


# ----------------------------------------------------------------------------- ACT/365 time conversion
def test_T_is_act365_not_act3652425_and_not_trading_time():
    obs = datetime(2025, 9, 5, 10, 1, tzinfo=IST)                      # Friday 10:01 (snapshot 10:00 + 60 s)
    t = aa.expected_t_years(obs, date(2025, 9, 11))                    # Thursday expiry 15:30 (pre-2026-08-03 schedule)
    secs = (datetime(2025, 9, 11, 15, 30, tzinfo=IST) - obs).total_seconds()
    assert t == pytest.approx(secs / (365 * 86400), abs=1e-15)
    assert t != pytest.approx(secs / (365.25 * 86400), rel=1e-6)
    assert aa.act365_years(obs, obs + timedelta(days=1)) * 365 == pytest.approx(1.0, abs=1e-12)


def test_expiry_instant_follows_close_schedule_15_30_then_15_40():
    obs = datetime(2026, 7, 30, 10, 1, tzinfo=IST)
    before = aa.expected_t_years(obs, date(2026, 7, 31))
    obs2 = datetime(2026, 8, 3, 10, 1, tzinfo=IST)
    after = aa.expected_t_years(obs2, date(2026, 8, 3))                # same-day expiry 15:40
    assert before == pytest.approx((5 * 3600 + 29 * 60 + 24 * 3600) / (365 * 86400), abs=1e-14)
    assert after == pytest.approx((5 * 3600 + 39 * 60) / (365 * 86400), abs=1e-14)


def test_implied_total_variance_is_iv_squared_times_T():
    assert aa.implied_total_variance(20.0, 0.25) == pytest.approx(0.04 * 0.25, abs=1e-15)


# ----------------------------------------------------------------------------- RV bases and the exact relation
def test_session_and_calendar_rv_hand_values():
    V = 0.0004
    assert aa.session_rv_pct(V, 2.0) == pytest.approx(100 * math.sqrt(252 * V / 2.0), abs=1e-12)
    assert aa.calendar_rv_pct(V, 3.0) == pytest.approx(100 * math.sqrt(365 * V / 3.0), abs=1e-12)


@pytest.mark.parametrize("span,se", [(1.0, 1.0), (3.0, 1.0), (7.0, 5.0), (9.877, 9.877), (2.2, 0.35), (31.0, 20.0)])
def test_rv_session_equals_rv_calendar_times_sqrt_kappa(span, se):
    V = 3.7e-4
    k = aa.kappa(span, se)
    assert aa.session_rv_pct(V, se) == pytest.approx(aa.calendar_rv_pct(V, span) * math.sqrt(k), rel=1e-13)


def test_clocks_agree_when_kappa_is_one():
    se = 5.0
    span = se * 365.0 / 252.0                                            # exactly the calendar span of `se` average sessions
    assert aa.kappa(span, se) == pytest.approx(1.0, abs=1e-15)
    assert aa.session_rv_pct(2e-4, se) == pytest.approx(aa.calendar_rv_pct(2e-4, span), rel=1e-13)


def test_weekday_window_is_not_interchangeable_with_calendar_basis():
    # one session, one calendar day (Tue->Wed): session basis annualizes 1/252, calendar basis 1/365 -> different numbers
    assert aa.kappa(1.0, 1.0) == pytest.approx(252 / 365, abs=1e-15)
    assert aa.calendar_rv_pct(1e-4, 1.0) > aa.session_rv_pct(1e-4, 1.0)
    # Friday close -> Monday close: one session over three calendar days
    assert aa.kappa(3.0, 1.0) == pytest.approx(3 * 252 / 365, abs=1e-15)
    assert aa.calendar_rv_pct(1e-4, 3.0) < aa.session_rv_pct(1e-4, 1.0)


# ----------------------------------------------------------------------------- window spans from the real target builders
def _targets(day_specs, days, snap_idx, hhmm, expiry):
    rows, _ = build(day_specs)
    idx, _ = make_index(rows, days)
    d = days[snap_idx]
    return idx, d, iv_rv.expiry_aligned_target(idx, d, hhmm, expiry, "hybrid"), iv_rv.forward_sessions_target(idx, d, 1, "hybrid")


PRE = [date(2025, 3, 3) + timedelta(days=i) for i in range(10) if (date(2025, 3, 3) + timedelta(days=i)).weekday() < 5]   # Mon.. (pre-change)


@pytest.mark.parametrize("hhmm", ["10:00", "13:00", "15:00"])
@pytest.mark.parametrize("calendar,gap_minutes", [("pre", 0.0), ("post", 10.0)])
def test_expiry_window_starts_at_observation_and_equals_T(hhmm, calendar, gap_minutes):
    days = (PRE if calendar == "pre" else WD)[:5]
    idx, d, t, _ = _targets([(x, alt(), 0.0) for x in days], days, 0, hhmm, days[3])
    obs = datetime.fromisoformat(iv_rv.observation_ts(d, hhmm))
    start, end = datetime.fromisoformat(t.start_ts), datetime.fromisoformat(t.end_ts)
    assert start == obs                                                  # first return begins at bar start + 60 s, never earlier
    assert end == datetime(days[3].year, days[3].month, days[3].day, 15, 30, tzinfo=IST)
    T = aa.expected_t_years(obs, days[3])
    # window span == option T under the 15:30 expiry rule; under the 15:40 rule (expiry on/after 2026-08-03) T is exactly 10 minutes longer
    assert (T - aa.span_days(start, end) / 365.0) * 365.0 * 1440.0 == pytest.approx(gap_minutes, abs=1e-6)


def test_f1_window_runs_from_snapshot_session_close_to_next_close_weekend_and_holiday():
    # Mon..Fri sessions, then a Tuesday after a weekday holiday gap (Monday missing)
    days = [WD[0], WD[1], WD[2], WD[3], WD[4], WD[5 + 1]]               # WD[5] is the next Monday -> skipped (holiday)
    assert (days[5] - days[4]).days == 4
    idx, _, _, f_thu = _targets([(x, alt(), 0.0) for x in days], days, 3, "10:00", days[4])
    assert aa.span_days(datetime.fromisoformat(f_thu.start_ts), datetime.fromisoformat(f_thu.end_ts)) == pytest.approx(1.0)
    _, _, _, f_fri = _targets([(x, alt(), 0.0) for x in days], days, 4, "13:00", days[5])
    assert aa.span_days(datetime.fromisoformat(f_fri.start_ts), datetime.fromisoformat(f_fri.end_ts)) == pytest.approx(4.0)   # Fri -> Tue
    assert f_fri.session_equivalents == 1.0                              # same number of sessions, different calendar time


def test_five_session_window_spans_seven_calendar_days_without_holidays_and_more_with_one():
    days5 = WD[:7]
    rows, _ = build(const_specs(7))
    idx, _ = make_index(rows, days5)
    t = iv_rv.forward_sessions_target(idx, days5[0], 5, "hybrid")        # Mon close -> next Mon close
    assert aa.span_days(datetime.fromisoformat(t.start_ts), datetime.fromisoformat(t.end_ts)) == pytest.approx(7.0)
    days_h = [d for d in WD[:8] if d != WD[2]]                           # remove a Wednesday (holiday inside the window)
    rows, _ = build([(d, alt(), 0.0) for d in days_h])
    idx, _ = make_index(rows, days_h)
    t = iv_rv.forward_sessions_target(idx, days_h[0], 5, "hybrid")
    assert aa.span_days(datetime.fromisoformat(t.start_ts), datetime.fromisoformat(t.end_ts)) == pytest.approx(8.0)
    assert t.session_equivalents == 5.0                                  # 5 sessions over 8 days: kappa > 0.967
    assert aa.kappa(8.0, 5.0) > aa.kappa(7.0, 5.0)


# ----------------------------------------------------------------------------- build_aligned on a hand-made long table
def long_row(**kw):
    base = dict(obs_id="o", population="primary", split="dev", day="2026-09-14", time="10:00", observation_ts_ist="2026-09-14T10:01:00+05:30",
                expiry="2026-09-17", expiry_day=False, T_days=3.2284722222222224, dte_bucket="3-7d", iv_pct=20.0, horizon="EXP", measure="hybrid",
                target_available=True, unavailable_reason=None, future_rv_pct=100 * math.sqrt(252 * 0.0004 / 3.0), n_valid_sessions=3, n_returns=1000, session_equivalents=3.0,
                target_start_ts="2026-09-14T10:01:00+05:30", target_end_ts="2026-09-17T15:30:00+05:30", rv_total_variance=0.0004,
                snapshot_to_expiry_has_weekend=False, recent_rv_regime="mid", iv_quartile="Q2")
    base.update(kw)
    return base


def test_build_aligned_hand_values_for_expiry_row():
    T_days = (datetime(2026, 9, 17, 15, 30, tzinfo=IST) - datetime(2026, 9, 14, 10, 1, tzinfo=IST)).total_seconds() / 86400
    a = aa.build_aligned(pd.DataFrame([long_row(T_days=T_days)])).iloc[0]
    assert a.span_days == pytest.approx(T_days, abs=1e-12)
    assert a.implied_total_variance == pytest.approx(0.04 * T_days / 365, abs=1e-15)
    assert a.total_variance_diff == pytest.approx(0.04 * T_days / 365 - 0.0004, abs=1e-15)
    assert a.total_variance_ratio == pytest.approx(0.04 * T_days / 365 / 0.0004, abs=1e-12)
    assert a.rv_calendar_pct == pytest.approx(100 * math.sqrt(0.0004 / (T_days / 365)), abs=1e-10)
    assert a.spread_session == pytest.approx(20.0 - 100 * math.sqrt(252 * 0.0004 / 3.0), abs=1e-12)
    assert a.annualization_effect == pytest.approx(a.spread_session - a.spread_calendar, abs=1e-12)
    assert a.annualization_effect == pytest.approx(a.rv_calendar_pct * (1 - math.sqrt(a.kappa)), abs=1e-10)
    # (IV / RV_cal)^2 == W_imp / V exactly when span == T
    assert (20.0 / a.rv_calendar_pct) ** 2 == pytest.approx(a.total_variance_ratio, rel=1e-12)
    assert abs(a.t_minus_span_minutes) < 1e-6


def test_post_2026_08_03_expiry_reports_ten_unmatched_minutes():
    obs = datetime(2026, 8, 4, 10, 1, tzinfo=IST)
    T_days = (datetime(2026, 8, 5, 15, 40, tzinfo=IST) - obs).total_seconds() / 86400
    row = long_row(day="2026-08-04", observation_ts_ist=obs.isoformat(), expiry="2026-08-05", T_days=T_days,
                   target_start_ts=obs.isoformat(), target_end_ts="2026-08-05T15:30:00+05:30")
    a = aa.build_aligned(pd.DataFrame([row])).iloc[0]
    assert a.t_minus_span_minutes == pytest.approx(10.0, abs=1e-6)
    assert bool(a.expiry_after_close_change)


def test_intraday_rows_get_no_calendar_basis_and_are_labelled_subset():
    rows = [long_row(measure="intraday"), long_row(horizon="F1", measure="intraday", target_start_ts="2026-09-15T09:15:00+05:30",
                                                   target_end_ts="2026-09-15T15:30:00+05:30", session_equivalents=1.0)]
    a = aa.build_aligned(pd.DataFrame(rows))
    assert a.rv_calendar_pct.isna().all() and a.kappa.isna().all() and not a.interval_complete.any()
    assert a.total_variance_basis.iloc[0].startswith("subset")


def test_zero_realized_variance_makes_ratio_invalid_but_difference_valid():
    a = aa.build_aligned(pd.DataFrame([long_row(rv_total_variance=0.0, future_rv_pct=0.0)])).iloc[0]
    assert math.isnan(a.total_variance_ratio) and not a.total_variance_ratio_valid
    assert a.total_variance_diff == pytest.approx(a.implied_total_variance)
    assert a.rv_calendar_pct == 0.0


def test_unavailable_targets_are_dropped_not_filled():
    rows = [long_row(), long_row(target_available=False, future_rv_pct=None, rv_total_variance=None, session_equivalents=None,
                                 target_start_ts=None, target_end_ts=None)]
    assert len(aa.build_aligned(pd.DataFrame(rows))) == 1


def test_fixed_horizon_row_uses_window_span_and_flags_flat_term_structure():
    row = long_row(horizon="F1", measure="hybrid", session_equivalents=1.0, rv_total_variance=1e-4, future_rv_pct=100 * math.sqrt(252e-4),
                   target_start_ts="2026-09-18T15:30:00+05:30", target_end_ts="2026-09-21T15:30:00+05:30", day="2026-09-18")
    a = aa.build_aligned(pd.DataFrame([row])).iloc[0]
    assert a.span_days == pytest.approx(3.0)
    assert a.kappa == pytest.approx(3 * 252 / 365)
    assert a.implied_variance_basis.startswith("window_span")
    assert a.implied_total_variance == pytest.approx(0.04 * 3.0 / 365)


# ----------------------------------------------------------------------------- look-ahead protection
def test_implied_side_never_depends_on_outcome_columns():
    base = pd.DataFrame([long_row()])
    shifted = pd.DataFrame([long_row(future_rv_pct=99.0, rv_total_variance=0.5)])
    a, b = aa.build_aligned(base).iloc[0], aa.build_aligned(shifted).iloc[0]
    for col in ("T_years", "implied_total_variance", "iv_pct", "T_days", "dte_bucket", "expiry"):
        assert a[col] == b[col]
    assert a.rv_calendar_pct != b.rv_calendar_pct                          # outcome side did change


@pytest.mark.parametrize("hhmm", ["10:00", "13:00", "15:00"])
def test_total_variance_ignores_everything_at_or_before_the_snapshot_and_changes_with_the_future(hhmm):
    days = WD[:5]
    spec_a = const_specs(5)
    base_idx, d, base, _ = _targets(spec_a, days, 0, hhmm, days[3])
    s0 = iv_rv.slot_of(hhmm)
    rets = alt()
    for i in range(0, s0 + 1):
        rets[i] = 0.02 if i % 2 == 0 else -0.02                           # a violent past inside/before the snapshot bar
    spec_b = [(days[0], rets, 0.0)] + spec_a[1:]
    _, _, past_changed, _ = _targets(spec_b, days, 0, hhmm, days[3])
    assert past_changed.total_variance == pytest.approx(base.total_variance, rel=1e-12)
    rets_f = alt()
    rets_f[s0 + 5] = 0.03                                                 # a move strictly after the snapshot
    spec_c = [(days[0], rets_f, 0.0)] + spec_a[1:]
    _, _, fut_changed, _ = _targets(spec_c, days, 0, hhmm, days[3])
    assert fut_changed.total_variance > base.total_variance
