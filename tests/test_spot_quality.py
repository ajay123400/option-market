"""Data-quality layer: every anomaly is detected, counted and never repaired by invention."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from optionsengine.research import session_calendar as sc, spot_quality as sq
from tests.rv_synth import frame, session_rows, ts_of

D1, D2 = date(2026, 9, 21), date(2026, 9, 22)            # Mon, Tue


def cal(*days):
    return sc.SessionCalendar(set(days), fallback=lambda d: False)


def full(day, base=100.0):
    return session_rows(day, [base + 0.01 * i for i in range(375)])


def assess(rows, *days, **kw):
    return sq.assess(frame(rows), cal(*days), **kw)


def q(rows, *days, day=None):
    s = assess(rows, *days)
    return {x.quality.day: x for x in s}[day or days[0]]


def test_complete_session_and_timestamp_mapping():
    s = q(full(D1), D1)
    assert s.quality.complete and s.quality.session_type == "regular" and s.quality.in_sequence
    assert s.quality.n_valid_bars == 375 and s.quality.n_missing_minutes == 0 and s.quality.longest_gap_minutes == 0
    assert (s.quality.first_bar, s.quality.last_bar) == ("09:15", "15:29")
    assert s.close[0] == pytest.approx(100.0) and s.close[374] == pytest.approx(100.0 + 3.74)   # bar START labels map to slots


def test_utc_date_differs_from_ist_date():
    # 23:45 IST on 21 Sep == 18:15 UTC the same day; 00:10 IST on 22 Sep == 18:40 UTC on 21 Sep -> must be the 22nd in IST
    rows = [(ts_of(D1, 23 * 60 + 45), 1.0, 1.0, 1.0, 1.0), (ts_of(D2, 10), 1.0, 1.0, 1.0, 1.0)]
    s = assess(rows, D1, D2)
    assert [x.quality.day for x in s] == [D1, D2]


def test_missing_minutes_are_nan_not_filled():
    closes = [100.0 + 0.01 * i for i in range(375)]
    closes[100] = None
    closes[200] = closes[201] = closes[202] = None
    s = q(session_rows(D1, closes), D1)
    assert s.quality.n_valid_bars == 371 and s.quality.n_missing_minutes == 4 and s.quality.longest_gap_minutes == 3
    assert np.isnan(s.close[100]) and np.isnan(s.close[201]) and not s.quality.complete
    assert "missing_minutes" in s.quality.reasons


def test_exact_duplicates_collapse_and_are_counted():
    rows = full(D1)
    rows = rows[:11] + [rows[10]] + rows[11:12] + [rows[11]] + rows[12:]        # duplicates adjacent, file stays time-ordered
    s = q(rows, D1)
    assert s.quality.n_duplicate_exact == 2 and s.quality.n_duplicate_conflict == 0 and s.quality.n_valid_bars == 375
    assert s.quality.complete and "duplicate_exact" in s.quality.reasons


def test_conflicting_duplicates_make_the_minute_untrustworthy():
    rows = full(D1)
    t, o, h, l, c = rows[50]
    rows = rows + [(t, o, h + 5, l, c + 3)]
    s = q(rows, D1)
    assert s.quality.n_duplicate_conflict == 1 and np.isnan(s.close[50]) and not s.quality.complete
    assert s.quality.n_valid_bars == 374


def test_out_of_order_records_are_flagged_and_session_not_complete():
    rows = full(D1)
    rows[20], rows[21] = rows[21], rows[20]
    s = q(rows, D1)
    assert s.quality.n_out_of_order == 1 and not s.quality.complete and s.quality.n_valid_bars == 375
    assert s.close[20] == pytest.approx(100.20) and s.close[21] == pytest.approx(100.21)        # processed in time order


@pytest.mark.parametrize("bad", [
    lambda t, o, h, l, c: (t, o, l - 1, l, c),                # high < low
    lambda t, o, h, l, c: (t, float("nan"), h, l, c),
    lambda t, o, h, l, c: (t, o, h, l, -5.0),
    lambda t, o, h, l, c: (t, o, h, l, 0.0),
    lambda t, o, h, l, c: (t, h + 1, h, l, c),                # open above high
    lambda t, o, h, l, c: (t, o, h, l, l - 1),                # close below low
    lambda t, o, h, l, c: (t, o, float("inf"), l, c),
])
def test_invalid_ohlc_is_excluded_not_repaired(bad):
    rows = full(D1)
    rows[30] = bad(*rows[30])
    s = q(rows, D1)
    assert s.quality.n_invalid_ohlc == 1 and np.isnan(s.close[30]) and s.quality.n_valid_bars == 374 and not s.quality.complete


def test_pre_open_and_after_hours_bars_are_counted_but_not_used():
    rows = sorted(full(D1) + session_rows(D1, [99.0] * 5, start_minute=9 * 60 + 10) + session_rows(D1, [101.0] * 3, start_minute=15 * 60 + 31))
    s = q(rows, D1)
    assert s.quality.n_outside_window == 8 and s.quality.n_valid_bars == 375 and s.quality.complete
    assert "bars_outside_regular_window" in s.quality.reasons


def test_appended_early_timestamps_are_detected_as_out_of_order():
    rows = full(D1) + session_rows(D1, [99.0] * 2, start_minute=9 * 60 + 10)    # earlier times after later ones in the file
    s = q(rows, D1)
    assert s.quality.n_out_of_order == 1 and not s.quality.complete


def test_weekend_special_session():
    sat = date(2026, 9, 19)
    s = {x.quality.day: x for x in assess(session_rows(sat, [100.0] * 45) + full(D1), D1)}[sat]
    assert s.quality.session_type == "special_weekend" and not s.quality.in_sequence and not s.quality.complete


def test_evening_muhurat_and_short_sessions_are_special():
    eve = session_rows(D1, [100.0] * 60, start_minute=18 * 60 + 15)
    s = q(eve, D1)
    assert s.quality.session_type == "special_offhours" and s.quality.n_regular_window == 0 and s.quality.n_outside_window == 60
    short = session_rows(D2, [100.0] * 61, start_minute=13 * 60 + 45)          # 13:45-14:45 Muhurat
    s2 = q(short, D2)
    assert s2.quality.session_type == "special_short" and not s2.quality.in_sequence and not s2.quality.complete


def test_special_threshold_is_configurable():
    rows = session_rows(D1, [100.0] * 200)
    assert q(rows, D1).quality.session_type == "regular"
    s = {x.quality.day: x for x in sq.assess(frame(rows), cal(D1), special_max_bars=250)}[D1]
    assert s.quality.session_type == "special_short"


def test_missing_session_placeholder_when_calendar_expects_a_date():
    rows = full(D1) + full(date(2026, 9, 23))                  # Mon and Wed present; Tue 22nd expected but absent
    s = {x.quality.day: x for x in assess(rows, D1, D2, date(2026, 9, 23))}
    miss = s[D2]
    assert miss.quality.session_type == "missing_session" and miss.quality.in_sequence and not np.isfinite(miss.close).any()
    # a date the calendar does NOT expect (holiday) gets no placeholder
    s2 = {x.quality.day: x for x in assess(rows, D1, date(2026, 9, 23))}
    assert D2 not in s2


def test_calendar_conflict_is_reported():
    s = q(full(D1), date(2026, 9, 20), day=D1)                 # data on a date the calendar evidence does not list
    assert "calendar_conflict_unexpected_date" in s.quality.reasons
