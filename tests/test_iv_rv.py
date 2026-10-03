"""Stage 2C targets: hand-calculable synthetic sessions; boundary, look-ahead and zero-RV behaviour."""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optionsengine.research import iv_rv, realized_vol as rv, session_calendar as sc, spot_quality as sq
from tests.rv_synth import frame, path_from_returns, session_rows

A = 252.0
WD = [date(2026, 9, 14) + timedelta(days=i) for i in range(40) if (date(2026, 9, 14) + timedelta(days=i)).weekday() < 5]


def build(day_specs, extra_rows=()):
    """day_specs: list of (day, returns375, overnight_log_gap or 0). Prices chain across days. Returns (rows, closes_by_day)."""
    rows, price, closes_by_day = [], 100.0, {}
    for d, rets, gap in day_specs:
        start = price * math.exp(gap)
        closes = path_from_returns(start, rets)
        rows += session_rows(d, closes, opens=[start] + closes[:-1])
        closes_by_day[d] = (start, closes)
        price = closes[-1]
    rows = sorted(rows + list(extra_rows))
    return rows, closes_by_day


def make_index(rows, days):
    cal = sc.SessionCalendar(set(days), fallback=lambda d: False)
    sessions = sq.assess(frame(rows), cal)
    daily = rv.daily_table(sessions)
    return iv_rv.SessionIndex(sessions, daily), daily


def alt(n=375, r=0.001):
    return [r if i % 2 == 0 else -r for i in range(n)]


def const_specs(n_days, rets=None, gaps=None):
    return [(WD[i], rets[i] if rets else alt(), (gaps or {}).get(i, 0.0)) for i in range(n_days)]


# ----------------------------------------------------------------------------- partial session boundaries
@pytest.mark.parametrize("hhmm,slot,n", [("10:00", 45, 329), ("13:00", 225, 149), ("15:00", 345, 29)])
def test_first_return_is_strictly_after_the_snapshot_bar(hhmm, slot, n):
    spec = const_specs(1)
    rows, cl = build(spec)
    idx, _ = make_index(rows, WD[:1])
    assert iv_rv.slot_of(hhmm) == slot
    part = iv_rv.partial_returns(idx.by_date[WD[0]], slot)
    assert len(part) == n
    closes = cl[WD[0]][1]
    assert part[0] == pytest.approx(math.log(closes[slot + 1] / closes[slot]), abs=1e-14)       # first return starts at the reference close
    assert part[-1] == pytest.approx(math.log(closes[374] / closes[373]), abs=1e-14)


def test_returns_at_or_before_the_snapshot_bar_never_enter_the_target():
    rets = alt()
    rets[45] = 0.05                                    # a huge move INSIDE the 10:00 bar (open->close) and
    rets[44] = -0.04                                   # one in the bar before it
    rows, _ = build([(WD[0], rets, 0.0), (WD[1], alt(), 0.0)])
    idx, _ = make_index(rows, WD[:2])
    t = iv_rv.expiry_aligned_target(idx, WD[0], "10:00", WD[1], "intraday")
    expected_total = 329 * 1e-6 + 375 * 1e-6
    assert t.available and t.total_variance == pytest.approx(expected_total, rel=1e-9)
    assert t.n_returns == 329 + 375 and t.includes_partial_first_session and t.session_equivalents == pytest.approx(329 / 375 + 1)


def test_snapshot_bar_close_is_the_reference_price_only():
    rets = alt()
    rows1, _ = build([(WD[0], rets, 0.0), (WD[1], alt(), 0.0)])
    rets2 = list(rets); rets2[100] = 0.02              # AFTER 10:00: must change the target
    rows2, _ = build([(WD[0], rets2, 0.0), (WD[1], alt(), 0.0)])
    i1, _ = make_index(rows1, WD[:2]); i2, _ = make_index(rows2, WD[:2])
    t1 = iv_rv.expiry_aligned_target(i1, WD[0], "10:00", WD[1], "intraday")
    t2 = iv_rv.expiry_aligned_target(i2, WD[0], "10:00", WD[1], "intraday")
    assert t2.total_variance > t1.total_variance


# ----------------------------------------------------------------------------- family A (full sessions after the snapshot)
def test_forward_1_and_5_session_hand_values_and_overnight_flag():
    vols = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010, 0.0011]
    gaps = {1: 0.01, 3: -0.02}                          # overnight moves into sessions 1 and 3
    rows, _ = build([(WD[i], alt(375, vols[i]), gaps.get(i, 0.0)) for i in range(7)])
    idx, daily = make_index(rows, WD[:7])
    f1 = iv_rv.forward_sessions_target(idx, WD[0], 1, "hybrid")
    assert f1.available and f1.includes_overnight and f1.n_sessions == 1 and not f1.includes_partial_first_session
    assert f1.total_variance == pytest.approx(0.01 ** 2 + 375 * vols[1] ** 2, rel=1e-9)
    assert f1.rv_pct == pytest.approx(100 * math.sqrt(A * (0.01 ** 2 + 375 * vols[1] ** 2)), rel=1e-9)
    f1i = iv_rv.forward_sessions_target(idx, WD[0], 1, "intraday")
    assert not f1i.includes_overnight and f1i.total_variance == pytest.approx(375 * vols[1] ** 2, rel=1e-9)
    assert f1i.start_ts == "2026-09-15T09:15:00+05:30" and f1.start_ts == "2026-09-14T15:30:00+05:30"
    f5 = iv_rv.forward_sessions_target(idx, WD[0], 5, "hybrid")
    tv = sum(375 * v ** 2 for v in vols[1:6]) + 0.01 ** 2 + 0.02 ** 2
    assert f5.available and f5.total_variance == pytest.approx(tv, rel=1e-9) and f5.n_returns == 5 * 376
    assert f5.rv_pct == pytest.approx(100 * math.sqrt(A * tv / 5), rel=1e-9) and f5.end_ts == "2026-09-21T15:30:00+05:30"
    cc = iv_rv.forward_sessions_target(idx, WD[0], 5, "close_to_close")
    ccs = daily.cc_return.to_numpy()[1:6]
    assert cc.total_variance == pytest.approx(float(np.sum(ccs ** 2)), rel=1e-9) and cc.n_returns == 5


def test_snapshot_time_does_not_change_family_a_targets():
    rows, _ = build(const_specs(8))
    idx, _ = make_index(rows, WD[:8])
    a = [iv_rv.forward_sessions_target(idx, WD[1], 5, "hybrid") for _ in ("10:00", "13:00", "15:00")]
    assert a[0] == a[1] == a[2]                          # overlapping snapshots share identical full-session targets


def test_missing_future_bar_makes_only_the_affected_horizons_unavailable():
    rows, _ = build(const_specs(8))
    drop_ts = session_rows(WD[3], [1.0] * 301)[300][0]                      # a minute inside session 3
    rows = [r for r in rows if r[0] != drop_ts]
    idx, _ = make_index(rows, WD[:8])
    assert iv_rv.forward_sessions_target(idx, WD[0], 2, "hybrid").available                      # sessions 1,2: fine
    t = iv_rv.forward_sessions_target(idx, WD[0], 5, "hybrid")
    assert not t.available and t.reason == "incomplete_or_missing_session_in_window"
    assert not iv_rv.forward_sessions_target(idx, WD[0], 5, "intraday").available
    assert iv_rv.forward_sessions_target(idx, WD[4], 3, "hybrid").available                       # later windows unaffected


def test_incomplete_session_just_before_window_does_not_block_a_window_after_it():
    rows, _ = build(const_specs(8))
    drop_ts = session_rows(WD[2], [1.0] * 11)[10][0]
    rows = [r for r in rows if r[0] != drop_ts]
    idx, _ = make_index(rows, WD[:8])
    assert iv_rv.forward_sessions_target(idx, WD[2], 3, "intraday").available                     # sessions 3,4,5
    assert iv_rv.forward_sessions_target(idx, WD[1], 1, "intraday").reason == "incomplete_or_missing_session_in_window"


def test_not_enough_future_data_is_beyond_data_end():
    rows, _ = build(const_specs(4))
    idx, _ = make_index(rows, WD[:4])
    assert iv_rv.forward_sessions_target(idx, WD[0], 3, "hybrid").available
    t = iv_rv.forward_sessions_target(idx, WD[0], 4, "hybrid")
    assert not t.available and t.reason == "beyond_data_end"
    assert iv_rv.forward_sessions_target(idx, WD[3], 1, "hybrid").reason == "beyond_data_end"


def test_zero_future_rv_constant_prices_is_available_and_exactly_zero():
    rows, _ = build([(WD[i], [0.0] * 375, 0.0) for i in range(7)])
    idx, _ = make_index(rows, WD[:7])
    t = iv_rv.forward_sessions_target(idx, WD[0], 5, "hybrid")
    assert t.available and t.rv_pct == 0.0 and t.total_variance == 0.0
    e = iv_rv.expiry_aligned_target(idx, WD[0], "13:00", WD[4], "hybrid")
    assert e.available and e.rv_pct == 0.0


# ----------------------------------------------------------------------------- family B (expiry aligned)
def test_expiry_aligned_hand_value_for_13_00_snapshot():
    vols = [0.0005, 0.0006, 0.0007, 0.0008]
    rows, cl = build([(WD[i], alt(375, vols[i]), 0.004 * i) for i in range(4)])
    idx, _ = make_index(rows, WD[:4])
    t = iv_rv.expiry_aligned_target(idx, WD[0], "13:00", WD[3], "hybrid")
    partial = 149 * vols[0] ** 2
    later = sum(375 * v ** 2 for v in vols[1:4]) + sum((0.004 * i) ** 2 for i in (1, 2, 3))
    assert t.available and t.total_variance == pytest.approx(partial + later, rel=1e-9)
    assert t.session_equivalents == pytest.approx(149 / 375 + 3) and t.n_sessions == 3
    assert t.rv_pct == pytest.approx(100 * math.sqrt(A * (partial + later) / (149 / 375 + 3)), rel=1e-9)
    assert t.start_ts == "2026-09-14T13:01:00+05:30" and t.end_ts == "2026-09-17T15:30:00+05:30" and t.includes_partial_first_session
    ti = iv_rv.expiry_aligned_target(idx, WD[0], "13:00", WD[3], "intraday")
    assert not ti.includes_overnight and ti.total_variance == pytest.approx(partial + sum(375 * v ** 2 for v in vols[1:4]), rel=1e-9)


def test_three_overlapping_snapshots_have_different_partial_windows():
    rows, _ = build(const_specs(5))
    idx, _ = make_index(rows, WD[:5])
    ts = {h: iv_rv.expiry_aligned_target(idx, WD[0], h, WD[4], "intraday") for h in ("10:00", "13:00", "15:00")}
    assert [round(t.session_equivalents * 375) - 4 * 375 for t in ts.values()] == [329, 149, 29]
    assert ts["10:00"].total_variance > ts["13:00"].total_variance > ts["15:00"].total_variance


def test_expiry_boundary_inclusive_of_expiry_session_exclusive_of_later_sessions():
    vols = [0.0005, 0.0005, 0.0005, 0.0005, 0.02]
    rows, _ = build([(WD[i], alt(375, vols[i]), 0.0) for i in range(5)])
    idx, _ = make_index(rows, WD[:5])
    t = iv_rv.expiry_aligned_target(idx, WD[0], "10:00", WD[3], "intraday")
    assert t.available and t.n_sessions == 3 and t.total_variance < 0.01                  # the wild session 4 (after expiry) is excluded
    assert t.end_ts.startswith(WD[3].isoformat())


def test_expiry_day_snapshot_has_no_expiry_aligned_target():
    rows, _ = build(const_specs(3))
    idx, _ = make_index(rows, WD[:3])
    t = iv_rv.expiry_aligned_target(idx, WD[1], "10:00", WD[1], "hybrid")
    assert not t.available and t.reason == "expiry_day_partial_session_only"
    assert iv_rv.expiry_aligned_target(idx, WD[2], "10:00", WD[1], "hybrid").reason == "expiry_before_snapshot"


def test_expiry_beyond_the_spot_data_is_unavailable_not_extrapolated():
    rows, _ = build(const_specs(3))
    idx, _ = make_index(rows, WD[:3])
    t = iv_rv.expiry_aligned_target(idx, WD[0], "10:00", WD[10], "hybrid")
    assert not t.available and t.reason == "beyond_data_end"


def test_incomplete_or_missing_session_inside_the_window_blocks_expiry_target():
    rows, _ = build(const_specs(5))
    drop = session_rows(WD[2], [1.0] * 50)[49][0]
    rows = [r for r in rows if r[0] != drop]
    idx, _ = make_index(rows, WD[:5])
    assert iv_rv.expiry_aligned_target(idx, WD[0], "13:00", WD[4], "hybrid").reason == "incomplete_or_missing_session_in_window"
    assert iv_rv.expiry_aligned_target(idx, WD[3], "13:00", WD[4], "hybrid").available


def test_missing_bar_after_the_snapshot_in_the_snapshot_session_blocks_target_but_not_before():
    rows, _ = build(const_specs(3))
    after = session_rows(WD[0], [1.0] * 301)[300][0]                         # slot 300 (after 13:00)
    before = session_rows(WD[1], [1.0] * 101)[100][0]
    r_after = [r for r in rows if r[0] != after]
    idx, _ = make_index(r_after, WD[:3])
    assert iv_rv.expiry_aligned_target(idx, WD[0], "13:00", WD[2], "intraday").reason == "snapshot_session_bars_missing_after_snapshot"
    r_before = [r for r in build(const_specs(3))[0] if r[0] != session_rows(WD[0], [1.0] * 101)[100][0]]
    idx2, _ = make_index(r_before, WD[:3])
    assert iv_rv.expiry_aligned_target(idx2, WD[0], "13:00", WD[2], "intraday").available          # a gap BEFORE the snapshot is irrelevant


def test_special_session_inside_the_window_neither_counts_nor_breaks_it():
    fri, sat, mon = date(2026, 9, 18), date(2026, 9, 19), date(2026, 9, 21)
    base = [(fri, alt(375, 0.0005), 0.0), (mon, alt(375, 0.0005), 0.0)]
    rows, _ = build(base, extra_rows=session_rows(sat, [150.0 + (i % 3) for i in range(45)]))
    idx, _ = make_index(rows, [fri, mon])
    t = iv_rv.expiry_aligned_target(idx, fri, "10:00", mon, "intraday")
    assert t.available and t.n_sessions == 1 and t.total_variance == pytest.approx(329 * 25e-8 + 375 * 25e-8, rel=1e-9)
    assert idx.var["hybrid"].get(sat) is None


# ----------------------------------------------------------------------------- no look-ahead
def test_future_beyond_expiry_never_changes_an_expiry_aligned_target():
    base = [(WD[i], alt(375, 0.0005), 0.0) for i in range(4)]
    wild = base + [(WD[4], alt(375, 0.03), 0.2), (WD[5], alt(375, 0.03), -0.2)]
    r1, _ = build(base); r2, _ = build(wild)
    i1, _ = make_index(r1, WD[:4]); i2, _ = make_index(r2, WD[:6])
    assert iv_rv.expiry_aligned_target(i1, WD[0], "13:00", WD[3], "hybrid") == iv_rv.expiry_aligned_target(i2, WD[0], "13:00", WD[3], "hybrid")


def test_bars_before_the_snapshot_cannot_change_any_target():
    base = [(WD[i], alt(375, 0.0005), 0.0) for i in range(4)]
    rets_early = alt(375, 0.0005)
    for j in range(0, 45):                                    # scramble everything BEFORE the 10:00 bar (slots 0..44) ...
        rets_early[j] = 0.01 * ((-1) ** j)
    rets_early[44] += sum(alt(375, 0.0005)[:45]) - sum(rets_early[:45])     # ... but keep the cumulative move identical
    tampered = [(WD[0], rets_early, 0.0)] + base[1:]
    r1, _ = build(base); r2, _ = build(tampered)
    i1, _ = make_index(r1, WD[:4]); i2, _ = make_index(r2, WD[:4])
    a = iv_rv.expiry_aligned_target(i1, WD[0], "10:00", WD[3], "intraday")
    b = iv_rv.expiry_aligned_target(i2, WD[0], "10:00", WD[3], "intraday")
    assert a.total_variance == pytest.approx(b.total_variance, rel=1e-9) and a.n_returns == b.n_returns
    fa = iv_rv.forward_sessions_target(i1, WD[0], 2, "intraday")
    fb = iv_rv.forward_sessions_target(i2, WD[0], 2, "intraday")
    assert fa.rv_pct == pytest.approx(fb.rv_pct, rel=1e-9) and fa.n_returns == fb.n_returns


def test_recent_rv_uses_only_sessions_completed_before_the_snapshot_date():
    dates = [date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 16)]
    roll = {dates[0]: 10.0, dates[1]: 20.0, dates[2]: 99.0}
    assert iv_rv.recent_rv_before(roll, dates, dates[2]) == 20.0                # not the snapshot day's own value
    assert iv_rv.recent_rv_before(roll, dates, dates[0]) is None                # nothing before the first session
    assert iv_rv.recent_rv_before(roll, dates, date(2026, 9, 19)) == 99.0       # a non-chain date after the last session


def test_observation_timestamp_is_bar_start_plus_one_minute():
    assert iv_rv.observation_ts(date(2026, 9, 21), "15:00") == "2026-09-21T15:01:00+05:30"
