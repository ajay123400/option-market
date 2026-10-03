"""Realized-volatility maths on hand-calculable synthetic sessions."""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from optionsengine.research import realized_vol as rv, session_calendar as sc, spot_quality as sq
from tests.rv_synth import alt_returns, frame, path_from_returns, session_rows

A = 252.0
DAYS = [date(2026, 9, 14) + timedelta(days=i) for i in range(40) if (date(2026, 9, 14) + timedelta(days=i)).weekday() < 5]


def run(rows, days=None, cfg=rv.RVConfig(), expected=None):
    cal = sc.SessionCalendar(set(expected if expected is not None else (days or DAYS)), fallback=lambda d: False)
    sessions = sq.assess(frame(rows), cal)
    daily = rv.daily_table(sessions, cfg)
    return sessions, daily, rv.rolling_table(daily, cfg)


def const_day(day, price=100.0, open_=None):
    return session_rows(day, [price] * 375, opens=[open_ if open_ is not None else price] + [price] * 374)


def alt_day(day, start=100.0, r=0.001):
    """375 returns of +/- r starting from open=start; open_i = close_{i-1}."""
    closes = path_from_returns(start, alt_returns(375, r))
    return session_rows(day, closes, opens=[start] + closes[:-1]), closes[-1]


# ------------------------------------------------------------------ constants / known returns / units
def test_constant_prices_give_exactly_zero_everything():
    rows = sum((const_day(d) for d in DAYS[:3]), [])
    _, daily, _ = run(rows, DAYS[:3])
    x = daily[daily.session_type == "regular"]
    assert (x.intraday_sumsq == 0).all() and (x.intraday_vol_ann_pct == 0).all()
    assert x.overnight_return.dropna().eq(0).all() and x.cc_return.dropna().eq(0).all() and x.n_valid_returns.eq(375).all()


def test_known_alternating_returns_hand_calculation_and_units():
    rows, _ = alt_day(DAYS[0], 100.0, 0.001)
    _, daily, _ = run(rows, DAYS[:1])
    r = daily.iloc[0]
    assert r.n_valid_returns == 375 and r.complete
    assert r.intraday_sumsq == pytest.approx(375 * 1e-6, rel=1e-12)                   # 375 returns of +/-0.001
    assert r.intraday_sum_r == pytest.approx(0.001, rel=1e-9)                         # alternating: net +0.001 (odd count)
    # vol in PERCENTAGE POINTS: 100 * sqrt(252 * 375e-6) = 30.74
    assert r.intraday_vol_ann_pct == pytest.approx(100 * 0.001 * math.sqrt(375 * 252), rel=1e-12)
    assert r.intraday_vol_ann_pct == pytest.approx(30.7409, abs=1e-3)


def test_sum_of_minute_returns_equals_log_open_to_close_identity():
    rng = np.random.default_rng(3)
    rets = list(rng.normal(0, 4e-4, 375))
    closes = path_from_returns(100.0, rets)
    rows = session_rows(DAYS[0], closes, opens=[100.0] + closes[:-1])
    s, daily, _ = run(rows, DAYS[:1])
    r = daily.iloc[0]
    assert r.intraday_sum_r == pytest.approx(math.log(closes[-1] / 100.0), abs=1e-12)
    assert r.intraday_sumsq == pytest.approx(float(np.sum(np.square(rets))), rel=1e-12)


def test_annualization_is_sessions_based_not_minutes_based():
    rows, _ = alt_day(DAYS[0], 100.0, 0.001)
    for days_per_year in (248.0, 252.0, 365.0):
        _, daily, _ = run(rows, DAYS[:1], rv.RVConfig(annualization_days=days_per_year))
        assert daily.iloc[0].intraday_vol_ann_pct == pytest.approx(100 * math.sqrt(days_per_year * 375e-6), rel=1e-12)


# ------------------------------------------------------------------ overnight / close-to-close
def test_overnight_gap_is_a_separate_return_not_a_minute_return():
    d1, d2 = DAYS[0], DAYS[1]
    rows1 = const_day(d1, 100.0)
    rows2 = const_day(d2, 101.0)                                  # opens 1 % above yesterday's close, then flat
    _, daily, _ = run(rows1 + rows2, [d1, d2])
    r2 = daily.iloc[1]
    assert r2.overnight_return == pytest.approx(math.log(1.01), abs=1e-15) and r2.overnight_calendar_days == 1
    assert r2.cc_return == pytest.approx(math.log(1.01), abs=1e-15)
    assert r2.intraday_sumsq == 0 and r2.n_valid_returns == 375          # the gap does NOT leak into the minute returns
    assert r2.cc_variance == pytest.approx(math.log(1.01) ** 2) and r2.hybrid_variance == pytest.approx(math.log(1.01) ** 2)
    assert math.isnan(daily.iloc[0].cc_return) and daily.iloc[0].cc_reason == "no_previous_session"


def test_close_to_close_equals_overnight_plus_intraday_open_to_close():
    d1, d2 = DAYS[0], DAYS[1]
    r1, c1 = alt_day(d1, 100.0, 0.0007)
    r2, c2 = alt_day(d2, c1 * 1.004, 0.0009)
    _, daily, _ = run(r1 + r2, [d1, d2])
    x = daily.iloc[1]
    assert x.cc_return == pytest.approx(x.overnight_return + x.intraday_sum_r, abs=1e-12)
    assert x.hybrid_variance == pytest.approx(x.overnight_return ** 2 + x.intraday_sumsq, rel=1e-12)


def test_weekend_overnight_spans_three_calendar_days_and_is_flagged():
    fri, mon = date(2026, 9, 18), date(2026, 9, 21)
    _, daily, _ = run(const_day(fri) + const_day(mon, 100.5), [fri, mon])
    assert daily.iloc[1].overnight_calendar_days == 3 and daily.iloc[1].prev_session_date == "2026-09-18"


def test_special_session_between_regular_sessions_does_not_count_or_break_the_chain():
    fri, sat, mon = date(2026, 9, 18), date(2026, 9, 19), date(2026, 9, 21)
    special = session_rows(sat, [107.0] * 45)                      # a Saturday special session at a different price
    _, daily, _ = run(const_day(fri, 100.0) + special + const_day(mon, 100.5), [fri, mon])
    m = daily[daily.session_date == "2026-09-21"].iloc[0]
    assert m.prev_session_date == "2026-09-18" and m.overnight_return == pytest.approx(math.log(1.005), abs=1e-15)
    s = daily[daily.session_date == "2026-09-19"].iloc[0]
    assert s.session_type == "special_weekend" and not s.in_chain and math.isnan(s.get("intraday_variance", float("nan")))


def test_missing_regular_session_breaks_close_to_close_and_overnight():
    d1, d2, d3 = DAYS[0], DAYS[1], DAYS[2]
    _, daily, _ = run(const_day(d1, 100.0) + const_day(d3, 103.0), [d1, d2, d3])
    x = daily[daily.session_date == d3.isoformat()].iloc[0]
    assert daily[daily.session_date == d2.isoformat()].iloc[0].session_type == "missing_session"
    assert math.isnan(x.cc_return) and x.cc_reason == "previous_regular_session_missing"
    assert math.isnan(x.overnight_return) and x.n_valid_returns == 375               # its own intraday statistics remain valid


# ------------------------------------------------------------------ missing minutes / incomplete sessions
def test_missing_minute_drops_exactly_two_returns_and_never_creates_a_gap_return():
    closes = path_from_returns(100.0, alt_returns(375, 0.001))
    k = 100
    holed = list(closes); holed[k] = None
    opens = [100.0] + closes[:-1]
    rows = session_rows(DAYS[0], holed, opens=opens)
    _, daily, _ = run(rows, DAYS[:1])
    r = daily.iloc[0]
    assert r.n_valid_bars == 374 and r.n_valid_returns == 373 and not r.complete
    assert r.intraday_sumsq == pytest.approx(373 * 1e-6, rel=1e-12)                  # returns k and k+1 absent; no 2-minute return
    assert math.isnan(r.intraday_variance) and math.isnan(r.intraday_vol_ann_pct)    # strict: not eligible
    assert r.intraday_variance_scaled == pytest.approx(373e-6 * 375 / 373, rel=1e-12)  # labelled coverage-qualified variant


def test_missing_first_bar_invalidates_overnight_but_not_close_to_close():
    d1, d2 = DAYS[0], DAYS[1]
    closes = [100.0] * 375
    holed = list(closes); holed[0] = None
    _, daily, _ = run(const_day(d1) + session_rows(d2, holed), [d1, d2])
    x = daily.iloc[1]
    assert math.isnan(x.overnight_return) and x.overnight_reason == "open_bar_missing" and x.cc_return == 0.0
    assert x.n_valid_returns == 373                                                  # r_0 and r_1 both unavailable


def test_missing_last_bar_invalidates_that_close_and_the_next_overnight():
    d1, d2, d3 = DAYS[0], DAYS[1], DAYS[2]
    holed = [100.0] * 374 + [None]
    _, daily, _ = run(const_day(d1) + session_rows(d2, holed) + const_day(d3), [d1, d2, d3])
    assert daily.iloc[1].cc_reason == "close_bar_missing"
    assert daily.iloc[2].cc_reason == "previous_close_bar_missing" and daily.iloc[2].overnight_reason == "previous_close_bar_missing"


def test_incomplete_session_is_never_eligible_for_strict_measures():
    closes = [100.0] * 375
    closes[10] = None
    _, daily, _ = run(session_rows(DAYS[0], closes), DAYS[:1])
    assert not daily.iloc[0].complete and math.isnan(daily.iloc[0].intraday_variance) and math.isnan(daily.iloc[0].hybrid_variance)


def test_incomplete_session_with_valid_overnight_still_gets_no_strict_hybrid():
    d1, d2 = DAYS[0], DAYS[1]
    closes = [100.0] * 375
    closes[200] = None
    _, daily, _ = run(const_day(d1) + session_rows(d2, closes, opens=[100.0] + [100.0] * 374), [d1, d2])
    x = daily.iloc[1]
    assert x.overnight_return == 0.0 and not x.complete
    assert math.isnan(x.hybrid_variance) and math.isnan(x.intraday_variance) and not math.isnan(x.hybrid_variance_scaled)


def test_short_special_session_gets_no_statistics_and_is_not_in_the_chain():
    d1, d2, d3 = DAYS[0], DAYS[1], DAYS[2]
    muhurat = session_rows(d2, [105.0 + 0.1 * (i % 2) for i in range(61)], start_minute=13 * 60 + 45)
    _, daily, _ = run(const_day(d1, 100.0) + muhurat + const_day(d3, 100.0), [d1, d3])
    m = daily[daily.session_date == d2.isoformat()].iloc[0]
    assert m.session_type == "special_short" and not m.in_chain
    assert pd.isna(m.get("intraday_sumsq")) and pd.isna(m.get("cc_return"))
    n = daily[daily.session_date == d3.isoformat()].iloc[0]
    assert n.prev_session_date == d1.isoformat() and n.overnight_return == 0.0       # the 105 Muhurat prices never touch the chain


def test_coverage_threshold_controls_scaled_variance():
    closes = path_from_returns(100.0, alt_returns(375, 0.001))
    for k in range(40, 80):                                        # 40 consecutive missing bars -> returns 40..80 lost (41) -> coverage 334/375
        closes[k] = None
    rows = session_rows(DAYS[0], [c for c in closes], opens=[100.0] + path_from_returns(100.0, alt_returns(375, 0.001))[:-1])
    _, d_lo, _ = run(rows, DAYS[:1], rv.RVConfig(cq_min_session_coverage=0.75))
    _, d_hi, _ = run(rows, DAYS[:1], rv.RVConfig(cq_min_session_coverage=0.90))
    assert d_lo.iloc[0].coverage == pytest.approx(334 / 375) and not math.isnan(d_lo.iloc[0].intraday_variance_scaled)
    assert math.isnan(d_hi.iloc[0].intraday_variance_scaled)


# ------------------------------------------------------------------ rolling windows
def flat_days_with_vol(n, vols):
    """n sessions; session k has 375 returns of +/- vols[k] => intraday variance 375*vols[k]^2; opens continue prices."""
    rows, price = [], 100.0
    for k in range(n):
        r, price = alt_day(DAYS[k], price, vols[k])
        rows += r
    return rows


def test_rolling_insufficient_history_and_exact_hand_values():
    vols = [0.0005, 0.0007, 0.0006, 0.0008, 0.0004, 0.0009, 0.0005]
    _, daily, roll = run(flat_days_with_vol(7, vols), DAYS[:7])
    r = roll[(roll.measure == "intraday") & (roll.window_sessions == 5) & (roll.method == "strict")].set_index("session_date")
    assert r.loc[DAYS[3].isoformat(), "note"] == "insufficient_history" and not r.loc[DAYS[3].isoformat(), "complete"]
    assert r.loc[DAYS[3].isoformat(), "n_obs"] == 0 and math.isnan(r.loc[DAYS[3].isoformat(), "rv_ann_pct"])
    row = r.loc[DAYS[4].isoformat()]                                   # sessions 0..4
    expect = 100 * math.sqrt(A * np.mean([375 * v ** 2 for v in vols[:5]]))
    assert row.complete and row.n_obs == 5 and row.rv_ann_pct == pytest.approx(expect, rel=1e-12)
    row2 = r.loc[DAYS[6].isoformat()]                                  # sessions 2..6
    assert row2.rv_ann_pct == pytest.approx(100 * math.sqrt(A * np.mean([375 * v ** 2 for v in vols[2:7]])), rel=1e-12)
    assert row2.first_session == DAYS[2].isoformat() and row2.asof_ts_ist == f"{DAYS[6].isoformat()}T15:30:00+05:30"


def test_close_to_close_window_needs_n_plus_one_closes():
    rows = flat_days_with_vol(7, [0.0005] * 7)
    _, daily, roll = run(rows, DAYS[:7])
    r = roll[(roll.measure == "close_to_close") & (roll.window_sessions == 5) & (roll.method == "strict")].set_index("session_date")
    assert r.loc[DAYS[4].isoformat(), "n_obs"] == 4 and not r.loc[DAYS[4].isoformat(), "complete"]       # session 0 has no previous close
    assert math.isnan(r.loc[DAYS[4].isoformat(), "rv_ann_pct"])
    assert r.loc[DAYS[5].isoformat(), "complete"] and r.loc[DAYS[5].isoformat(), "n_obs"] == 5
    ccs = daily.cc_return.to_numpy()[1:6]
    assert r.loc[DAYS[5].isoformat(), "rv_ann_pct"] == pytest.approx(100 * math.sqrt(A * np.mean(ccs ** 2)), rel=1e-12)


def test_strict_window_is_never_silently_shortened_but_coverage_qualified_is_labelled():
    rows, price = [], 100.0
    for k in range(7):
        if k == 3:                                                  # session 3 loses 5 bars (coverage 365/375 returns approx)
            closes = path_from_returns(price, alt_returns(375, 0.0006))
            holed = list(closes)
            for j in range(100, 105):
                holed[j] = None
            rows += session_rows(DAYS[k], holed, opens=[price] + closes[:-1]); price = closes[-1]
        else:
            r, price = alt_day(DAYS[k], price, 0.0006); rows += r
    _, daily, roll = run(rows, DAYS[:7])
    s = roll[(roll.measure == "intraday") & (roll.window_sessions == 5) & (roll.session_date == DAYS[6].isoformat())].set_index("method")
    assert math.isnan(s.loc["strict", "rv_ann_pct"]) and s.loc["strict", "n_obs"] == 4 and not s.loc["strict", "complete"]
    cq = s.loc["coverage_qualified"]
    assert cq.n_obs == 5 and cq.complete and not math.isnan(cq.rv_ann_pct)       # session 3 qualifies at 90 % coverage (scaled)
    # a session below the coverage threshold drops out of the cq window with an explicit note
    strict_cfg = rv.RVConfig(cq_min_session_coverage=0.999)
    _, _, roll2 = run(rows, DAYS[:7], strict_cfg)
    c2 = roll2[(roll2.measure == "intraday") & (roll2.window_sessions == 5) & (roll2.session_date == DAYS[6].isoformat()) & (roll2.method == "coverage_qualified")].iloc[0]
    assert c2.n_obs == 4 and not c2.complete and "4/5" in c2.note and not math.isnan(c2.rv_ann_pct)   # >= 80 % of 5 sessions valid


def test_window_below_cq_fraction_has_no_value():
    rows = []
    price = 100.0
    for k in range(6):
        if k in (2, 3):
            closes = [price] * 375; closes[50] = None
            rows += session_rows(DAYS[k], closes)
        else:
            rows += const_day(DAYS[k], price)
    _, _, roll = run(rows, DAYS[:6], rv.RVConfig(cq_min_session_coverage=0.9999))
    x = roll[(roll.measure == "intraday") & (roll.window_sessions == 5) & (roll.session_date == DAYS[5].isoformat()) & (roll.method == "coverage_qualified")].iloc[0]
    assert x.n_obs == 3 and math.isnan(x.rv_ann_pct) and "insufficient_valid_sessions" in x.note


def test_coverage_qualified_requires_intact_open_and_close_edges():
    closes = path_from_returns(100.0, alt_returns(375, 0.001))
    opens = [100.0] + closes[:-1]
    for hole in (3, 370, 0):                                      # inside the first 15 / last 15 bars / the very first bar
        c = list(closes); c[hole] = None
        _, d, _ = run(session_rows(DAYS[0], c, opens=opens), DAYS[:1])
        assert d.iloc[0].coverage > 0.98 and math.isnan(d.iloc[0].intraday_variance_scaled), hole
        _, d0, _ = run(session_rows(DAYS[0], c, opens=opens), DAYS[:1], rv.RVConfig(cq_edge_minutes=0))
        assert not math.isnan(d0.iloc[0].intraday_variance_scaled)   # the guard is configurable and explicit
    c = list(closes); c[200] = None                                # a mid-session hole is accepted at 99 % coverage
    _, d, _ = run(session_rows(DAYS[0], c, opens=opens), DAYS[:1])
    assert not math.isnan(d.iloc[0].intraday_variance_scaled)


def test_special_sessions_do_not_count_toward_window_length():
    vols = [0.0005] * 6
    rows = flat_days_with_vol(6, vols)
    sat = date(2026, 9, 19)
    rows = sorted(rows + session_rows(sat, [100.0] * 45))
    exp = DAYS[:6]
    _, daily, roll = run(rows, exp)
    assert (daily[daily.session_type == "special_weekend"].in_chain == False).all()
    r = roll[(roll.measure == "intraday") & (roll.window_sessions == 5) & (roll.method == "strict") & roll.complete]
    assert len(r) == 2 and set(r.session_date) == {DAYS[4].isoformat(), DAYS[5].isoformat()}         # windows are 5 REGULAR sessions


def test_no_look_ahead_future_data_cannot_change_earlier_rows():
    vols = [0.0005, 0.0007, 0.0006, 0.0008, 0.0004, 0.0009]
    _, _, base = run(flat_days_with_vol(6, vols), DAYS[:6])
    # (a) sessions that do not exist yet: truncated history gives the same rows
    # (b) a very different future (vol 0.01 for the next two sessions) leaves every earlier row bit-identical
    _, _, future = run(flat_days_with_vol(8, vols + [0.01, 0.01]), DAYS[:8])
    earlier = future[future.session_date <= DAYS[5].isoformat()].reset_index(drop=True)
    pd.testing.assert_frame_equal(base.reset_index(drop=True), earlier)
    later = future[future.session_date == DAYS[7].isoformat()]
    assert later[(later.measure == "intraday") & (later.window_sessions == 5) & (later.method == "strict")].rv_ann_pct.iloc[0] > 100


def test_session_must_be_complete_at_observation_time_asof_is_the_session_close():
    _, _, roll = run(flat_days_with_vol(6, [0.0005] * 6), DAYS[:6])
    assert roll.asof_ts_ist.str.endswith("T15:30:00+05:30").all()
    assert (roll.asof_ts_ist.str[:10] == roll.session_date).all()


def test_sampled_returns_helper_non_overlapping():
    closes = np.array([100.0 * math.exp(0.001 * i) for i in range(375)])
    r5 = rv.sampled_returns(closes, 5)
    assert len(r5) == 74 and np.allclose(r5, 0.005)
    c = closes.copy(); c[9] = np.nan
    assert len(rv.sampled_returns(c, 5)) == 72                    # both neighbouring 5-minute returns lose the missing close
