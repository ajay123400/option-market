"""Regression tests for the expiry close-time fix: 15:30 IST before 2026-08-03, 15:40 IST on/after."""
from datetime import date, datetime, time, timedelta, timezone

import pytest

from optionsengine import (CloseTimeRule, ExpiryCloseSchedule, InvalidInputError, NSE_FNO_CLOSE_SCHEDULE,
                           expiry_at_close, time_to_expiry_years)

IST = timezone(timedelta(hours=5, minutes=30))


@pytest.mark.parametrize("d,hm", [
    (date(2021, 9, 30), (15, 30)), (date(2025, 9, 23), (15, 30)),
    (date(2026, 7, 28), (15, 30)), (date(2026, 7, 31), (15, 30)),       # last Friday before the change
    (date(2026, 8, 3), (15, 40)),                                       # first date on the new close (inclusive)
    (date(2026, 8, 4), (15, 40)), (date(2026, 9, 29), (15, 40)), (date(2027, 1, 1), (15, 40)),
])
def test_close_time_before_and_after_change(d, hm):
    e = expiry_at_close(d)
    assert (e.hour, e.minute) == hm and e.date() == d and e.utcoffset() == timedelta(hours=5, minutes=30)


def test_rule_is_keyed_on_expiry_date_not_observation_date():
    # snapshot on the old-close side of the boundary, contract expiring on the new-close side
    obs = datetime(2026, 7, 31, 15, 30, tzinfo=IST)
    T = time_to_expiry_years(obs, expiry_at_close(date(2026, 8, 4)))
    assert T == pytest.approx((4 * 86400 + 10 * 60) / (365 * 86400), rel=1e-12)   # 4 days + 10 min (15:40 expiry)


def test_year_fraction_differs_by_exactly_ten_minutes():
    obs = datetime(2026, 9, 29, 15, 15, tzinfo=IST)
    new = time_to_expiry_years(obs, expiry_at_close(date(2026, 9, 29)))
    old = time_to_expiry_years(obs, expiry_at_close(date(2026, 9, 29), close=time(15, 30)))
    assert new == pytest.approx(25 * 60 / (365 * 86400), rel=1e-12)
    assert old == pytest.approx(15 * 60 / (365 * 86400), rel=1e-12)


def test_explicit_close_overrides_schedule():
    assert expiry_at_close(date(2026, 10, 6), close=time(15, 30)).time() == time(15, 30)


def test_custom_schedule_is_supported_without_code_changes():
    sched = ExpiryCloseSchedule((CloseTimeRule(date.min, time(15, 30), "a"),
                                 CloseTimeRule(date(2026, 8, 3), time(15, 40), "b"),
                                 CloseTimeRule(date(2027, 1, 1), time(15, 45), "hypothetical future change")))
    assert expiry_at_close(date(2026, 12, 31), schedule=sched).time() == time(15, 40)
    assert expiry_at_close(date(2027, 1, 1), schedule=sched).time() == time(15, 45)
    assert sched.rule_for(date(2027, 6, 1)).source == "hypothetical future change"


def test_default_schedule_carries_sources():
    assert all(r.source for r in NSE_FNO_CLOSE_SCHEDULE.rules)
    assert NSE_FNO_CLOSE_SCHEDULE.rule_for(date(2026, 8, 3)).effective_from == date(2026, 8, 3)


@pytest.mark.parametrize("rules", [
    (),
    (CloseTimeRule(date(2026, 1, 1), time(15, 30), "no date.min start"),),
    (CloseTimeRule(date.min, time(15, 30), "a"), CloseTimeRule(date.min, time(15, 40), "dup")),
    (CloseTimeRule(date.min, time(15, 30), "a"), CloseTimeRule(date(2026, 8, 3), time(15, 40), "b"),
     CloseTimeRule(date(2026, 1, 1), time(15, 45), "out of order")),
])
def test_invalid_schedules_rejected(rules):
    with pytest.raises(InvalidInputError):
        ExpiryCloseSchedule(rules)


def test_datetime_is_not_accepted_as_expiry_date():
    with pytest.raises(InvalidInputError):
        NSE_FNO_CLOSE_SCHEDULE.close_time(datetime(2026, 9, 29, 12, 0))


def test_old_default_would_have_misstated_expiry_day_iv():
    """Documents the size of the bug: ATM, expiry day, 15:15, true vol 14 %. Pricing with the correct
    15:40 expiry and inverting with the old 15:30 default overstates IV by x1.29 (the historical audit measured a median
    +22 vol pts at 15:15 on real expiry days)."""
    from optionsengine import bsm_price, implied_carry_yield, implied_volatility
    S, K, r = 24500.0, 24500.0, 0.065
    obs = datetime(2026, 9, 29, 15, 15, tzinfo=IST)
    T_true = time_to_expiry_years(obs, expiry_at_close(date(2026, 9, 29)))
    T_old = time_to_expiry_years(obs, expiry_at_close(date(2026, 9, 29), close=time(15, 30)))
    price = bsm_price(S, K, T_true, r, 0.0, 0.14, "call")
    iv_old = implied_volatility(price, S, K, T_old, r, 0.0, "call").iv
    assert iv_old == pytest.approx(0.14 * (T_true / T_old) ** 0.5, rel=2e-3)   # ATM IV scales with 1/sqrt(T): x1.29
    assert iv_old - 0.14 > 0.035                                               # ~ +4 vol pts at a 14 % level
