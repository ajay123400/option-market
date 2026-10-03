import hashlib
import os
from datetime import date

import pytest

from optionsengine.research import session_calendar as sc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_weekend_is_never_expected():
    cal = sc.SessionCalendar({date(2024, 1, 5)}, fallback=lambda d: True)
    assert cal.expected_trading_date(date(2024, 1, 6)) == (False, "weekend")      # Saturday
    assert cal.expected_trading_date(date(2024, 1, 7)) == (False, "weekend")      # Sunday


def test_inside_evidence_range_membership_decides():
    ev = {date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)}                   # Thu 4th missing => treated as non-trading
    cal = sc.SessionCalendar(ev, fallback=lambda d: True)
    assert cal.expected_trading_date(date(2024, 1, 3)) == (True, "participant_oi")
    assert cal.expected_trading_date(date(2024, 1, 4)) == (False, "participant_oi")


def test_outside_evidence_range_uses_repo_calendar_fallback():
    ev = {date(2024, 1, 2), date(2024, 1, 3)}
    seen = []
    cal = sc.SessionCalendar(ev, fallback=lambda d: seen.append(d) or d.day != 10)
    assert cal.expected_trading_date(date(2024, 1, 9)) == (True, "repo_market_calendar")
    assert cal.expected_trading_date(date(2024, 1, 10)) == (False, "repo_market_calendar")
    assert date(2023, 12, 29) not in seen
    assert cal.expected_trading_date(date(2023, 12, 29)) == (True, "repo_market_calendar")    # before coverage as well


def test_participant_oi_dates_parse_only_dated_csvs(tmp_path):
    for n in ("2021-09-01.csv", "2021-09-02.csv", "all.parquet", "notes.csv", "2021-13-40.csv"):
        (tmp_path / n).write_text("x")
    assert sc.participant_oi_dates(str(tmp_path)) == {date(2021, 9, 1), date(2021, 9, 2)}
    assert sc.participant_oi_dates(str(tmp_path / "missing")) == set()


def test_repo_calendar_use_is_read_only_and_matches_known_2026_facts():
    path = os.path.join(ROOT, "holidays.json")
    if not os.path.exists(path):
        pytest.skip("holidays.json not present")
    before = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert sc.repo_is_trading_day(date(2026, 9, 14)) is False      # in the repo holiday list
    assert sc.repo_is_trading_day(date(2026, 9, 15)) is True
    assert sc.repo_is_trading_day(date(2026, 9, 12)) is False      # Saturday
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == before


def test_real_participant_calendar_covers_spot_days_when_data_present():
    d = os.path.join(ROOT, "data", "participant_oi")
    if not os.path.isdir(d):
        pytest.skip("no participant_oi data")
    ev = sc.participant_oi_dates(d)
    assert date(2021, 9, 1) in ev and not any(x.weekday() >= 5 for x in ev)
