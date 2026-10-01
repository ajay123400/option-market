"""Expiry close-time schedule for NSE index options.

Why this exists: time to expiry T is measured to the instant the contract
expires, and near expiry the result is very sensitive to that instant (a
10-minute error moves expiry-day ATM IV by ~22 vol points at 15:15). The
close time is NOT part of any dataset in this repo (no file carries an
expiry time-of-day), so it must come from a documented source:

* NSE F&O close moved from 15:30 to 15:40 IST on 2026-08-03.
  Sources in this repo: `market_calendar.py:27` ("NSE F&O close since
  2026-08-03 (was 15:30)") and `simulator.py:32` (`F_O_CLOSE_CHANGE`).
* Independent confirmation from the recorded data: in `data/hist1m/options`
  the last 1-minute bar of every trading day starts at 15:29 up to
  2026-07-31 and at 15:39 from 2026-08-03 (verified over all 263 expiry
  files, see the audit report), i.e. sessions end 15:30 / 15:40.

This is deliberately a small, explicit, overridable SCHEDULE rather than a
universal constant:

* Prefer provider/contract metadata. If a feed supplies the real expiry
  date-time, put it straight into `OptionContract.expiry` and never call
  this module.
* Use `NSE_FNO_CLOSE_SCHEDULE` only for date-only expiries, and mark the
  contract `expiry_time_assumed=True` so the quality layer flags it.
* If the exchange changes the close again, append a `CloseTimeRule`
  (or build your own `ExpiryCloseSchedule`); nothing else needs editing.

The rule is keyed on the EXPIRY date (the contract's own settlement day),
not on the observation date: a snapshot taken on 2026-07-31 of a contract
expiring 2026-08-04 uses the 15:40 close.
Limitation: the 15:30 -> 15:40 rule is the repo's convention for expiry-day
settlement; no exchange circular was available to this project to confirm
that settlement itself (as opposed to continuous trading) moved too.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from .errors import InvalidInputError

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True)
class CloseTimeRule:
    effective_from: date   # first EXPIRY date the rule applies to (inclusive)
    close: time            # IST wall-clock expiry/close time
    source: str            # where this rule comes from


@dataclass(frozen=True)
class ExpiryCloseSchedule:
    """Ordered rules; the rule with the latest `effective_from` <= expiry
    date wins. The first rule must start at `date.min` so every date maps."""
    rules: tuple

    def __post_init__(self):
        rules = tuple(self.rules)
        if not rules:
            raise InvalidInputError("schedule needs at least one rule")
        if rules[0].effective_from != date.min:
            raise InvalidInputError("first rule must have effective_from == date.min")
        dates = [r.effective_from for r in rules]
        if dates != sorted(set(dates)):
            raise InvalidInputError("rules must be strictly increasing by effective_from")
        object.__setattr__(self, "rules", rules)

    def rule_for(self, expiry_date: date) -> CloseTimeRule:
        if not isinstance(expiry_date, date) or isinstance(expiry_date, datetime):
            raise InvalidInputError(f"expiry_date must be a date (got {expiry_date!r})")
        chosen = self.rules[0]
        for r in self.rules:
            if r.effective_from <= expiry_date:
                chosen = r
        return chosen

    def close_time(self, expiry_date: date) -> time:
        return self.rule_for(expiry_date).close

    def close_datetime(self, expiry_date: date, tz: timezone = IST) -> datetime:
        return datetime.combine(expiry_date, self.close_time(expiry_date), tzinfo=tz)


NSE_FNO_CLOSE_SCHEDULE = ExpiryCloseSchedule((
    CloseTimeRule(date.min, time(15, 30), "NSE F&O regular close before 2026-08-03"),
    CloseTimeRule(date(2026, 8, 3), time(15, 40),
                  "NSE F&O close moved to 15:40 on 2026-08-03: repo market_calendar.py:27 / simulator.py:32; "
                  "confirmed by recorded 1-min bars (last bar 15:39 from that date)"),
))
