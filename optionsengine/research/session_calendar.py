"""Layered expected-session calendar for the NIFTY spot dataset (Stage 2B).

The repository's `market_calendar.py` carries NSE holidays for 2026 ONLY (holidays.json / its
fallback list); it has no 2021-2025 holidays, so on its own it would call every 2021-2025
weekday a trading day. Evidence used, in priority order:

1. Weekends are never regular sessions (Saturday/Sunday special sessions are classified later).
2. Inside the coverage of `data/participant_oi/` (one file per NSE trading date, 2021-09-01 ..
   2026-09-25, no weekend dates) a weekday is an expected trading date iff its file exists.
   This is a source independent of the spot file. Muhurat evenings are trading dates here
   although the regular session did not run - see `spot_quality` for how those are classified.
3. Outside that coverage the repository utility `market_calendar.is_trading_day` decides
   (weekday and not in its holiday set, covering 2026).

`expected_trading_date` therefore means "a trading-session of SOME kind was expected", not
"a 09:15-15:30 session was expected"; the session type comes from the bars themselves.
Known limitation: no historical holiday list exists in the repository, so a regular session
missing on a date that is also absent from participant_oi cannot be distinguished from a
holiday inside 2021-09-01..2026-09-25 (none observed: every participant date has spot bars).
"""
from __future__ import annotations

import os
import time as _time
from datetime import date
from typing import Callable, Iterable, Optional, Tuple


def participant_oi_dates(root: str = "data/participant_oi") -> set:
    """Dates of the participant-wise OI files (YYYY-MM-DD.csv); `all.parquet` etc. are ignored."""
    out = set()
    if os.path.isdir(root):
        for fn in os.listdir(root):
            if fn.endswith(".csv") and len(fn) == 14:
                try:
                    out.add(date.fromisoformat(fn[:-4]))
                except ValueError:
                    pass
    return out


def repo_is_trading_day(d: date) -> bool:
    """`market_calendar.is_trading_day`, with its NSE refresh/cache write suppressed (read-only use)."""
    import market_calendar as mc          # repo-root module (read-only here)
    mc._last_fetch_attempt = _time.time()  # makes its _load() skip the network fetch + holidays.json rewrite
    return mc.is_trading_day(d)


class SessionCalendar:
    def __init__(self, evidence_dates: Iterable[date], fallback: Optional[Callable[[date], bool]] = None):
        ev = set(evidence_dates)
        self.evidence = ev
        self.lo = min(ev) if ev else None
        self.hi = max(ev) if ev else None
        self.fallback = fallback or repo_is_trading_day

    def expected_trading_date(self, d: date) -> Tuple[bool, str]:
        if d.weekday() >= 5:
            return False, "weekend"
        if self.lo is not None and self.lo <= d <= self.hi:
            return (d in self.evidence), "participant_oi"
        return bool(self.fallback(d)), "repo_market_calendar"
