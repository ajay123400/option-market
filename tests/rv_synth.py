"""Synthetic 1-minute spot data builders for the Stage 2B tests (hand-calculable)."""
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))


def ts_of(day: date, minute_of_day: int) -> int:
    return int(datetime(day.year, day.month, day.day, minute_of_day // 60, minute_of_day % 60, tzinfo=IST).timestamp())


def session_rows(day: date, closes, opens=None, start_minute=9 * 60 + 15, spread=0.0):
    """Rows (ts, open, high, low, close) for consecutive minutes starting at start_minute.
    opens default to the previous close (first open = first close unless given)."""
    rows = []
    prev = None
    for i, c in enumerate(closes):
        if c is None:
            prev = None if opens is None else prev
            continue
        o = opens[i] if opens is not None else (prev if prev is not None else c)
        hi, lo = max(o, c) * (1 + spread), min(o, c) * (1 - spread)
        rows.append((ts_of(day, start_minute + i), o, hi, lo, c))
        prev = c
    return rows


def frame(rows):
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])


def path_from_returns(start_price, returns):
    """closes after applying log returns; first element is the close of bar 0 given open=start_price."""
    p, out = start_price, []
    for r in returns:
        p *= math.exp(r)
        out.append(p)
    return out


def alt_returns(n=375, r=0.001):
    return [r if i % 2 == 0 else -r for i in range(n)]
