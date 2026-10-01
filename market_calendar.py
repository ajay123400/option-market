"""market_calendar.py -- the ONE place that knows NSE F&O market hours,
trading holidays and "what time is it in India".

Before this, every module carried its own copy: paper_trade.py,
manual_monitor.py and historical_recorder.py each had their own
MARKET_OPEN/MARKET_CLOSE, the Data Health page used 15:30 while everything
else used 15:40, none of them knew about NSE holidays (so a Monday holiday
was polled and recorded like a trading day), and all of them used the
machine's local clock instead of IST.

Holidays come from NSE's own holiday-master API (F&O segment), cached in
holidays.json next to the app and refreshed weekly; the hardcoded 2026 list
below is only a fallback for when NSE can't be reached.
"""
import json
import os
import threading
import time
from datetime import date, datetime, time as dtime, timedelta, timezone

import requests

import paths

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 40)     # NSE F&O close since 2026-08-03 (was 15:30)
SQUARE_OFF_TIME = dtime(15, 25)  # intraday auto square-off, 15 min before close

HOLIDAYS_FILE = os.path.join(paths.BASE_DIR, "holidays.json")
NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
_REFRESH_SEC = 7 * 86400
_RETRY_SEC = 3600

# NSE F&O trading holidays 2026 (from NSE's holiday master, fetched
# 2026-09-23) -- fallback only.
_FALLBACK_HOLIDAYS = {
    "2026-01-15", "2026-01-26", "2026-02-15", "2026-03-03", "2026-03-21", "2026-03-26",
    "2026-03-31", "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
    "2026-08-15", "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-08", "2026-11-10",
    "2026-11-24", "2026-12-25",
}

_lock = threading.Lock()
_holidays = None
_loaded_at = 0.0
_last_fetch_attempt = 0.0


def now_ist():
    return datetime.now(IST)


def _fetch_from_nse():
    resp = requests.get(NSE_HOLIDAY_URL, timeout=10, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"})
    resp.raise_for_status()
    rows = resp.json().get("FO", [])
    out = {datetime.strptime(r["tradingDate"], "%d-%b-%Y").date().isoformat() for r in rows}
    if not out:
        raise ValueError("NSE holiday list came back empty")
    return out


def _load():
    """Holiday set, refreshing the on-disk cache when it's older than a
    week. Never raises -- worst case is the built-in fallback list."""
    global _holidays, _loaded_at, _last_fetch_attempt
    with _lock:
        if _holidays is not None and time.time() - _loaded_at < 3600:
            return _holidays
        cached, fetched_at = None, 0
        try:
            with open(HOLIDAYS_FILE) as f:
                data = json.load(f)
            cached, fetched_at = set(data["holidays"]), data.get("fetched_at", 0)
        except Exception:
            pass
        stale = cached is None or time.time() - fetched_at > _REFRESH_SEC
        if stale and time.time() - _last_fetch_attempt > _RETRY_SEC:
            _last_fetch_attempt = time.time()
            try:
                fresh = _fetch_from_nse()
                # keep past years already cached -- NSE only lists the current year
                cached = (cached or set()) | fresh
                paths.atomic_write_json(HOLIDAYS_FILE, {"fetched_at": time.time(), "holidays": sorted(cached)})
            except Exception:
                pass
        _holidays = (cached or set()) | _FALLBACK_HOLIDAYS
        _loaded_at = time.time()
        return _holidays


def is_holiday(d):
    return d.isoformat() in _load()


def is_trading_day(d):
    return d.weekday() < 5 and not is_holiday(d)


def is_market_open(now=None):
    now = now or now_ist()
    return is_trading_day(now.date()) and MARKET_OPEN <= now.time() <= MARKET_CLOSE


def is_square_off_window(now=None):
    now = now or now_ist()
    return is_trading_day(now.date()) and SQUARE_OFF_TIME <= now.time() <= MARKET_CLOSE


def market_state(now=None):
    """One of OPEN / PRE_OPEN / CLOSED / WEEKEND / HOLIDAY, plus a label."""
    now = now or now_ist()
    d = now.date()
    if d.weekday() >= 5:
        return {"state": "WEEKEND", "label": "Market closed (weekend)"}
    if is_holiday(d):
        return {"state": "HOLIDAY", "label": "Market closed (NSE holiday)"}
    if now.time() < MARKET_OPEN:
        return {"state": "PRE_OPEN", "label": "Market opens 09:15"}
    if now.time() > MARKET_CLOSE:
        return {"state": "CLOSED", "label": "Market closed for the day"}
    return {"state": "OPEN", "label": "Market open"}


def previous_trading_day(d):
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def wait_until_open(poll_sec=10, stop_event=None):
    """Blocks until the market is open today. Returns False (without
    waiting) if today isn't a trading day or the close has already passed."""
    while True:
        now = now_ist()
        if not is_trading_day(now.date()) or now.time() > MARKET_CLOSE:
            return False
        if now.time() >= MARKET_OPEN:
            return True
        if stop_event is not None:
            if stop_event.wait(poll_sec):
                return False
        else:
            time.sleep(poll_sec)


if __name__ == "__main__":
    print("now IST:", now_ist().isoformat(timespec="seconds"), market_state())
    hs = sorted(_load())
    print(f"{len(hs)} holidays known; next ones:", [h for h in hs if h >= now_ist().date().isoformat()][:5])
