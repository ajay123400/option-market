"""historical_recorder.py -- background live-capture that turns TODAY's
Fyers option-chain polling into TOMORROW's Simulator data, with no manual
pull needed. Runs as a daemon thread inside app.py for as long as the
app/exe stays open during market hours: polls the same options-chain-v3
endpoint the live pages already use (via fyers_option_chain.get_chain(),
so it shares that module's 2s cache -- effectively free on top of the
existing live traffic), buckets into 5-min OHLCV+OI candles the same way
paper_trade.py does, and appends each finished candle straight into
data/<symbol>.csv -- the exact file/column format simulator.py already
reads, so a session recorded today shows up in the Simulator tomorrow
with zero conversion step.

Fyers' own symbol string ('NSE:NIFTY2690823500CE') has its exchange
prefix stripped before use, which makes it identical to the trading-
symbol convention the original Kite-fetched CSVs already use -- so newly
recorded candles append onto the SAME file as that older history rather
than starting a parallel one (confirmed live: Fyers and Kite use the same
NSE-listed contract symbol once the "NSE:" prefix is dropped).

Always tracks whichever two expiries Fyers' own list_expiries() currently
reports as nearest -- so the tracked pair rolls forward past every weekly
expiry on its own, no manual expiries.json edits. A strike seen for the
first time is appended to that expiry's instruments file automatically.

KNOWN LIMITATION: this only records while the app is actually open --
unlike a real historical-data API pull, it can't backfill a session the
app wasn't running for. Missing candles for a gap like that are a real
gap, same as any live recorder / paper-trader would have.
"""
import json
import os
import threading
from datetime import datetime, time as dtime

import requests

import fyers_auth
import fyers_option_chain as chain_mod
import market_calendar as mc
import paths

HISTORY_URL = "https://api-t1.fyers.in/data/history"

BASE = paths.BASE_DIR
POLL_INTERVAL_SEC = 15
IDLE_POLL_SEC = 30  # how often to check "has the market opened yet" while outside hours
CANDLE_MINUTES = 5
STRIKECOUNT = 20  # matches the ~80-strike spread the original Kite-fetched CSVs cover
MARKET_OPEN = mc.MARKET_OPEN
MARKET_CLOSE = mc.MARKET_CLOSE  # 15:40 -- single source of truth is market_calendar.py
HOLIDAY_CHECK_DELAY_SEC = 300  # how long to watch for real volume before assuming a holiday
HOLIDAY_VOLUME_THRESHOLD = 5000  # a genuine trading day clears this within seconds of open, not minutes
TZ_SUFFIX = "+05:30"  # IST is a fixed offset (no DST) -- matches existing CSV rows exactly

EXPIRIES_FILE = os.path.join(BASE, "expiries.json")
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_thread = None
_stop_event = None


def _bucket_start(ts: datetime) -> datetime:
    """Floors ts to the current 5-min NSE candle boundary, anchored at
    09:15 -- identical alignment to paper_trade.py and simulator.py's own
    resample origin, so live-recorded and historically-resampled candles
    never land on different grids."""
    base_min = 9 * 60 + 15
    now_min = ts.hour * 60 + ts.minute
    offset = ((now_min - base_min) // CANDLE_MINUTES) * CANDLE_MINUTES
    bucket_min = base_min + offset
    h, m = divmod(bucket_min, 60)
    return ts.replace(hour=h, minute=m, second=0, microsecond=0)


def _clean_symbol(raw_symbol):
    return raw_symbol.split(":", 1)[-1]


def _instruments_filename(expiry_date_str):
    d, m, _y = expiry_date_str.split("-")
    return f"instruments_wk{m}{d}.json"


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _save_json(path, data):
    paths.atomic_write_json(path, data)


def _ensure_expiry_registered(expiry_info):
    """expiry_info: {'date': '08-09-2026', 'expiry': '...', 'expiry_flag': 'W'}.
    Returns (expiry_key, instruments_file). Adds a new expiries.json entry
    (with a fresh, empty instruments file) only if this expiry isn't
    already tracked -- never touches an existing entry."""
    expiries = _load_json(EXPIRIES_FILE, [])
    d, m, y = expiry_info["date"].split("-")
    expiry_key = f"{y}-{m}-{d}"
    for e in expiries:
        if e["expiry"] == expiry_key:
            return expiry_key, e["instruments_file"]
    inst_file = _instruments_filename(expiry_info["date"])
    label = f"{d}-{_MONTH_NAMES[int(m) - 1]}-{y} ({'Weekly' if expiry_info['expiry_flag'] == 'W' else 'Monthly'})"
    expiries.append({"expiry": expiry_key, "label": label, "instruments_file": inst_file})
    _save_json(EXPIRIES_FILE, expiries)
    inst_path = os.path.join(BASE, inst_file)
    if not os.path.exists(inst_path):
        _save_json(inst_path, [])
    return expiry_key, inst_file


def _ensure_instrument_registered(inst_file, symbol, strike, opt_type):
    inst_path = os.path.join(BASE, inst_file)
    instruments = _load_json(inst_path, [])
    if any(i["symbol"] == symbol for i in instruments):
        return
    instruments.append({"token": None, "symbol": symbol, "strike": strike, "type": opt_type})
    _save_json(inst_path, instruments)


def _last_row_date(path):
    """Cheap tail-read (no pandas, no full-file read) of just the last
    line's date field -- used to make appends idempotent across an app
    restart. Returns None on any read hiccup (missing/empty/header-only
    file) rather than raising, since this is only a best-effort dedupe
    check, not something that should ever block a write."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200), os.SEEK_SET)
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        if not lines or lines[-1].startswith("date,"):
            return None
        return lines[-1].split(",", 1)[0]
    except Exception:
        return None


def _append_candle(symbol, ts, o, h, l, c, vol, oi):
    """Appends one finished candle, but never a second row for a
    timestamp already on disk -- restarting the app (or the recorder
    thread) mid-bucket must not duplicate/corrupt that bucket's row, since
    the new accumulator would otherwise start from scratch and re-finalize
    the same 5-min window a second time."""
    data_dir = os.path.join(BASE, "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}.csv")
    is_new = not os.path.exists(path)
    ts_str = f"{ts.strftime('%Y-%m-%dT%H:%M:%S')}{TZ_SUFFIX}"
    if not is_new:
        last_date = _last_row_date(path)
        if last_date is not None and last_date >= ts_str:
            return
    with open(path, "a") as f:
        if is_new:
            f.write("date,open,high,low,close,volume,oi\n")
        f.write(f"{ts_str},{o},{h},{l},{c},{vol},{oi}\n")


def _poll_into_state(state, expiry_key, inst_file, chain, now):
    st = state.setdefault(expiry_key, {"bucket": None, "acc": {}, "inst_file": inst_file})
    b = _bucket_start(now)
    if st["bucket"] is not None and b != st["bucket"]:
        _finalize_and_write(st)
        st["acc"] = {}
    st["bucket"] = b

    for row in chain.get("strikes", []):
        for side_key, opt_type in (("ce", "CE"), ("pe", "PE")):
            leg = row.get(side_key)
            if not leg or leg.get("ltp") is None:
                continue
            symbol = _clean_symbol(leg["symbol"])
            ltp = leg["ltp"]
            vol = leg.get("volume") or 0
            oi = leg.get("oi") or 0
            a = st["acc"].get(symbol)
            if a is None:
                st["acc"][symbol] = {"open": ltp, "high": ltp, "low": ltp, "close": ltp,
                                      "start_vol": vol, "last_vol": vol, "oi": oi,
                                      "strike": row["strike"], "type": opt_type}
            else:
                a["high"] = max(a["high"], ltp)
                a["low"] = min(a["low"], ltp)
                a["close"] = ltp
                a["last_vol"] = vol
                a["oi"] = oi


def _finalize_and_write(st):
    ts = st["bucket"]
    if ts is None:
        return
    for symbol, a in st["acc"].items():
        vol = max(0, a["last_vol"] - a["start_vol"])
        _append_candle(symbol, ts, a["open"], a["high"], a["low"], a["close"], vol, a["oi"])
        _ensure_instrument_registered(st["inst_file"], symbol, a["strike"], a["type"])


def _flush_all(state):
    for st in state.values():
        if st["acc"]:
            _finalize_and_write(st)


def poll_once(state, now):
    """Runs exactly one poll cycle across the nearest two expiries,
    updating `state` in place. Shared by the in-app background thread
    (_record_loop, used when the GUI is open) and run_headless_session()
    (used by the standalone --record mode a Scheduled Task launches
    independently of the GUI) so both poll identically."""
    try:
        expiries = chain_mod.list_expiries()
    except Exception:
        return
    for exp_info in expiries[:2]:  # nearest two -- rolls forward past every expiry on its own
        try:
            expiry_key, inst_file = _ensure_expiry_registered(exp_info)
            chain = chain_mod.get_chain(strikecount=STRIKECOUNT, expiry_timestamp=exp_info["expiry"])
            _poll_into_state(state, expiry_key, inst_file, chain, now)
        except Exception:
            continue  # one bad poll shouldn't kill the recorder or the other expiry's data


def _record_loop(stop_event):
    if needs_backfill():
        print("[historical_recorder] Gap detected in recent trading days -- backfilling from Fyers...")
        try:
            new_rows = backfill_missing_days()
            print(f"[historical_recorder] Backfill done: {new_rows} new rows.")
        except Exception as e:
            print(f"[historical_recorder] Backfill failed (will retry next start): {e}")

    state = {}
    while not stop_event.is_set():
        # IST + NSE calendar: a weekend/holiday used to be polled and written
        # to data/*.csv as flat zero-volume "trading days" whenever the app
        # happened to be open on one.
        now = mc.now_ist().replace(tzinfo=None)
        _maybe_refresh_eod(now)
        if not mc.is_market_open():
            if state:
                _flush_all(state)
                state = {}
            stop_event.wait(IDLE_POLL_SEC)
            continue
        poll_once(state, now)
        stop_event.wait(POLL_INTERVAL_SEC)
    _flush_all(state)


_eod_refreshed_on = None


def _maybe_refresh_eod(now):
    """Once a day after 19:00 IST (NSE has published the bhavcopy by then):
    download any missing days and rebuild the ATM-IV / PCR summary that the
    chain's IV Rank reads. Without this the IV history silently went stale
    (it had stopped at 2026-09-09)."""
    global _eod_refreshed_on
    if now.hour < 19 or _eod_refreshed_on == now.date():
        return
    _eod_refreshed_on = now.date()
    try:
        from datetime import timedelta as _td
        import eod_analysis
        import nse_eod_fetcher
        if nse_eod_fetcher.backfill_range(now.date() - _td(days=10), now.date()):
            eod_analysis.build_daily_summary()
    except Exception as e:
        print(f"[historical_recorder] EOD bhavcopy refresh failed (retry tomorrow): {e}")


def _state_total_volume(state):
    """Sums each symbol's latest-seen cumulative day-volume straight out
    of the poll_once() accumulator state -- reuses data already being
    fetched for recording, so the holiday check costs zero extra Fyers
    calls instead of polling separately on the side."""
    return sum(a.get("last_vol", 0) for st in state.values() for a in st["acc"].values())


def run_headless_session():
    """Blocking, single-trading-day run with NO dashboard/browser/paper-
    trading -- meant for a Windows Scheduled Task to launch independently
    of whether the user ever opens the GUI that day. Waits for market
    open if started early, then starts recording IMMEDIATELY (a Windows
    Scheduled Task fires every Mon-Fri regardless of NSE's own holiday
    calendar, so whether today is a genuine trading day is only known in
    hindsight) -- an earlier version checked for a holiday BEFORE
    recording by watching volume for HOLIDAY_CHECK_DELAY_SEC first, which
    meant every single real trading day silently lost its opening 5
    minutes of candles waiting on that check. Instead, the holiday
    verdict is now decided from the very same in-progress recording after
    HOLIDAY_CHECK_DELAY_SEC has elapsed: if cumulative volume hasn't
    moved by then, the whole day-so-far is discarded (a closed market's
    handful of flat candles, worth losing); otherwise recording just
    continues uninterrupted through the moment nothing was ever lost.
    Safe to run at the same time as the GUI's in-app recorder thread --
    _append_candle's duplicate-timestamp check means whichever of the two
    writes a given candle first, the other just skips it."""
    import time as time_mod
    print("Historical recorder (headless) starting...")

    if needs_backfill():
        print("Gap detected in recent trading days -- backfilling from Fyers before today's recording...")
        try:
            new_rows = backfill_missing_days()
            print(f"Backfill done: {new_rows} new rows.")
        except Exception as e:
            print(f"Backfill failed (will retry next run): {e}")

    if not mc.is_trading_day(mc.now_ist().date()):
        print(f"{mc.now_ist().date()} is not an NSE trading day ({mc.market_state()['label']}) -- nothing to record.")
        return
    if mc.now_ist().time() < MARKET_OPEN:
        print("Waiting for market open (09:15 IST)...")
    if not mc.wait_until_open():
        print("Market already closed for today -- nothing to record.")
        return

    print("Recording started. Will discard today's data if it turns out to be an NSE holiday "
          f"(checked from the recording itself after {HOLIDAY_CHECK_DELAY_SEC // 60} min).")
    state = {}
    vol_at_open = None
    holiday_checked = False
    check_after = datetime.now().timestamp() + HOLIDAY_CHECK_DELAY_SEC

    while mc.now_ist().time() <= MARKET_CLOSE:
        poll_once(state, mc.now_ist().replace(tzinfo=None))
        if vol_at_open is None:
            vol_at_open = _state_total_volume(state)
        if not holiday_checked and datetime.now().timestamp() >= check_after:
            holiday_checked = True
            if _state_total_volume(state) - vol_at_open <= HOLIDAY_VOLUME_THRESHOLD:
                print("No real trading volume seen -- looks like a market holiday. Discarding today's data.")
                return
            print("Real trading activity confirmed -- continuing to record normally.")
        time_mod.sleep(POLL_INTERVAL_SEC)

    _flush_all(state)
    print("Market closed -- recording finished for today.")


def _fetch_history_candles(fyers_symbol, from_date, to_date, resolution=CANDLE_MINUTES):
    """fyers_symbol must include the 'NSE:' prefix. Returns a list of
    (epoch_ts, open, high, low, close, volume, oi) tuples straight from
    Fyers' own /data/history endpoint -- the SAME data a real historical
    API (like Kite's) would give, just from the broker we're already
    authenticated with. Only ever has data for a CURRENTLY-LISTED (not
    yet expired) contract -- once an expiry passes, this stops returning
    anything for it, same limitation every broker's historical API has
    for options. Returns [] on any failure or genuine no-data response
    (e.g. a strike that simply never traded)."""
    try:
        headers = {"Authorization": fyers_auth.get_auth_header()}
        params = {"symbol": fyers_symbol, "resolution": str(resolution), "date_format": "1",
                  "range_from": from_date, "range_to": to_date, "cont_flag": "0", "oi_flag": "1"}
        resp = requests.get(HISTORY_URL, headers=headers, params=params, timeout=15)
        body = resp.json()
        if body.get("s") != "ok":
            return []
        return body.get("candles", [])
    except Exception:
        return []


def _currently_listed_expiries():
    """The nearest-two expiries as Fyers reports them RIGHT NOW, matched
    back to their expiries.json entries. expiries.json only ever grows
    (every past expiry stays registered forever, so the Simulator can
    still browse old weeks) -- but Fyers stops serving history for a
    contract once it expires, so backfill/gap-checking must only ever
    look at the currently-listed pair, not that whole accumulated list.
    Skipping this would mean the wasted-call cost grows every single
    week as expiries.json keeps accumulating dead entries. Falls back to
    every registered expiry if the live Fyers lookup itself fails, so a
    transient API hiccup doesn't silently stop backfill from working."""
    try:
        live = chain_mod.list_expiries()[:2]
    except Exception:
        return _load_json(EXPIRIES_FILE, [])
    registered = _load_json(EXPIRIES_FILE, [])
    by_key = {e["expiry"]: e for e in registered}
    out = []
    for exp_info in live:
        d, m, y = exp_info["date"].split("-")
        key = f"{y}-{m}-{d}"
        if key in by_key:
            out.append(by_key[key])
    return out


def _last_recorded_datetime(inst_file):
    """Best-effort newest (date, time) recorded for this expiry -- checks
    a handful of instruments (not just one, since an illiquid strike can
    have stopped trading long before the real close) and returns the
    latest. Used to catch a TRAILING gap: a day that has SOME data and so
    isn't "missing" from list_missing_weekdays()'s point of view, but
    still stopped short of the real close -- e.g. the app used a stale
    MARKET_CLOSE value (exactly what happened when NSE's F&O close moved
    from 15:30 to 15:40 and the old exe build kept recording to the old
    time), or genuinely closed a few minutes early that one day."""
    from datetime import datetime as dt_cls
    instruments = _load_json(os.path.join(BASE, inst_file), [])
    latest = None
    for inst in instruments[:10]:
        path = os.path.join(BASE, "data", f"{inst['symbol']}.csv")
        if not os.path.exists(path):
            continue
        last = _last_row_date(path)
        if last is None:
            continue
        try:
            ts = dt_cls.strptime(last[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def needs_backfill(days_back=15):
    """Cheap, network-free check for whether a real backfill is worth
    running: any weekday in the last `days_back` days (up to and
    INCLUDING yesterday -- today is deliberately excluded since the live
    recorder is what's supposed to fill today in, not this) that's
    missing from at least one CURRENTLY-LISTED expiry's already-recorded
    dates. Lets the recorder skip the slow ~176-symbol network backfill
    entirely on a normal day when nothing's actually missing, instead of
    paying that cost every single startup."""
    from data_health import _recorded_dates  # recorded CSVs (the Simulator now reads data/hist1m)
    from datetime import datetime, timedelta
    expiries = _currently_listed_expiries()
    if not expiries:
        return False
    yesterday = (datetime.now() - timedelta(days=1)).date()
    window_start = yesterday - timedelta(days=days_back)
    today = datetime.now().date()
    for exp in expiries:
        have = set(_recorded_dates(exp["expiry"]))
        if not have:
            return True
        # Only gaps AFTER the first recorded day count -- days before it are
        # usually before the contract was even listed (a new weekly has no
        # history to fetch), and counting them re-ran a full ~176-symbol
        # backfill on every single startup, forever.
        d = max(window_start, datetime.strptime(min(have), "%Y-%m-%d").date())
        while d <= yesterday:
            if mc.is_trading_day(d) and d.strftime("%Y-%m-%d") not in have:
                return True
            d += timedelta(days=1)

        # Trailing-gap check: the most recent FULLY PAST trading day might
        # be "present" but still stop short of the real close.
        latest_dt = _last_recorded_datetime(exp["instruments_file"])
        if latest_dt is not None and latest_dt.date() < today:
            latest_close = datetime.combine(latest_dt.date(), MARKET_CLOSE)
            if (latest_close - latest_dt).total_seconds() > CANDLE_MINUTES * 60:
                return True
    return False


def _merge_candles(symbol, candles):
    """Merges Fyers history rows into data/<symbol>.csv BY TIMESTAMP and
    rewrites the file sorted, atomically. The old path appended through
    _append_candle(), whose "skip anything not newer than the last row"
    guard meant a gap in the MIDDLE of the history (a day the app wasn't
    running, with later days already recorded) could never be filled --
    every backfill fetched ~176 symbols and wrote nothing. Exchange history
    rows also REPLACE live-recorded rows for the same timestamp: those were
    built from 15s LTP polling and understate true highs/lows. Returns the
    number of timestamps that weren't on disk before."""
    import csv
    path = os.path.join(BASE, "data", f"{symbol}.csv")
    rows = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("date"):
                    rows[r["date"]] = r
    before = set(rows)
    for c in candles:
        ts = datetime.fromtimestamp(c[0], mc.IST)
        key = f"{ts.strftime('%Y-%m-%dT%H:%M:%S')}{TZ_SUFFIX}"
        rows[key] = {"date": key, "open": c[1], "high": c[2], "low": c[3], "close": c[4],
                     "volume": c[5], "oi": c[6] if len(c) > 6 else 0}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume", "oi"], extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    os.replace(tmp, path)
    return len(set(rows) - before)


def backfill_missing_days(days_back=15, on_progress=None):
    """One-shot catch-up for a gap the live recorder missed (app was
    closed, a holiday was misjudged, etc.) -- for every strike already
    known in the nearest-two-expiries' instruments files, pulls whatever
    `days_back` calendar days of 5-min history Fyers still has and
    appends any candle not already on disk (via _append_candle's own
    duplicate-timestamp guard, so it's safe to re-run any time -- already-
    recorded candles are simply skipped, not duplicated).

    MUST be run before that expiry's contracts actually expire -- Fyers,
    like every broker, stops serving history for an option once it's no
    longer currently listed (confirmed live: a genuinely-traded strike's
    history is available end-to-end up to now, but there's no reason to
    expect that survives past expiry -- the same documented limitation
    already known for Kite). `on_progress(symbol, new_rows)` is called
    after each symbol if given, for a caller that wants to show progress.
    Returns the total number of new rows written."""
    from datetime import datetime, timedelta
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    expiries = _currently_listed_expiries()
    total_new = 0
    for exp in expiries:
        inst_path = os.path.join(BASE, exp["instruments_file"])
        for inst in _load_json(inst_path, []):
            symbol = inst["symbol"]
            candles = _fetch_history_candles(f"NSE:{symbol}", from_date, to_date)
            new_for_symbol = _merge_candles(symbol, candles) if candles else 0
            total_new += new_for_symbol
            if on_progress:
                on_progress(symbol, new_for_symbol)
    return total_new


def start_background_recorder():
    """Idempotent -- safe to call once at app startup. No-op if already running."""
    global _thread, _stop_event
    if _thread is not None and _thread.is_alive():
        return _thread
    _stop_event = threading.Event()
    _thread = threading.Thread(target=_record_loop, args=(_stop_event,), daemon=True, name="historical-recorder")
    _thread.start()
    return _thread
