"""data_health.py -- read-only visibility into the historical data this
app has actually accumulated: which trading days are recorded per expiry,
which are missing, how fresh the last-recorded candle is, how much disk
space it's using, and whether the "OptionMarket Historical Recorder"
Scheduled Task is actually registered and running. Exists so the answer
to "is my data okay?" is a page load, not a manual poke through data/*.csv.
"""
import json
import os
import re
import subprocess
from datetime import datetime

import paths

BASE = paths.BASE_DIR


# ---- recorded-CSV inventory (data/<symbol>.csv written by historical_recorder).
# The Simulator no longer replays these (it uses the downloaded 1-min history in
# data/hist1m/), but Data Health still reports what the recorder has captured.
def _recorded_expiries():
    with open(f"{BASE}/expiries.json") as f:
        return json.load(f)


def _recorded_instruments(expiry):
    for e in _recorded_expiries():
        if e["expiry"] == expiry:
            with open(f"{BASE}/{e['instruments_file']}") as f:
                return json.load(f)
    raise ValueError(f"Unknown expiry {expiry!r} -- not in expiries.json")


_dates_cache = {}  # path -> ((mtime, size), set_of_dates)


def _file_dates(path):
    """YYYY-MM-DD dates in one CSV (first 10 chars of each row), cached by mtime/size."""
    try:
        st = os.stat(path)
    except OSError:
        return set()
    key = (st.st_mtime, st.st_size)
    hit = _dates_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    days = set()
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            next(f, None)
            for line in f:
                d = line[:10]
                if len(d) == 10 and d[4] == "-" and d[7] == "-":
                    days.add(d)
    except OSError:
        pass
    _dates_cache[path] = (key, days)
    return days


def _recorded_dates(expiry):
    days = set()
    for inst in _recorded_instruments(expiry):
        days |= _file_dates(f"{BASE}/data/{inst['symbol']}.csv")
    return sorted(days)


def _recorded_missing(dates):
    """Trading days between the first and last recorded date with no data (NSE holidays excluded)."""
    import market_calendar as mc
    from datetime import date, timedelta
    if len(dates) < 2:
        return []
    have, out = set(dates), []
    d, end = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
    while d <= end:
        if mc.is_trading_day(d) and d.isoformat() not in have:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out

BASE = paths.BASE_DIR
DATA_DIR = os.path.join(BASE, "data")
SCHEDULED_TASK_NAME = "OptionMarket Daily History"
HIST_DIR = os.path.join(BASE, "data", "hist1m")
DAILY_STATUS = os.path.join(HIST_DIR, "daily_status.json")
RECENT_EXPIRIES = 15


def _dir_size_and_count(path):
    total_bytes, count = 0, 0
    if not os.path.isdir(path):
        return 0, 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total_bytes += os.path.getsize(os.path.join(root, name))
            count += 1
    return total_bytes, count


def _last_recorded_at():
    """Newest mtime across every data/*.csv -- the freshest signal for
    "when did anything last actually get written", independent of which
    expiry or symbol it was."""
    latest = None
    if not os.path.isdir(DATA_DIR):
        return None
    for entry in os.scandir(DATA_DIR):
        if entry.is_file() and entry.name.endswith(".csv"):
            mtime = entry.stat().st_mtime
            if latest is None or mtime > latest:
                latest = mtime
    return latest


def _parse_ps_date(val):
    """PowerShell's ConvertTo-Json renders DateTime as '/Date(<ms>)/'
    (the old Microsoft JSON date convention) -- converts that to epoch
    seconds for the frontend's existing fmtWhen()/minutesAgo() helpers.
    A task that's never run reports Windows' sentinel default
    (30-Nov-1999), which is surfaced here as None ("never") rather than
    a confusing 1999 timestamp."""
    if not val:
        return None
    m = re.match(r"/Date\((-?\d+)\)/", val)
    if not m:
        return None
    epoch = int(m.group(1)) / 1000
    try:
        if datetime.fromtimestamp(epoch).year <= 1999:
            return None
    except Exception:
        pass
    return epoch


def _scheduled_task_status():
    """Best-effort -- returns {'exists': False} on any failure (task not
    created yet, schtasks/powershell unavailable, non-Windows, etc.)
    rather than raising, since this is purely informational."""
    try:
        cmd = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            f"Get-ScheduledTask -TaskName '{SCHEDULED_TASK_NAME}' -ErrorAction Stop | "
            f"Get-ScheduledTaskInfo | Select-Object LastRunTime,LastTaskResult,NextRunTime | ConvertTo-Json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"exists": False}
        data = json.loads(proc.stdout)
        return {
            "exists": True,
            "last_run": _parse_ps_date(data.get("LastRunTime")),
            "last_result": data.get("LastTaskResult"),
            "next_run": _parse_ps_date(data.get("NextRunTime")),
        }
    except Exception:
        return {"exists": False}


def get_health_report():
    # The PowerShell scheduled-task query takes ~2s on its own -- run it in
    # parallel with the file scan instead of after it.
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    task_future = pool.submit(_scheduled_task_status)
    # data/hist1m -- the 1-min history the Simulator, Charts and backtests use,
    # kept current each evening by daily_history.py (scheduled task above)
    import simulator
    expiries_report = []
    man = simulator._manifest()
    for e in simulator.list_expiries()[:RECENT_EXPIRIES]:
        rec = man.get(e["expiry"], {})
        w0, _ = simulator._week_of(e["expiry"])
        dates = [d for d in simulator.list_available_dates(e["expiry"]) if d >= w0.isoformat()]
        expiries_report.append({
            "expiry": e["expiry"],
            "label": e["label"] + (" · live (updated each evening)" if rec.get("status") == "live" else ""),
            "num_instruments": rec.get("contracts"),
            "num_days": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "missing_dates": simulator.list_missing_weekdays(e["expiry"]),
        })

    total_bytes, num_files = _dir_size_and_count(HIST_DIR)
    last_recorded = None
    try:
        with open(DAILY_STATUS, encoding="utf-8") as f:
            last_recorded = datetime.fromisoformat(json.load(f)["last_run"]).timestamp()
    except (OSError, ValueError, KeyError):
        pass

    return {
        "expiries": expiries_report,
        "storage": {
            "total_bytes": total_bytes,
            "num_files": num_files,
            "data_dir": HIST_DIR,
        },
        "last_recorded_at": last_recorded,
        "scheduled_task": task_future.result(),
    }
