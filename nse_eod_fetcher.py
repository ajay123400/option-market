"""nse_eod_fetcher.py -- downloads NSE's own daily F&O Bhavcopy (a public,
free, permanent archive -- unlike Fyers' broker API, it's never restricted
to "currently listed" contracts and goes back years) and saves NIFTY index
options + futures rows into data/eod/<date>.csv, one file per trading day.

This is a SEPARATE, lower-granularity dataset from the 5-min intraday CSVs
historical_recorder.py builds -- Bhavcopy is only ONE row per contract per
day (open/high/low/close/settlement/OI/volume, plus the underlying's own
close price), not intraday candles, so it can't feed the Simulator's
candle-by-candle replay. It's meant for longer-history analysis/ML
training instead, where the Simulator's month or two of 5-min data isn't
enough -- confirmed live that a single day's file already carries every
NIFTY option strike/expiry combination in ~1,600 rows, all the way back
however many years NSE keeps this archive for.

NSE requires a real browser-like session -- Akamai bot protection blocks a
bare request to nseindia.com outright (403, no cookies at all). Hitting
nseindia.com/all-reports first (with a normal browser User-Agent) sets the
Akamai cookies the archive download needs; the archive URL itself then
works with those cookies plus a Referer header pointing back at that page.
"""
import csv
import io
import os
import time
import zipfile
from datetime import datetime, timedelta

import requests

import paths

BASE = paths.BASE_DIR
EOD_DIR = os.path.join(BASE, "data", "eod")
NSE_HOME = "https://www.nseindia.com/all-reports"
BHAVCOPY_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
REQUEST_DELAY_SEC = 0.5  # be a polite scraper, not a hammer
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _session():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(NSE_HOME, timeout=15)  # sets the Akamai cookies the archive URL needs
    return s


def fetch_day(session, date_obj):
    """Returns filtered NIFTY index option/future rows for one date as a
    list of dicts, or None if that date has no bhavcopy (weekend, holiday,
    or not yet published -- NSE just 404s or returns a tiny/invalid body
    for those, both treated the same way here)."""
    date_str = date_obj.strftime("%Y%m%d")
    url = BHAVCOPY_URL.format(date=date_str)
    try:
        resp = session.get(url, timeout=20, headers={"Referer": NSE_HOME})
    except requests.RequestException:
        return None
    if resp.status_code != 200 or len(resp.content) < 1000:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile:
        return None
    with z.open(z.namelist()[0]) as f:
        text = f.read().decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    return [row for row in reader if row["TckrSymb"] == "NIFTY" and row["FinInstrmTp"] in ("IDO", "IDF")]


def save_day(date_obj, rows):
    os.makedirs(EOD_DIR, exist_ok=True)
    path = os.path.join(EOD_DIR, f"{date_obj.strftime('%Y-%m-%d')}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def backfill_range(from_date, to_date, on_progress=None):
    """Downloads every trading day's NIFTY F&O bhavcopy between from_date
    and to_date (inclusive) into data/eod/, skipping days already saved
    (idempotent -- safe to re-run/extend the range any time) and weekends.
    `on_progress(date, num_rows_or_None)` is called per day if given.
    Returns the number of new day-files written."""
    session = _session()
    written = 0
    d = from_date
    while d <= to_date:
        path = os.path.join(EOD_DIR, f"{d.strftime('%Y-%m-%d')}.csv")
        if d.weekday() < 5 and not os.path.exists(path):
            rows = fetch_day(session, d)
            if rows:
                save_day(d, rows)
                written += 1
            if on_progress:
                on_progress(d, len(rows) if rows else None)
            time.sleep(REQUEST_DELAY_SEC)
        d += timedelta(days=1)
    return written


if __name__ == "__main__":
    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=30)

    def progress(d, n):
        print(f"{d}: {n} rows" if n else f"{d}: no data (holiday/weekend/not yet published)")

    total = backfill_range(from_date, to_date, on_progress=progress)
    print(f"\nDone -- {total} new day-file(s) written to {EOD_DIR}")
