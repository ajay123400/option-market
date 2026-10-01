"""participant_oi.py -- NSE's daily "Participant wise Open Interest" file
(Client / DII / FII / Pro positions in index & stock futures and options,
end of day), downloaded once per trading day and cached.

Files:  data/participant_oi/<YYYY-MM-DD>.csv   (raw, as published)
        data/participant_oi/all.parquet        (one row per day x client type)
Usage:  python participant_oi.py [--from 2021-09-01]
"""
import io
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

import market_calendar as mc
import paths

DIR = os.path.join(paths.BASE_DIR, "data", "participant_oi")
URL = "https://archives.nseindia.com/content/nsccl/fao_participant_oi_{d}.csv"
HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "text/csv,*/*"}
COLS = ["client", "fut_idx_long", "fut_idx_short", "fut_stk_long", "fut_stk_short", "idx_call_long", "idx_put_long",
        "idx_call_short", "idx_put_short", "stk_call_long", "stk_put_long", "stk_call_short", "stk_put_short",
        "total_long", "total_short"]


def fetch(start, end=None, pause=0.35, progress=print):
    os.makedirs(DIR, exist_ok=True)
    end = end or date.today()
    d, got, miss = start, 0, []
    s = requests.Session()
    while d <= end:
        path = os.path.join(DIR, f"{d.isoformat()}.csv")
        if mc.is_trading_day(d) and not os.path.exists(path):
            try:
                r = s.get(URL.format(d=d.strftime("%d%m%Y")), headers=HEAD, timeout=30)
                if r.ok and b"Client" in r.content:
                    with open(path, "wb") as f:
                        f.write(r.content)
                    got += 1
                else:
                    miss.append(d.isoformat())
            except requests.RequestException:
                miss.append(d.isoformat())
            time.sleep(pause)
            if got and got % 50 == 0:
                progress(f"{d} ... {got} files")
        d += timedelta(days=1)
    return got, miss


def load():
    rows = []
    for fn in sorted(os.listdir(DIR)):
        if not fn.endswith(".csv"):
            continue
        txt = open(os.path.join(DIR, fn), encoding="utf-8", errors="replace").read()
        lines = [ln for ln in txt.splitlines() if ln.split(",")[0].strip() in ("Client", "DII", "FII", "Pro", "TOTAL")]
        df = pd.read_csv(io.StringIO("\n".join(lines)), header=None).iloc[:, :15]
        df.columns = COLS
        df["client"] = df.client.str.strip()
        df.insert(0, "day", fn[:-4])
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    for c in COLS[1:]:
        out[c] = pd.to_numeric(out[c].astype(str).str.strip(), errors="coerce")
    out.to_parquet(os.path.join(DIR, "all.parquet"), index=False)
    return out


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    frm = date.fromisoformat(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else date(2021, 9, 1)
    got, miss = fetch(frm)
    print("downloaded", got, "missing", len(miss), miss[:20])
    df = load()
    print(df.day.nunique(), "days", df.day.min(), "->", df.day.max())
