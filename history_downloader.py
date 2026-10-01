"""history_downloader.py -- 5 years of NIFTY 1-minute history from Fyers'
"Expired F&O Contracts Data" API (+ the normal history API for the index).

What it stores (under data/hist1m/):
  NIFTY50_1m.parquet            NIFTY 50 spot, 1-min OHLC (ts = epoch seconds)
  options/<expiry>.parquet      every contract of that expiry within ATM +/-10
                                strikes: symbol, type, strike, ts, OHLC, volume, oi
  manifest.json                 per expiry: status, strikes, contracts, rows (resume point)
  download.log                  progress log

Strike selection per expiry ("ATM +/-10 strikes"): NIFTY's low and high over
that expiry's trading week (the day after the previous expiry -> expiry, plus
one prior day for the baseline) are rounded to 50-pt strikes; every strike
from lowest ATM - 10 strikes to highest ATM + 10 strikes is kept, so ATM +/-10
is covered at every moment of the week. Each contract is fetched over its
whole traded life (up to the API's 100-day window) -- one call either way.

Rate limits (Fyers Standard plan): 10/s, 200/min, 1,00,000/day. This stays
under ~180/min and backs off on 429s. Safe to stop and re-run: finished
expiries are skipped (manifest), newest weeks are fetched first.

Usage:  python history_downloader.py                 # last 5 years
        python history_downloader.py --years 1       # or a shorter span
        python history_downloader.py --test          # one expiry only
"""
import json
import os
import re
import sys
import time
from collections import deque
from datetime import date, datetime, timedelta

import pandas as pd
import requests

import fyers_auth
import market_calendar as mc
import paths

OUT = os.path.join(paths.BASE_DIR, "data", "hist1m")
OPT_DIR = os.path.join(OUT, "options")
SPOT_PATH = os.path.join(OUT, "NIFTY50_1m.parquet")
MANIFEST = os.path.join(OUT, "manifest.json")
LOG = os.path.join(OUT, "download.log")
EXP_API = "https://api-t1.fyers.in/data/history/fno/expired"
HIST_API = "https://api-t1.fyers.in/data/history"
INDEX = "NSE:NIFTY50-INDEX"
STEP = 50                 # NIFTY strike interval
ATM_SIDE = 10             # strikes kept on each side of ATM
PER_MIN, PER_SEC = 180, 8  # stay under Fyers' 200/min and 10/s
_MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_WK = {**{i: str(i) for i in range(1, 10)}, 10: "O", 11: "N", 12: "D"}


def log(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---- rate-limited GET ----------------------------------------------------------
_calls = deque()


def _get(url, params, tries=5):
    for attempt in range(tries):
        now = time.time()
        while _calls and now - _calls[0] > 60:
            _calls.popleft()
        if len(_calls) >= PER_MIN:
            time.sleep(60 - (now - _calls[0]) + 0.2)
        recent = [t for t in _calls if now - t < 1]
        if len(recent) >= PER_SEC:
            time.sleep(1.0)
        _calls.append(time.time())
        try:
            r = requests.get(url, headers={"Authorization": fyers_auth.get_auth_header()}, params=params, timeout=60)
            if r.status_code == 429:
                log("rate limited (429) -- waiting 65s")
                time.sleep(65)
                continue
            body = r.json()
            if body.get("code") == -429 or "rate limit" in str(body.get("message", "")).lower():
                log("rate limited -- waiting 65s")
                time.sleep(65)
                continue
            return body
        except Exception as e:
            log(f"request error ({type(e).__name__}: {e}) -- retry {attempt + 1}")
            time.sleep(5 * (attempt + 1))
    return {"s": "error", "message": "gave up after retries"}


def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


# ---- NIFTY spot -----------------------------------------------------------------
def download_spot(start, end):
    """NIFTY 50 1-min OHLC from start to end, in <=90-day chunks; appends to
    whatever is already on disk."""
    have = pd.read_parquet(SPOT_PATH) if os.path.exists(SPOT_PATH) else pd.DataFrame()
    if not have.empty:
        last = datetime.fromtimestamp(int(have["ts"].max()), mc.IST).date()
        first = datetime.fromtimestamp(int(have["ts"].min()), mc.IST).date()
        if first <= start + timedelta(days=5):
            start = last  # re-pull the last saved day in full (it may have been saved mid-session)
    parts = [have] if not have.empty else []
    d = start
    while d <= end:
        e = min(d + timedelta(days=89), end)
        b = _get(HIST_API, {"symbol": INDEX, "resolution": "1", "date_format": "1",
                            "range_from": d.isoformat(), "range_to": e.isoformat(), "cont_flag": "1"})
        c = b.get("candles") or []
        if c:
            parts.append(pd.DataFrame(c, columns=["ts", "open", "high", "low", "close", "volume"]).drop(columns="volume"))
        log(f"spot {d} -> {e}: {len(c)} candles ({b.get('s')})")
        d = e + timedelta(days=1)
    if parts:
        df = pd.concat(parts).drop_duplicates("ts", keep="last").sort_values("ts")
        df["ts"] = df["ts"].astype("int64")
        os.makedirs(OUT, exist_ok=True)
        df.to_parquet(SPOT_PATH, compression="zstd", index=False)
        return df
    return have


# ---- expiries / contracts ---------------------------------------------------------
def list_expiries(start, end):
    out = set()
    d = start
    while d <= end:
        e = min(d + timedelta(days=365), end)
        b = _get(f"{EXP_API}/expiry-dates", {"symbol": INDEX, "range_from": d.isoformat(),
                                             "range_to": e.isoformat(), "date_format": 1})
        out |= set(((b.get("data") or {}).get("expiry_dates") or {}).get("options") or [])
        d = e + timedelta(days=1)
    return sorted(out)


def _parse_symbol(sym, exp):
    """(strike, 'CE'|'PE') from an expired NIFTY option symbol, using the
    known expiry to strip the date part (weekly NIFTY<yy><m><dd>, monthly
    NIFTY<yy><MON>) so strike digits aren't confused with date digits."""
    m = re.match(r"^NSE:NIFTY(.+)(CE|PE)$", sym)
    if not m:
        return None, None
    body, typ = m.group(1), m.group(2)
    yy = f"{exp.year % 100:02d}"
    for pre in (f"{yy}{_WK[exp.month]}{exp.day:02d}", f"{yy}{_MON[exp.month - 1]}"):
        if body.startswith(pre) and body[len(pre):].isdigit():
            return int(body[len(pre):]), typ
    return None, None


def _week_window(exp, prev_exp):
    start = (prev_exp + timedelta(days=1)) if prev_exp else exp - timedelta(days=7)
    return mc.previous_trading_day(start) if start <= exp else exp - timedelta(days=7), exp


def download_expiry(exp, prev_exp, spot):
    """All ATM +/-10 contracts of one expiry -> options/<expiry>.parquet."""
    w0, w1 = _week_window(exp, prev_exp)
    t0 = int(datetime.combine(w0, datetime.min.time(), mc.IST).timestamp())
    t1 = int(datetime.combine(w1, datetime.max.time(), mc.IST).timestamp())
    wk = spot[(spot["ts"] >= t0) & (spot["ts"] <= t1)]
    if wk.empty:
        return None, "no NIFTY spot data for the week"
    lo_atm = int(round(wk["low"].min() / STEP) * STEP)
    hi_atm = int(round(wk["high"].max() / STEP) * STEP)
    lo_k, hi_k = lo_atm - ATM_SIDE * STEP, hi_atm + ATM_SIDE * STEP
    b = _get(f"{EXP_API}/underlying-symbols", {"symbol": INDEX, "expiry_date": exp.isoformat()})
    syms = ((b.get("data") or {}).get("contracts") or {}).get("options") or []
    wanted = []
    for s in syms:
        k, typ = _parse_symbol(s, exp)
        if k is not None and lo_k <= k <= hi_k:
            wanted.append((s, k, typ))
    if not wanted:
        return None, f"no contracts in {lo_k}-{hi_k} (list: {len(syms)}, {b.get('s')} {b.get('message')})"
    frames, empty = [], 0
    rf = (exp - timedelta(days=99)).isoformat()
    for s, k, typ in wanted:
        h = _get(f"{EXP_API}/historical-data", {"symbol": s, "resolution": "1", "date_format": 1,
                                                "range_from": rf, "range_to": exp.isoformat(), "include_oi": 1})
        c = h.get("candles") or []
        if not c:
            empty += 1
            continue
        cols = h.get("columns") or ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
        df = pd.DataFrame(c, columns=cols[:len(c[0])])
        df = df.rename(columns={"timestamp": "ts", "open_interest": "oi"})
        if "oi" not in df:
            df["oi"] = 0
        df.insert(0, "symbol", s)
        df.insert(1, "type", typ)
        df.insert(2, "strike", k)
        frames.append(df)
    if not frames:
        return None, "no candles for any contract"
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = out["ts"].astype("int64")
    for c in ("volume", "oi"):
        out[c] = out[c].fillna(0).astype("int64")
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].astype("float32")
    out["strike"] = out["strike"].astype("int32")
    os.makedirs(OPT_DIR, exist_ok=True)
    out.to_parquet(os.path.join(OPT_DIR, f"{exp.isoformat()}.parquet"), compression="zstd", index=False)
    return {"status": "done", "strikes": [lo_k, hi_k], "contracts": len(wanted), "empty": empty,
            "rows": int(len(out)), "week": [w0.isoformat(), w1.isoformat()],
            "fetched_at": datetime.now().isoformat(timespec="seconds")}, None


def ensure_strikes(expiry, from_day, progress=None):
    """Top up one expiry's file so ATM +/-10 strikes are covered from
    `from_day` through expiry (the bulk download only covered the expiry
    week). Fetches just the missing contracts and merges them in. Strikes
    that came back empty are remembered so they're never asked for twice.
    Returns (contracts_added, error_or_None)."""
    exp = date.fromisoformat(expiry) if isinstance(expiry, str) else expiry
    from_day = date.fromisoformat(from_day) if isinstance(from_day, str) else from_day
    path = os.path.join(OPT_DIR, f"{exp.isoformat()}.parquet")
    if not os.path.exists(path) or not os.path.exists(SPOT_PATH):
        return 0, "no downloaded data for this expiry"
    spot = pd.read_parquet(SPOT_PATH)
    t0 = int(datetime.combine(from_day, datetime.min.time(), mc.IST).timestamp())
    t1 = int(datetime.combine(exp, datetime.max.time(), mc.IST).timestamp())
    wk = spot[(spot["ts"] >= t0) & (spot["ts"] <= t1)]
    if wk.empty:
        return 0, None
    lo_k = int(round(wk["low"].min() / STEP) * STEP) - ATM_SIDE * STEP
    hi_k = int(round(wk["high"].max() / STEP) * STEP) + ATM_SIDE * STEP
    return ensure_strike_range(exp, lo_k, hi_k, progress, why=f"from {from_day}")


def ensure_strike_range(expiry, lo_k, hi_k, progress=None, why=""):
    """Make sure every CE/PE strike from lo_k to hi_k (50-pt steps) of this
    expiry is on disk -- fetches only what's missing. Returns (added, error)."""
    exp = date.fromisoformat(expiry) if isinstance(expiry, str) else expiry
    path = os.path.join(OPT_DIR, f"{exp.isoformat()}.parquet")
    if not os.path.exists(path):
        return 0, "no downloaded data for this expiry"
    have = pd.read_parquet(path)
    manifest = _load_json(MANIFEST, {})
    rec = manifest.get(exp.isoformat()) or {}
    tried = set(rec.get("tried_empty") or [])
    have_syms = set(have["symbol"].unique())
    need = [(k, t) for k in range(lo_k, hi_k + 1, STEP) for t in ("CE", "PE")]
    b = None
    missing = []
    for k, t in need:
        if any(s.endswith(f"{k}{t}") and _parse_symbol(s, exp) == (k, t) for s in have_syms):
            continue
        missing.append((k, t))
    missing = [(k, t) for k, t in missing if f"{k}{t}" not in tried]
    if not missing:
        return 0, None
    b = _get(f"{EXP_API}/underlying-symbols", {"symbol": INDEX, "expiry_date": exp.isoformat()})
    syms = ((b.get("data") or {}).get("contracts") or {}).get("options") or []
    by_key = {}
    for sym in syms:
        k, t = _parse_symbol(sym, exp)
        if k is not None:
            by_key[(k, t)] = sym
    frames = []
    rf = (exp - timedelta(days=99)).isoformat()
    for n, (k, t) in enumerate(missing, 1):
        if progress:
            progress(n, len(missing))
        sym = by_key.get((k, t))
        if not sym:
            tried.add(f"{k}{t}")
            continue
        h = _get(f"{EXP_API}/historical-data", {"symbol": sym, "resolution": "1", "date_format": 1,
                                                "range_from": rf, "range_to": exp.isoformat(), "include_oi": 1})
        c = h.get("candles") or []
        if not c:
            if h.get("s") in ("ok", "no_data"):
                tried.add(f"{k}{t}")
            continue
        cols = h.get("columns") or ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
        df = pd.DataFrame(c, columns=cols[:len(c[0])]).rename(columns={"timestamp": "ts", "open_interest": "oi"})
        if "oi" not in df:
            df["oi"] = 0
        df.insert(0, "symbol", sym)
        df.insert(1, "type", t)
        df.insert(2, "strike", k)
        frames.append(df)
    if frames:
        out = pd.concat([have] + frames, ignore_index=True).drop_duplicates(["symbol", "ts"], keep="last")
        out["ts"] = out["ts"].astype("int64")
        for c in ("volume", "oi"):
            out[c] = out[c].fillna(0).astype("int64")
        for c in ("open", "high", "low", "close"):
            out[c] = out[c].astype("float32")
        out["strike"] = out["strike"].astype("int32")
        out = out.sort_values(["symbol", "ts"])
        tmp = path + ".tmp"
        out.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, path)
        rec["strikes"] = [int(out["strike"].min()), int(out["strike"].max())]
        rec["contracts"] = int(out["symbol"].nunique())
        rec["rows"] = int(len(out))
    rec["tried_empty"] = sorted(tried)
    manifest[exp.isoformat()] = rec
    paths.atomic_write_json(MANIFEST, manifest)
    log(f"top-up {exp} {why}: +{len(frames)} contracts ({len(missing)} missing, {lo_k}-{hi_k})")
    return len(frames), None


def widen_all(extra=500):
    """Add `extra` points of strikes on both sides of every downloaded expiry
    (for far-OTM hedges). Resumable: already-present / known-empty strikes
    are skipped."""
    fyers_auth.login()
    manifest = _load_json(MANIFEST, {})
    todo = sorted((e for e, r in manifest.items() if isinstance(r, dict) and r.get("status") == "done"), reverse=True)
    log(f"==== widen strikes by {extra} pts on {len(todo)} expiries ====")
    for i, e in enumerate(todo, 1):
        rec = _load_json(MANIFEST, {}).get(e) or {}
        lo, hi = rec.get("strikes") or [None, None]
        if lo is None or rec.get("widened", 0) >= extra:
            continue
        t = time.time()
        n, err = ensure_strike_range(e, lo - extra, hi + extra, why=f"widen +/-{extra}")
        m = _load_json(MANIFEST, {})
        if not err:
            m[e]["widened"] = extra
            paths.atomic_write_json(MANIFEST, m)
        log(f"[{i}/{len(todo)}] {e}: +{n} contracts {err or ''} {time.time() - t:.0f}s")
    log("==== widen finished ====")


def run(years=5, test=False):
    fyers_auth.login()
    today = mc.now_ist().date()
    start = today - timedelta(days=int(365.25 * years))
    log(f"==== history download: {start} -> {today} (ATM ±{ATM_SIDE} strikes, 1-min) ====")
    spot = download_spot(start - timedelta(days=10), today)
    log(f"spot on disk: {len(spot):,} candles")
    # the expiry-dates API rejects today as range_to ("Invalid input") -- end at yesterday
    exps = [date.fromisoformat(e) for e in list_expiries(start, today - timedelta(days=1)) if e < today.isoformat()]
    log(f"{len(exps)} expired option expiries in range")
    manifest = _load_json(MANIFEST, {})
    todo = [e for e in sorted(exps, reverse=True) if (manifest.get(e.isoformat()) or {}).get("status") != "done"]
    if test:
        todo = todo[:1]
    log(f"{len(todo)} expiries to fetch" + ("" if test else f" ({len(exps) - len(todo)} already done)"))
    all_exps = sorted(exps)
    for i, exp in enumerate(todo, 1):
        k = all_exps.index(exp)
        prev = all_exps[k - 1] if k > 0 else None
        t = time.time()
        info, err = download_expiry(exp, prev, spot)
        if err:
            manifest[exp.isoformat()] = {"status": "failed", "error": err}
            log(f"[{i}/{len(todo)}] {exp}: FAILED -- {err}")
        else:
            manifest[exp.isoformat()] = info
            log(f"[{i}/{len(todo)}] {exp}: {info['contracts']} contracts ({info['strikes'][0]}-{info['strikes'][1]}), "
                f"{info['rows']:,} rows, {time.time() - t:.0f}s")
        paths.atomic_write_json(MANIFEST, manifest)
    done = sum(1 for v in manifest.values() if v.get("status") == "done")
    size = sum(os.path.getsize(os.path.join(OPT_DIR, f)) for f in os.listdir(OPT_DIR)) if os.path.isdir(OPT_DIR) else 0
    log(f"==== finished: {done} expiries on disk, options data {size / 1e9:.2f} GB ====")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    yrs = 5
    if "--years" in sys.argv:
        yrs = float(sys.argv[sys.argv.index("--years") + 1])
    if "--widen" in sys.argv:
        widen_all(int(sys.argv[sys.argv.index("--widen") + 1]))
    else:
        run(years=yrs, test="--test" in sys.argv)
