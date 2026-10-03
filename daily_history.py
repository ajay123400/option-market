"""daily_history.py -- the evening job that keeps data/hist1m/ current, in
place of the old intraday option-chain recorder (historical_recorder.py).

Run once each trading evening (scheduled task "OptionMarket Daily History",
`OptionMarket.exe --daily-history`). It needs no app open during the day:
everything comes from Fyers' history APIs, so a day the PC was off is filled
the next time it runs.

  1. NIFTY 50 spot: 1-min bars appended up to today.
  2. Expiries that have passed without a final download (or were missed
     entirely): fetched from the expired-F&O API, then widened +/-500 pts --
     the same files history_downloader.py produces.
  3. The CURRENT weekly expiry (still trading): every CE/PE from the week's
     lowest ATM - 1000 to highest ATM + 1000 (ATM +/-10 strikes + 500 for
     hedges), 1-min OHLC + OI over the contract's life, re-fetched in full
     each evening -- so any gap in the week is filled automatically. Its
     manifest entry is status "live" until it expires and step 2 finalises it.

Usage:  python daily_history.py
"""
import os
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd

import fyers_auth
import history_downloader as hd
import market_calendar as mc
import paths

WIDEN = 500
STATUS = os.path.join(hd.OUT, "daily_status.json")  # last run summary (Data Health shows it)
FINAL_LOOKBACK_DAYS = 70  # re-check this far back for expiries still missing a final download


def _manifest():
    return hd._load_json(hd.MANIFEST, {})


def _save(m):
    paths.atomic_write_json(hd.MANIFEST, m)


def finalize_expired(today, spot):
    """Step 2: any expiry in the last FINAL_LOOKBACK_DAYS that is not a
    finished ('done') download -- live ones that have now expired, or weeks
    the job never saw -- is fetched from the expired-F&O API and widened."""
    exps = hd.list_expiries(today - timedelta(days=FINAL_LOOKBACK_DAYS), today - timedelta(days=1))
    exps = [date.fromisoformat(e) for e in exps if e < today.isoformat()]
    allm = sorted(set(_manifest()) | {e.isoformat() for e in exps})
    done = 0
    for exp in exps:
        m = _manifest()
        if (m.get(exp.isoformat()) or {}).get("status") == "done":
            continue
        k = allm.index(exp.isoformat())
        prev = date.fromisoformat(allm[k - 1]) if k > 0 else None
        info, err = hd.download_expiry(exp, prev, spot)
        m = _manifest()
        if err:
            hd.log(f"daily: finalise {exp} FAILED -- {err}")
            continue
        m[exp.isoformat()] = info
        _save(m)
        n, err2 = hd.ensure_strike_range(exp, info["strikes"][0] - WIDEN, info["strikes"][1] + WIDEN, why=f"widen +/-{WIDEN}")
        m = _manifest()
        if not err2:
            m[exp.isoformat()]["widened"] = WIDEN
            _save(m)
        hd.log(f"daily: finalised {exp} -- {info['contracts']} contracts, +{n} widened")
        done += 1
    return done


def update_current(today, spot, cur=None):
    """Step 3: the weekly expiry that is still trading (or the one given)."""
    import fyers_option_symbols as fos
    if cur is None:
        listed = [date.fromisoformat(e) for e in fos.list_expiries("NIFTY")]
        cur = min((e for e in listed if e >= today), default=None)
    if cur is None:
        hd.log("daily: no current NIFTY expiry listed")
        return None
    known = sorted(date.fromisoformat(e) for e in _manifest() if e < cur.isoformat())
    prev = known[-1] if known else cur - timedelta(days=7)
    w0, w1 = prev, min(today, cur)          # the week window: previous expiry day -> today
    t0 = int(datetime.combine(w0, datetime.min.time(), mc.IST).timestamp())
    t1 = int(datetime.combine(w1, datetime.max.time(), mc.IST).timestamp())
    wk = spot[(spot["ts"] >= t0) & (spot["ts"] <= t1)]
    if wk.empty:
        hd.log(f"daily: no NIFTY spot yet for {w0} -> {w1}")
        return None
    lo = int(round(wk["low"].min() / hd.STEP) * hd.STEP) - hd.ATM_SIDE * hd.STEP - WIDEN
    hi = int(round(wk["high"].max() / hd.STEP) * hd.STEP) + hd.ATM_SIDE * hd.STEP + WIDEN
    syms = fos.chain_symbols("NIFTY", cur.isoformat(), list(range(lo, hi + 1, hd.STEP)))
    frames, empty = [], 0
    rf = (cur - timedelta(days=99)).isoformat()
    for k, pair in syms.items():
        for typ in ("CE", "PE"):
            sym = pair.get(typ)
            if not sym:
                continue
            b = hd._get(hd.HIST_API, {"symbol": sym, "resolution": "1", "date_format": "1", "range_from": rf,
                                      "range_to": today.isoformat(), "cont_flag": "1", "oi_flag": "1"})
            c = b.get("candles") or []
            if not c:
                empty += 1
                continue
            df = pd.DataFrame(c, columns=["ts", "open", "high", "low", "close", "volume", "oi"][:len(c[0])])
            if "oi" not in df:
                df["oi"] = 0
            df.insert(0, "symbol", sym)
            df.insert(1, "type", typ)
            df.insert(2, "strike", int(k))
            frames.append(df)
    if not frames:
        hd.log(f"daily: {cur} -- no candles returned ({empty} empty)")
        return None
    path = os.path.join(hd.OPT_DIR, f"{cur.isoformat()}.parquet")
    parts = [pd.read_parquet(path)] if os.path.exists(path) else []
    out = pd.concat(parts + frames, ignore_index=True).drop_duplicates(["symbol", "ts"], keep="last")
    out["ts"] = out["ts"].astype("int64")
    for col in ("volume", "oi"):
        out[col] = out[col].fillna(0).astype("int64")
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype("float32")
    out["strike"] = out["strike"].astype("int32")
    out = out.sort_values(["symbol", "ts"])
    os.makedirs(hd.OPT_DIR, exist_ok=True)
    tmp = path + ".tmp"
    out.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)
    m = _manifest()
    m[cur.isoformat()] = {"status": "live", "strikes": [int(out.strike.min()), int(out.strike.max())],
                          "contracts": int(out.symbol.nunique()), "empty": empty, "rows": int(len(out)),
                          "week": [w0.isoformat(), cur.isoformat()], "widened": WIDEN,
                          "fetched_at": datetime.now().isoformat(timespec="seconds")}
    _save(m)
    hd.log(f"daily: {cur} (live) -- {len(frames)} contracts updated, {lo}-{hi}, data to {today}")
    return cur


def update():
    t = time.time()
    fyers_auth.login()
    today = mc.now_ist().date()
    hd.log(f"==== daily history update {today} ====")
    spot = hd.download_spot(today - timedelta(days=10), today)
    hd.log(f"daily: NIFTY spot on disk to {datetime.fromtimestamp(int(spot['ts'].max()), mc.IST):%Y-%m-%d %H:%M}")
    n = finalize_expired(today, spot)
    cur = update_current(today, spot)
    # on an expiry day the IV rule already uses NEXT week's expiry (1-4 days
    # left), so fetch it today too -- otherwise the day's IV readings only
    # reach the history a day late
    if cur == today:
        try:
            import fyers_option_symbols as fos
            nxt = min((date.fromisoformat(e) for e in fos.list_expiries("NIFTY") if e > today.isoformat()), default=None)
            if nxt:
                update_current(today, spot, cur=nxt)
        except Exception as e:
            hd.log(f"daily: next-expiry fetch failed -- {e}")
    # the Strategy Ideas "5-yr record": re-run weekly so finished weeks join it (~3 min)
    track = os.path.join(paths.BASE_DIR, "results", "template_bt", "summary.parquet")
    try:
        stale = not os.path.exists(track) or time.time() - os.path.getmtime(track) > 6 * 86400
        if stale and n:
            import template_backtest
            template_backtest.run(verbose=False)
            hd.log("daily: 5-year template record refreshed")
    except Exception as e:
        hd.log(f"daily: template record refresh failed -- {e}")
    # the IV rule's history (ATM IV at 09:30 / 12:00 / 14:30 + put skew) for the new days
    try:
        import sell_rules
        added = sell_rules.update_history(progress=hd.log)
        hd.log(f"daily: IV rule history +{added} rows")
    except Exception as e:
        hd.log(f"daily: IV rule history update failed -- {e}")
    status = {"last_run": datetime.now().isoformat(timespec="seconds"), "finalised": n,
              "current": cur.isoformat() if cur else None, "seconds": round(time.time() - t)}
    paths.atomic_write_json(STATUS, status)
    hd.log(f"==== daily history update done in {time.time() - t:.0f}s ====")
    return status


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print(update())
