"""arrow_history.py -- the evening job's data source on Arrow (iRage), in the
shapes daily_history.py already writes to data/hist1m (same columns, same
Fyers-style symbols). Used when .env has DATA_SOURCE=arrow.

Arrow's candle API serves only currently-listed contracts (no expired-F&O
endpoint) and at most ~1 month of 1-min bars per request, so:
  - every request is chunked into <= 30-day windows;
  - an expiry is "finalised" from the job's own live file, which by the
    evening of expiry day already holds the whole contract life (the job
    re-fetches the live expiry in full every evening). An expiry whose live
    file stops short of expiry day (PC off that evening) is logged as a gap.
"""
import os
from datetime import date, datetime, timedelta

import pandas as pd

import arrow_chain
import history_downloader as hd
import market_calendar as mc

CHUNK_DAYS = 30


def _chunks(start, end):
    d = start
    while d <= end:
        e = min(d + timedelta(days=CHUNK_DAYS - 1), end)
        yield d, e
        d = e + timedelta(days=1)


def candles_1m(fyers_symbol, start, end):
    """[(ts, o, h, l, c, v, oi), ...] for start..end (dates), chunked."""
    rows = []
    for d, e in _chunks(start, end):
        rows += arrow_chain.history_candles(fyers_symbol, d.isoformat(), e.isoformat(), 1)
    return rows


def download_spot(start, end):
    """NIFTY 50 1-min OHLC appended to hist1m/NIFTY50_1m.parquet (same as
    history_downloader.download_spot, from Arrow)."""
    path = hd.SPOT_PATH
    have = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    if not have.empty:
        last = datetime.fromtimestamp(int(have["ts"].max()), mc.IST).date()
        first = datetime.fromtimestamp(int(have["ts"].min()), mc.IST).date()
        if first <= start + timedelta(days=5):
            start = last
    parts = [have] if not have.empty else []
    c = candles_1m(arrow_chain.INDEX_SYMBOL, start, end)
    hd.log(f"spot {start} -> {end}: {len(c)} candles (arrow)")
    if c:
        df = pd.DataFrame(c, columns=["ts", "open", "high", "low", "close", "volume", "oi"]).drop(columns=["volume", "oi"])
        parts.append(df)
    if parts:
        df = pd.concat(parts).drop_duplicates("ts", keep="last").sort_values("ts")
        df["ts"] = df["ts"].astype("int64")
        os.makedirs(hd.OUT, exist_ok=True)
        df.to_parquet(path, compression="zstd", index=False)
        return df
    return have


def list_expiries():
    """Currently listed NIFTY option expiries, 'YYYY-MM-DD' (like fyers_option_symbols.list_expiries)."""
    return [e["date"].isoformat() for e in arrow_chain.instruments()["expiries"]]


def chain_symbols(expiry_iso, strikes):
    """{strike: {'CE': sym, 'PE': sym}} for listed contracts (like fyers_option_symbols.chain_symbols)."""
    inst = arrow_chain.instruments()
    d = date.fromisoformat(expiry_iso)
    out = {}
    for k in strikes:
        for typ in ("CE", "PE"):
            rec = inst["by_key"].get((d, float(k), typ))
            if rec:
                out.setdefault(k, {})[typ] = rec["fyers"]
    return out


def finalize_from_live(m, exp_iso):
    """Manifest entry for an expired week, from the job's own live file.
    Returns (entry, error)."""
    path = os.path.join(hd.OPT_DIR, f"{exp_iso}.parquet")
    ent = dict(m.get(exp_iso) or {})
    if not os.path.exists(path):
        return None, "no live file (Arrow has no expired contracts to fetch it from)"
    df = pd.read_parquet(path, columns=["ts"])
    last = datetime.fromtimestamp(int(df["ts"].max()), mc.IST).date().isoformat()
    if last < exp_iso:
        return None, f"live file ends {last}, before expiry day (gap; Arrow cannot back-fill expired contracts)"
    ent.update(status="done", finalised_from="live", finalised_at=datetime.now().isoformat(timespec="seconds"))
    return ent, None
