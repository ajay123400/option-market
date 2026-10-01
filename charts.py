"""charts.py -- candle data for the Charts tab, read from the downloaded 1-min
history (data/hist1m/, see history_downloader.py / simulator.py).

Two views:
  single    one contract (strike + CE/PE): OHLC, volume, OI
  spot      NIFTY 50 itself (for drawing support / resistance)
  straddle  CE(ce_strike) + PE(pe_strike) summed minute by minute -- a
            straddle when both strikes match, a strangle otherwise. This is
            the premium a seller collects, so the chart shows its decay.

Intervals: 1/3/5/15/30/60 minutes (resampled from 1-min bars anchored at
09:15 each day) or "D" (one candle per day). Times go out as epoch seconds
shifted to IST, so the chart library (which renders UTC) shows IST clock times.
"""
import pandas as pd

import simulator

INTERVALS = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "D": None}
IST_SHIFT = 19800  # +05:30


def meta(expiry):
    """Strikes on disk, trading days and a sensible default strike for one expiry."""
    df = simulator._load_options(expiry)
    days = sorted(d.isoformat() for d in df["day"].unique())
    strikes = sorted(int(k) for k in df["strike"].unique())
    week_start = simulator.week_start(expiry)
    # default strike: ATM at the first candle of the expiry week
    spot = simulator._load_spot()
    atm = None
    if spot is not None:
        s = spot.loc[week_start:f"{expiry} 23:59"]
        if not s.empty:
            ref = float(s["close"].iloc[0])
            atm = min(strikes, key=lambda k: abs(k - ref)) if strikes else None
    default_from = next((d for d in days if d >= week_start), days[0] if days else None)
    return {"strikes": strikes, "days": days, "week_start": week_start, "atm": atm,
            "default_from": default_from, "lot_size": simulator.lot_size_for(expiry)}


def _series(df, strike, typ):
    x = df[(df["strike"] == strike) & (df["type"] == typ)]
    return x.sort_values("ts").set_index("date")[["open", "high", "low", "close", "volume", "oi"]]


def _resample(x, interval):
    """1-min OHLCV/OI frame (DatetimeIndex, IST) -> candles of `interval`."""
    if x.empty:
        return x
    mins = INTERVALS[interval]
    if mins == 1:
        return x
    if mins is None:
        lab = x.index.normalize()
    else:
        lab = simulator._bucket_labels(x.index, mins)
    g = x.groupby(lab)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(), "oi": g["oi"].last()})
    if "ce" in x:
        out["ce"] = g["ce"].last()
        out["pe"] = g["pe"].last()
    return out


def _straddle(df, ce_strike, pe_strike):
    """Minute-by-minute CE + PE. Each leg's last price is carried forward over
    minutes it didn't trade, so the sum is always of two live quotes; a
    minute's candle = open (sum of opens) -> close (sum of closes)."""
    ce, pe = _series(df, ce_strike, "CE"), _series(df, pe_strike, "PE")
    if ce.empty or pe.empty:
        return pd.DataFrame()
    j = ce.join(pe, how="outer", lsuffix="_c", rsuffix="_p").sort_index()
    for side in ("_c", "_p"):
        j["close" + side] = j["close" + side].ffill()
        j["open" + side] = j["open" + side].fillna(j["close" + side])
        j["oi" + side] = j["oi" + side].ffill()
        j["volume" + side] = j["volume" + side].fillna(0)
    j = j.dropna(subset=["close_c", "close_p"])
    o = j["open_c"] + j["open_p"]
    c = j["close_c"] + j["close_p"]
    return pd.DataFrame({"open": o, "high": pd.concat([o, c], axis=1).max(axis=1),
                         "low": pd.concat([o, c], axis=1).min(axis=1), "close": c,
                         "volume": j["volume_c"] + j["volume_p"], "oi": j["oi_c"].fillna(0) + j["oi_p"].fillna(0),
                         "ce": j["close_c"], "pe": j["close_p"]})


def _spot_frame(date_from, date_to, expiry):
    """NIFTY 50 1-min bars (09:15-15:30) for the chosen days, shaped like an option series."""
    s = simulator._load_spot()
    if s is None:
        return pd.DataFrame()
    d0 = date_from or expiry
    d1 = date_to or expiry
    x = s.loc[str(d0):f"{d1} 23:59"]
    hm = x.index.hour * 60 + x.index.minute
    x = x[(hm >= 555) & (hm <= 930)][["open", "high", "low", "close"]].copy()
    x["volume"] = 0
    x["oi"] = float("nan")
    return x


def data(expiry, interval="5", mode="single", strike=None, typ="CE", ce_strike=None, pe_strike=None,
         date_from=None, date_to=None):
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {list(INTERVALS)}")
    df = simulator._load_options(expiry)
    if date_from:
        df = df[df["day"] >= pd.Timestamp(date_from).date()]
    if date_to:
        df = df[df["day"] <= pd.Timestamp(date_to).date()]
    df = df[df["date"].dt.time <= simulator.MARKET_CLOSE]
    if mode == "spot":
        x = _spot_frame(date_from, date_to, expiry)
        title = "NIFTY 50"
    elif mode == "straddle":
        x = _straddle(df, int(ce_strike), int(pe_strike))
        title = f"{int(ce_strike)} CE + {int(pe_strike)} PE" + (" (straddle)" if int(ce_strike) == int(pe_strike) else " (strangle)")
    else:
        x = _series(df, int(strike), typ)
        title = f"{int(strike)} {typ}"
    x = _resample(x, interval)
    if x.empty:
        return {"title": title, "candles": [], "stats": None}

    t = (x.index.tz_convert("UTC").astype("int64") // 10**9) + IST_SHIFT
    candles = []
    for ts, r in zip(t, x.itertuples()):
        row = {"time": int(ts), "open": round(float(r.open), 2), "high": round(float(r.high), 2),
               "low": round(float(r.low), 2), "close": round(float(r.close), 2),
               "volume": int(r.volume), "oi": int(r.oi) if pd.notna(r.oi) else None}
        if mode == "straddle":
            row["ce"] = round(float(r.ce), 2)
            row["pe"] = round(float(r.pe), 2)
        candles.append(row)

    # NIFTY spot on the same buckets (for the legend and an optional line)
    spot = simulator._load_spot()
    spot_line = []
    if spot is not None and mode != "spot":
        s = spot.loc[x.index[0].strftime("%Y-%m-%d"):x.index[-1].strftime("%Y-%m-%d") + " 23:59"]
        if not s.empty:
            mins = INTERVALS[interval]
            lab = s.index.normalize() if mins is None else (s.index if mins == 1 else simulator._bucket_labels(s.index, mins))
            sc = s["close"].groupby(lab).last()
            st = (sc.index.tz_convert("UTC").astype("int64") // 10**9) + IST_SHIFT
            spot_line = [{"time": int(a), "value": round(float(b), 2)} for a, b in zip(st, sc.values)]

    first, last = candles[0], candles[-1]
    stats = {"first_open": first["open"], "last_close": last["close"],
             "change": round(last["close"] - first["open"], 2),
             "change_pct": round((last["close"] - first["open"]) / first["open"] * 100, 1) if first["open"] else None,
             "high": max(c["high"] for c in candles), "low": min(c["low"] for c in candles),
             "volume": sum(c["volume"] for c in candles),
             "oi_last": last["oi"], "oi_max": max((c["oi"] or 0) for c in candles),
             "from": x.index[0].strftime("%d %b %Y"), "to": x.index[-1].strftime("%d %b %Y"),
             "bars": len(candles)}
    return {"title": title, "candles": candles, "spot": spot_line, "stats": stats,
            "lot_size": simulator.lot_size_for(expiry)}
