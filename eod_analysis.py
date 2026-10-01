"""eod_analysis.py -- builds a daily summary time series from data/eod/*.csv
(NSE's own Bhavcopy, one row per NIFTY option contract per day) for
exploratory analysis / ML feature engineering: ATM implied vol, put-call
ratio, OI (in notional terms, not raw lot count -- see below), volume, and
days-to-expiry, one row per trading day.

Two structural-change traps this deliberately avoids (per the earlier
discussion): NIFTY's lot size has changed more than once, and so has which
weekday it expires on. Bhavcopy carries the ACTUAL lot size
(NewBrdLotQty) and actual expiry date (XpryDt) on every single row, so:
- OI is summed in NOTIONAL terms (contracts * lot size * strike), not raw
  lot count -- comparable across a lot-size change instead of jumping
  discontinuously at the boundary.
- "Days to expiry" is computed from the real expiry date each row already
  carries, never from an assumed weekday.
"""
import glob
import os
from datetime import datetime

import pandas as pd

import greeks as greeks_mod
import paths

BASE = paths.BASE_DIR
EOD_DIR = os.path.join(BASE, "data", "eod")
OUT_PATH = os.path.join(BASE, "results", "eod_daily_summary.csv")


def _nearest_expiry_rows(df, spot):
    """Rows for whichever expiry in this day's file is nearest to (but not
    before) the trade date -- the "front week" contract, which is what a
    daily ATM-IV/PCR series conventionally tracks."""
    trad_dt = pd.Timestamp(df["TradDt"].iloc[0])
    expiries = sorted(pd.Timestamp(x) for x in df["XpryDt"].unique() if pd.Timestamp(x) >= trad_dt)
    if not expiries:
        return None, None
    nearest = expiries[0]
    return df[pd.to_datetime(df["XpryDt"]) == nearest], nearest


def _atm_iv(rows, forward, expiry_ts, trad_dt):
    """Averages the IV of the single nearest-to-forward CE and PE strike
    -- simplest, most standard definition of "ATM IV" for a daily series."""
    T = (expiry_ts - trad_dt).days / 365.0
    if T <= 0:
        return None
    ce = rows[rows["OptnTp"] == "CE"].copy()
    pe = rows[rows["OptnTp"] == "PE"].copy()
    if ce.empty or pe.empty:
        return None
    ce["dist"] = (ce["StrkPric"] - forward).abs()
    pe["dist"] = (pe["StrkPric"] - forward).abs()
    ce_row = ce.sort_values("dist").iloc[0]
    pe_row = pe.sort_values("dist").iloc[0]
    ivs = []
    for row, is_call in ((ce_row, True), (pe_row, False)):
        price = row["ClsPric"] if row["ClsPric"] > 0 else row["SttlmPric"]
        iv = greeks_mod.implied_vol_fwd(price, forward, row["StrkPric"], T, greeks_mod.RISK_FREE_RATE, is_call)
        if iv is not None:
            ivs.append(iv)
    return round(sum(ivs) / len(ivs) * 100, 2) if ivs else None


def build_daily_summary():
    files = sorted(glob.glob(os.path.join(EOD_DIR, "*.csv")))
    rows_out = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df = df[df["FinInstrmTp"] == "IDO"]  # options only for this series (futures excluded)
        if df.empty:
            continue
        trad_dt = pd.Timestamp(df["TradDt"].iloc[0])
        spot = float(df["UndrlygPric"].iloc[0])

        near, expiry_ts = _nearest_expiry_rows(df, spot)
        if near is None or near.empty:
            continue

        quotes = []
        for strike, grp in near.groupby("StrkPric"):
            ce_row = grp[grp["OptnTp"] == "CE"]
            pe_row = grp[grp["OptnTp"] == "PE"]
            c = float(ce_row["ClsPric"].iloc[0]) if not ce_row.empty and ce_row["ClsPric"].iloc[0] > 0 else None
            p = float(pe_row["ClsPric"].iloc[0]) if not pe_row.empty and pe_row["ClsPric"].iloc[0] > 0 else None
            quotes.append((strike, c, p))
        forward = greeks_mod.synthetic_forward(quotes) or spot

        atm_iv = _atm_iv(near, forward, expiry_ts, trad_dt)

        ce_oi_notional = (near[near["OptnTp"] == "CE"]["OpnIntrst"] *
                          near[near["OptnTp"] == "CE"]["NewBrdLotQty"] *
                          near[near["OptnTp"] == "CE"]["StrkPric"]).sum()
        pe_oi_notional = (near[near["OptnTp"] == "PE"]["OpnIntrst"] *
                          near[near["OptnTp"] == "PE"]["NewBrdLotQty"] *
                          near[near["OptnTp"] == "PE"]["StrkPric"]).sum()
        pcr = round(pe_oi_notional / ce_oi_notional, 3) if ce_oi_notional else None
        # Standard PCR = put OI / call OI in contracts. The notional one above
        # weights by strike, which biases it low (puts sit at lower strikes).
        ce_oi = near[near["OptnTp"] == "CE"]["OpnIntrst"].sum()
        pe_oi = near[near["OptnTp"] == "PE"]["OpnIntrst"].sum()
        pcr_oi = round(pe_oi / ce_oi, 3) if ce_oi else None

        rows_out.append({
            "date": trad_dt.strftime("%Y-%m-%d"),
            "underlying_close": spot,
            "forward": round(forward, 2),
            "atm_iv_pct": atm_iv,
            "pcr_oi": pcr_oi,
            "pcr_notional": pcr,
            "days_to_expiry": (expiry_ts - trad_dt).days,
            "total_volume": int(df["TtlTradgVol"].sum()),
            "total_oi_notional": round((ce_oi_notional + pe_oi_notional) / 1e7, 2),  # crores
        })

    out = pd.DataFrame(rows_out).sort_values("date")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    return out


_iv_cache = {"mtime": None, "series": None}


def iv_rank(current_iv_pct, lookback_days=252, min_dte=2):
    """IV Rank = where today's ATM IV sits between the past year's low and
    high (0-100); IV Percentile = % of past days with LOWER IV. Uses the
    front-week ATM IV series from NSE bhavcopy (results/eod_daily_summary.csv),
    skipping days < min_dte to expiry -- expiry-eve IVs spike mechanically
    and would distort both numbers."""
    try:
        mtime = os.path.getmtime(OUT_PATH)
    except OSError:
        return {}
    if _iv_cache["mtime"] != mtime:
        df = pd.read_csv(OUT_PATH)
        df = df[(df["days_to_expiry"] >= min_dte) & df["atm_iv_pct"].notna()]
        _iv_cache.update(mtime=mtime, series=df.tail(lookback_days)["atm_iv_pct"].tolist())
    s = _iv_cache["series"] or []
    if len(s) < 20 or current_iv_pct is None:
        return {}
    lo, hi = min(s), max(s)
    return {
        "iv_rank": round((current_iv_pct - lo) / (hi - lo) * 100, 1) if hi > lo else None,
        "iv_percentile": round(sum(v < current_iv_pct for v in s) / len(s) * 100, 1),
        "iv_1y_low": round(lo, 2), "iv_1y_high": round(hi, 2), "iv_history_days": len(s),
    }


if __name__ == "__main__":
    df = build_daily_summary()
    print(f"{len(df)} days summarized -> {OUT_PATH}")
    print(df.tail(10).to_string(index=False))
