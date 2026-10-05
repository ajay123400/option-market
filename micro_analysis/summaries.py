"""Grouped descriptive summaries (medians, quantiles, shares) of the row-level and ATM-level metrics. Descriptive only."""
import numpy as np
import pandas as pd

from .config import AnalysisConfig


def _q(s, p):
    s = s.dropna()
    return float(np.quantile(s, p)) if len(s) else np.nan


def _med(s):
    s = s.dropna()
    return float(np.median(s)) if len(s) else np.nan


def liquidity_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out = []
    for key, g in rows.groupby(["dte_bucket", "moneyness_bucket", "time_bucket"], sort=True):
        out.append(dict(dte_bucket=key[0], moneyness_bucket=key[1], time_bucket=key[2], n_rows=len(g), n_days=g["day"].nunique(), n_cycles=g["cycle_id"].nunique(),
                        spread_rs_median=_med(g["spread_rs"]), half_spread_rs_median=_med(g["half_spread_rs"]), spread_pct_median=_med(g["spread_pct"]), spread_pct_p90=_q(g["spread_pct"], 0.9),
                        bid_size_median=_med(g["bid_size"]), ask_size_median=_med(g["ask_size"]), half_spread_vol_pts_vega_median=_med(g["half_spread_vol_pts_vega"]),
                        half_spread_vol_pts_iv_median=_med(g["half_spread_vol_pts_iv"]), n_with_iv_spread=int(g["half_spread_vol_pts_iv"].notna().sum()),
                        share_ltp_at_ask=float((g["ltp_at"] == "ask").mean()), share_ltp_at_bid=float((g["ltp_at"] == "bid").mean()), share_ltp_inside=float((g["ltp_at"] == "inside").mean())))
    return pd.DataFrame(out)


def last_trade_bias(rows: pd.DataFrame) -> pd.DataFrame:
    """Signed last-trade-minus-mid, using only rows whose last trade is fresh (<= max age). In half-spreads, rupees and (where both IVs exist) vol points."""
    g0 = rows[rows["ltp_fresh"] & rows["ltp_age_bucket"].notna()] if not rows.empty else rows
    out = []
    for key, g in g0.groupby(["dte_bucket", "moneyness_bucket", "ltp_age_bucket"], sort=True):
        h = g["ltp_minus_mid_halfspreads"]
        out.append(dict(dte_bucket=key[0], moneyness_bucket=key[1], ltp_age_bucket=key[2], n_rows=len(g), n_days=g["day"].nunique(), mean_halfspreads=float(h.mean()), median_halfspreads=_med(h),
                        p10_halfspreads=_q(h, 0.1), p90_halfspreads=_q(h, 0.9), mean_rs=float(g["ltp_minus_mid_rs"].mean()), median_rs=_med(g["ltp_minus_mid_rs"]),
                        median_vol_pts=_med(g["ltp_minus_mid_vol_pts"]), mean_vol_pts=float(g["ltp_minus_mid_vol_pts"].mean()), n_with_vol_pts=int(g["ltp_minus_mid_vol_pts"].notna().sum()),
                        share_at_ask=float((g["ltp_at"] == "ask").mean()), share_at_bid=float((g["ltp_at"] == "bid").mean())))
    return pd.DataFrame(out)


def last_trade_bias_by_day(rows: pd.DataFrame, max_abs_offset: int = 2) -> pd.DataFrame:
    """Day-by-day mean/median of the signed bias (in half-spreads) near the money: shows whether any average is stable across days or driven by one."""
    if rows.empty:
        return pd.DataFrame()
    g0 = rows[rows["ltp_fresh"] & (rows["offset_strikes"].abs() <= max_abs_offset)]
    out = []
    for key, g in g0.groupby(["day", "dte_bucket"], sort=True):
        out.append(dict(day=key[0], dte_bucket=key[1], n_rows=len(g), mean_halfspreads=float(g["ltp_minus_mid_halfspreads"].mean()), median_halfspreads=_med(g["ltp_minus_mid_halfspreads"]),
                        mean_rs=float(g["ltp_minus_mid_rs"].mean()), share_at_ask=float((g["ltp_at"] == "ask").mean()), share_at_bid=float((g["ltp_at"] == "bid").mean())))
    return pd.DataFrame(out)


def cost_scenarios(atm: pd.DataFrame, cfg: AnalysisConfig = AnalysisConfig()) -> pd.DataFrame:
    """Executable-IV SCENARIOS at the ATM: what ATM IV a seller obtains if he crosses a fraction f of the half-spread, relative to the last-trade ATM IV (the quantity the historical study used).
    Scenario arithmetic on recorded quotes: not a forecast, not a trade rule."""
    a = atm[atm["atm_iv_mid"].notna() & atm["atm_iv_bid"].notna() & atm["atm_iv_ask"].notna()] if not atm.empty else atm
    out = []
    for key, g in a.groupby(["dte_bucket", "primary"], sort=True):
        rec = dict(dte_bucket=key[0], primary_instant=bool(key[1]), n_cycle_expiries=len(g), n_days=g["day"].nunique(), atm_iv_mid_median=_med(g["atm_iv_mid"]) * 100, atm_spread_vol_pts_median=_med(g["atm_spread_vol_pts"]),
                   atm_half_spread_vol_pts_median=_med(g["atm_half_spread_vol_pts"]), atm_half_spread_vol_pts_p90=_q(g["atm_half_spread_vol_pts"], 0.9), ltp_minus_mid_vol_pts_median=_med(g["atm_ltp_minus_mid_vol_pts"]),
                   n_with_ltp_iv=int(g["atm_ltp_minus_mid_vol_pts"].notna().sum()))
        for f in cfg.cross_fractions:
            rec[f"sell_minus_ltp_vol_pts_cross{f:g}_median"] = _med(g[f"sell_minus_ltp_vol_pts_cross{f:g}"])
            rec[f"sell_minus_mid_vol_pts_cross{f:g}_median"] = _med(100.0 * (g[f"sell_iv_cross{f:g}"] - g["atm_iv_mid"]))
        out.append(rec)
    return pd.DataFrame(out)


def primary_instants(atm: pd.DataFrame) -> pd.DataFrame:
    """The 10:01 / 13:01 / 15:01 cycles: mid vs last-trade ATM IV at exactly the instants the historical Stage 2A snapshots used."""
    cols = ["day", "time", "expiry_date", "dte_days", "dte_bucket", "spot", "forward", "forward_status", "atm_iv_mid", "atm_iv_bid", "atm_iv_ask", "atm_iv_ltp", "atm_spread_vol_pts", "atm_ltp_minus_mid_vol_pts",
            "atm_strike_low", "atm_strike_high", "n_eligible", "india_vix", "skew_est_s"]
    return atm[atm["primary"]][cols].reset_index(drop=True) if not atm.empty else pd.DataFrame(columns=cols)


def exclusion_summary(ex: pd.DataFrame, rows: pd.DataFrame, atm: pd.DataFrame) -> pd.DataFrame:
    if ex.empty:
        return pd.DataFrame(dict(reason=[], n=[]))
    s = ex.groupby("reason")["n"].sum().reset_index()
    total = int(atm["n_option_rows"].sum()) if not atm.empty else 0
    s["share_of_option_rows"] = s["n"] / total if total else np.nan
    return s
