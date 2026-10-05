"""Synthetic recorder databases with KNOWN implied vols, spreads and last-trade biases, for the micro_analysis tests."""
import math
import os
from datetime import date, datetime

from collector.config import IST
from collector.store import Store
from optionsengine.bsm import OptionType, bsm_price

R = 0.065
YEAR = 365.0 * 86400.0


def make_db(directory, day=date(2026, 10, 5), cycles=(("11:01", "2026-10-13"),), spot=22500.0, carry=0.003, sigma=0.15, half_spread=0.10, ltp_bias_hs=0.0, skew=1.5, n_strikes=12, step=50,
            dte_days=8.0, volume=1000, feed_age=1.0, ltp_age=3.0, smile=0.0, status="ok"):
    """One row group per (cycle, expiry): flat (or linearly skewed) vol `sigma`, forward = spot*(1+carry), bid/ask = mid -/+ half_spread (rupees), ltp = mid + ltp_bias_hs*half_spread.
    Returns (db_path, truth) where truth has the forward and per-strike true mid prices."""
    s = Store(directory, day, {})
    truth = {}
    grouped = {}
    for hhmm, exp in cycles:                                   # several expiries may share one cycle
        grouped.setdefault(hhmm, []).append(exp)
    for hhmm, exps in grouped.items():
        h, m = map(int, hhmm.split(":"))
        cap = datetime(day.year, day.month, day.day, h, m, 0, tzinfo=IST).timestamp() + skew          # our clock runs `skew` seconds ahead
        exp_ts = cap - skew + dte_days * 86400.0
        T = (exp_ts - (cap - skew)) / YEAR
        F = spot * (1 + carry)
        q = R - math.log(F / spot) / T                                                                      # S e^{(r-q)T} = F
        atm = int(round(spot / step) * step)
        rows = [dict(symbol="NSE:NIFTY50-INDEX", kind="index", ltp=spot, data_source="fyers:ws-full", quote_feed_ts=int(cap) - 1, capture_ts=cap)]
        for exp in exps:
          for off in range(-n_strikes, n_strikes + 1):
            K = atm + off * step
            iv = sigma + smile * math.log(K / F)
            for typ in ("CE", "PE"):
                mid = bsm_price(spot, K, T, R, q, iv, OptionType.CALL if typ == "CE" else OptionType.PUT)
                truth[(hhmm, exp, K, typ)] = dict(mid=mid, iv=iv, T=T, F=F, q=q)
                rows.append(dict(symbol=f"NSE:NIFTY{exp}{K}{typ}", kind="option", expiry_date=exp, expiry_ts=int(exp_ts), dte_days=int(dte_days), strike=float(K), option_type=typ, atm_strike=atm, offset_strikes=off,
                                 ltp=mid + ltp_bias_hs * half_spread, bid=mid - half_spread, ask=mid + half_spread, bid_size=130, ask_size=195, volume=volume, last_traded_qty=65, oi=1000,
                                 quote_feed_ts=int(cap) - int(feed_age), last_trade_ts=int(cap) - int(ltp_age), capture_ts=cap, capture_minus_feed_s=feed_age + skew, capture_minus_last_trade_s=ltp_age + skew,
                                 data_source="fyers:ws-full", spot=spot, flags=""))
        s.write_cycle(dict(cycle_id=f"{day.isoformat()}T{hhmm}", scheduled_ts=cap - skew, capture_start_ts=cap, capture_end_ts=cap + 5, status=status, n_rows=len(rows), spot=spot, atm_strike=atm,
                           india_vix=15.0, future_fp=F, skew_est_s=skew, ws_connected=1), rows)
    path = s.path
    s.close()
    return path, truth
