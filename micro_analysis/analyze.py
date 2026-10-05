"""Row-level and ATM-level metrics from recorded quotes. Pure functions over pandas frames (no file I/O)."""
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from optionsengine.bsm import OptionType, implied_carry_yield
from optionsengine.forward import ForwardStatus

from .config import AnalysisConfig, dte_bucket, ltp_age_bucket, moneyness_bucket, time_bucket
from .ivcalc import atm_iv, parity_forward, solve_iv, time_to_expiry_years, vega_per_vol_point

NAN = float("nan")
EXCLUSION_ORDER = ["not_websocket_row", "no_bid_or_ask", "crossed", "stale_quote", "never_traded_today"]


def exclusion_reason(r, skew: float, cfg: AnalysisConfig):
    """First matching reason for dropping an option row from the statistics (None = eligible). Fixed order; counted in exclusions.csv."""
    if r["data_source"] != "fyers:ws-full":
        return "not_websocket_row"
    bid, ask = r["bid"], r["ask"]
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
        return "no_bid_or_ask"
    if bid > ask:
        return "crossed"
    cmf = r["capture_minus_feed_s"]
    if pd.isna(cmf) or (cmf - skew) > cfg.max_quote_age_s:
        return "stale_quote"
    if not r["volume"] or pd.isna(r["volume"]):
        return "never_traded_today"
    return None


def _kind(t: str) -> OptionType:
    return OptionType.CALL if t == "CE" else OptionType.PUT


def analyze_cycle_expiry(cyc: pd.Series, rows: pd.DataFrame, spot: float, cfg: AnalysisConfig) -> Tuple[List[dict], dict, Counter]:
    """One (cycle, expiry): returns (row records, ATM record, exclusion counts)."""
    skew = 0.0 if pd.isna(cyc["skew_est_s"]) else float(cyc["skew_est_s"])
    first = rows.iloc[0]
    T = time_to_expiry_years(first["expiry_ts"], cyc["capture_start_ts"], skew)
    hhmm = cyc["cycle_id"][-5:]
    base = dict(day=cyc["day"], cycle_id=cyc["cycle_id"], time=hhmm, time_bucket=time_bucket(hhmm, cfg), primary=hhmm in cfg.primary_times, expiry_date=first["expiry_date"],
                dte_days=T * 365.0, dte_bucket=dte_bucket(T * 365.0, cfg), spot=spot)
    excl = Counter()
    elig = []
    for _, r in rows.iterrows():
        why = exclusion_reason(r, skew, cfg)
        if why:
            excl[why] += 1
        else:
            elig.append(r)
    atm_rec = dict(base, n_option_rows=len(rows), n_eligible=len(elig), forward=NAN, forward_status="not_attempted", forward_dispersion=NAN)
    for v in ("mid", "bid", "ask", "ltp"):
        atm_rec[f"atm_iv_{v}"] = NAN
    atm_rec.update(atm_strike_low=NAN, atm_strike_high=NAN)
    if T <= 0 or not elig or not spot:
        return [], atm_rec, excl
    # ---- forward from mid prices (same gate as Stage 2A)
    mids = {(r["strike"], r["option_type"]): (r["bid"] + r["ask"]) / 2 for r in elig}
    pairs = [(k, mids[(k, "CE")], mids[(k, "PE")]) for k in sorted({k for k, _ in mids}) if (k, "CE") in mids and (k, "PE") in mids]
    F, q = None, None
    if len(pairs) >= 2:
        fe = parity_forward(pairs, cfg.rate, T, spot)
        atm_rec.update(forward_status=fe.status.value, forward_dispersion=fe.dispersion if fe.dispersion is not None else NAN)
        if fe.status is ForwardStatus.OK:
            F = fe.forward
            q = implied_carry_yield(spot, F, T, cfg.rate)
            atm_rec["forward"] = F
    else:
        atm_rec["forward_status"] = "fewer_than_2_pairs"
    recs, pts = [], {v: {} for v in ("mid", "bid", "ask", "ltp")}
    iv_at_strike = {}
    for r in elig:
        k, typ = float(r["strike"]), r["option_type"]
        bid, ask, ltp = float(r["bid"]), float(r["ask"]), r["ltp"]
        mid = (bid + ask) / 2
        ltp_age = r["capture_minus_last_trade_s"]
        ltp_ok = (not pd.isna(ltp)) and ltp > 0 and (not pd.isna(ltp_age)) and (ltp_age - skew) <= cfg.max_quote_age_s
        rec = dict(base, strike=k, option_type=typ, offset_strikes=int(r["offset_strikes"]), moneyness_bucket=moneyness_bucket(r["offset_strikes"], cfg), bid=bid, ask=ask, mid=mid,
                   spread_rs=ask - bid, half_spread_rs=(ask - bid) / 2, spread_pct=100.0 * (ask - bid) / mid, bid_size=r["bid_size"], ask_size=r["ask_size"], volume=r["volume"], ltp=ltp,
                   ltp_age_s=(ltp_age - skew) if not pd.isna(ltp_age) else NAN, ltp_fresh=bool(ltp_ok), oi=r["oi"])
        d = ltp - mid if not pd.isna(ltp) else NAN
        rec["ltp_minus_mid_rs"] = d
        rec["ltp_minus_mid_halfspreads"] = d / rec["half_spread_rs"] if rec["half_spread_rs"] > 0 and not pd.isna(d) else NAN
        rec["ltp_at"] = ("ask" if ltp >= ask else "bid" if ltp <= bid else "inside") if not pd.isna(ltp) else None
        rec["ltp_age_bucket"] = ltp_age_bucket(rec["ltp_age_s"], cfg) if ltp_ok else None
        for v in ("mid", "bid", "ask", "ltp"):
            rec[f"iv_{v}"] = NAN
            rec[f"iv_{v}_usable"] = False
        rec["otm_side"] = False
        if F is not None:
            otm = (typ == "CE") if k >= F else (typ == "PE")
            rec["otm_side"], rec["log_moneyness"] = bool(otm), math.log(k / F)
            if otm:
                kind = _kind(typ)
                for v, price, allowed in (("mid", mid, True), ("bid", bid, True), ("ask", ask, True), ("ltp", ltp, ltp_ok)):
                    if not allowed:
                        continue
                    iv, ok = solve_iv(price, spot, k, T, cfg.rate, q, kind)
                    if iv is not None:
                        rec[f"iv_{v}"], rec[f"iv_{v}_usable"] = iv, ok
                    pts[v][k] = (math.log(k / F), iv, ok)
                if rec["iv_mid"] == rec["iv_mid"]:
                    iv_at_strike[k] = rec["iv_mid"]
        recs.append(rec)
    # ---- vega (strike-level, from the OTM mid IV) and the IV-based half-spreads
    for rec in recs:
        iv = iv_at_strike.get(rec["strike"])
        rec["vega_rs_per_volpt"] = vega_per_vol_point(F, rec["strike"], T, cfg.rate, iv) if (F is not None and iv) else NAN
        rec["half_spread_vol_pts_vega"] = rec["half_spread_rs"] / rec["vega_rs_per_volpt"] if rec["vega_rs_per_volpt"] == rec["vega_rs_per_volpt"] and rec["vega_rs_per_volpt"] > 0 else NAN
        both = rec["iv_bid_usable"] and rec["iv_ask_usable"]
        rec["iv_spread_vol_pts"] = 100.0 * (rec["iv_ask"] - rec["iv_bid"]) if both else NAN
        rec["half_spread_vol_pts_iv"] = rec["iv_spread_vol_pts"] / 2 if both else NAN
        rec["ltp_minus_mid_vol_pts"] = 100.0 * (rec["iv_ltp"] - rec["iv_mid"]) if rec["iv_ltp_usable"] and rec["iv_mid_usable"] else NAN
    # ---- ATM per variant
    if F is not None:
        for v in ("mid", "bid", "ask", "ltp"):
            iv, lo, hi = atm_iv(pts[v], cfg.atm_max_bracket_pts)
            if iv is not None:
                atm_rec[f"atm_iv_{v}"] = iv
                if v == "mid":
                    atm_rec["atm_strike_low"], atm_rec["atm_strike_high"] = lo, hi
        m, b, a, l = (atm_rec[f"atm_iv_{v}"] for v in ("mid", "bid", "ask", "ltp"))
        atm_rec["atm_spread_vol_pts"] = 100.0 * (a - b) if a == a and b == b else NAN
        atm_rec["atm_half_spread_vol_pts"] = atm_rec["atm_spread_vol_pts"] / 2
        atm_rec["atm_ltp_minus_mid_vol_pts"] = 100.0 * (l - m) if l == l and m == m else NAN
        for f in cfg.cross_fractions:
            sell = m - f * (m - b) if m == m and b == b else NAN              # IV obtained when SELLING and crossing a fraction f of the half-spread
            buy = m + f * (a - m) if m == m and a == a else NAN
            atm_rec[f"sell_iv_cross{f:g}"], atm_rec[f"buy_iv_cross{f:g}"] = sell, buy
            atm_rec[f"sell_minus_ltp_vol_pts_cross{f:g}"] = 100.0 * (sell - l) if sell == sell and l == l else NAN
    return recs, atm_rec, excl


def analyze_frames(cycles: pd.DataFrame, quotes: pd.DataFrame, cfg: AnalysisConfig = AnalysisConfig()) -> Dict[str, pd.DataFrame]:
    """All recorded cycles -> {'rows', 'atm', 'exclusions'}. Cycles that are not ok/partial-with-quotes are skipped (missed cycles carry no quotes)."""
    opt = quotes[quotes["kind"] == "option"]
    idx = quotes[quotes["kind"] == "index"].set_index(["day", "cycle_id"])["ltp"].to_dict()
    fut = quotes[quotes["kind"] == "future"].set_index(["day", "cycle_id"])["ltp"].to_dict()
    cyc_by = {(r["day"], r["cycle_id"]): r for _, r in cycles.iterrows()}
    row_recs, atm_recs, excl_rows = [], [], []
    for (day, cid, exp), g in opt.groupby(["day", "cycle_id", "expiry_date"], sort=True):
        cyc = cyc_by.get((day, cid))
        if cyc is None or cyc["status"] == "missed":
            continue
        spot = idx.get((day, cid))
        if spot is None or pd.isna(spot):
            spot = cyc["spot"]
        recs, atm, excl = analyze_cycle_expiry(cyc, g, spot, cfg)
        row_recs += recs
        atm_recs.append(dict(atm, fut_ltp=fut.get((day, cid), NAN), india_vix=cyc["india_vix"], future_fp=cyc["future_fp"], skew_est_s=cyc["skew_est_s"], cycle_status=cyc["status"]))
        for why, n in excl.items():
            excl_rows.append(dict(day=day, cycle_id=cid, expiry_date=exp, reason=why, n=n))
    rows = pd.DataFrame(row_recs)
    atm = pd.DataFrame(atm_recs)
    ex = pd.DataFrame(excl_rows, columns=["day", "cycle_id", "expiry_date", "reason", "n"])
    return dict(rows=rows, atm=atm, exclusions=ex)
