"""Phase A Step 3 (micro_analysis): synthetic recorder databases with known truths; consistency with the Stage 2A research engine; summaries by hand; CLI. Fully offline."""
import hashlib
import json
import math
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest
import scipy.optimize as so

from micro_analysis import analyze as AN, config as C, ivcalc as IV, loader as LD, run as RUN, summaries as SM
from optionsengine.bsm import OptionType, bsm_price, implied_carry_yield
from optionsengine.surface import OptionQuote, build_smile
from tests.micro_fakes import R, YEAR, make_db

CFG = C.AnalysisConfig()


def analyzed(tmp_path, **kw):
    p, truth = make_db(str(tmp_path), **kw)
    cyc, q = LD.load_db(p)
    return AN.analyze_frames(cyc, q, CFG), truth, p


def black76_iv(price, F, K, T, r, call):
    """Independent solver (scipy brentq on Black-76) used to cross-check the engine's BSM-with-carry solver."""
    def f(s):
        d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
        d2 = d1 - s * math.sqrt(T)
        n = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
        df = math.exp(-r * T)
        return (df * (F * n(d1) - K * n(d2)) if call else df * (K * n(-d2) - F * n(-d1))) - price
    return so.brentq(f, 1e-4, 5.0, xtol=1e-14)


# ================================================================================================ config buckets
def test_dte_buckets_match_the_research_edges():
    f = C.dte_bucket
    assert [f(x) for x in (0.2, 1.0, 1.0001, 3.0, 3.01, 7.0, 7.5, 14.0, 14.01, 20)] == ["<=1d", "<=1d", "1-3d", "1-3d", "3-7d", "3-7d", "7-14d", "7-14d", ">14d", ">14d"]
    from optionsengine.research import build_iv_rv
    for x in (0.5, 1.0, 1.2, 3.0, 4.0, 7.0, 9.0, 14.0, 14.5):
        assert f(x) == build_iv_rv.dte_bucket(x), x


def test_moneyness_time_and_age_buckets():
    assert [C.moneyness_bucket(o) for o in (0, 1, -2, 3, -6, 7, 12, 13)] == ["ATM", "1-2", "1-2", "3-6", "3-6", "7-12", "7-12", ">12"]
    assert [C.time_bucket(t) for t in ("09:16", "10:29", "10:30", "12:59", "13:00", "14:29", "14:30", "15:36")] == ["open", "open", "midday", "midday", "afternoon", "afternoon", "close", "close"]
    assert [C.ltp_age_bucket(a) for a in (0, 10, 10.1, 60, 61, 300, 301)] == ["<=10s", "<=10s", "10-60s", "10-60s", "60-300s", "60-300s", ">300s"]


# ================================================================================================ ivcalc
def test_time_to_expiry_and_skew_adjustment():
    exp = 1_800_000_000
    assert IV.time_to_expiry_years(exp, exp - 86400, 0.0) == pytest.approx(1 / 365)
    assert IV.time_to_expiry_years(exp, exp - 86400 + 1.5, 1.5) == pytest.approx(1 / 365)               # capture is 1.5 s ahead of true time
    assert IV.time_to_expiry_years(exp, exp - 86400, None) == pytest.approx(1 / 365) and IV.time_to_expiry_years(exp, exp + 10, 0) < 0


def test_vega_matches_a_finite_difference():
    S, K, T, sig = 22500.0, 22500.0, 8 / 365, 0.15
    F = S * math.exp(0.003)
    q = R - math.log(F / S) / T
    v = IV.vega_per_vol_point(F, K, T, R, sig)
    fd = (bsm_price(S, K, T, R, q, sig + 0.0005, OptionType.CALL) - bsm_price(S, K, T, R, q, sig - 0.0005, OptionType.CALL)) / 0.1      # per vol point (0.001 sigma = 0.1 pt)
    assert v == pytest.approx(fd, rel=1e-6) and math.isnan(IV.vega_per_vol_point(F, K, 0, R, sig)) and math.isnan(IV.vega_per_vol_point(F, K, T, R, 0))


def test_atm_interpolation_by_hand_and_failure_modes():
    pts = {22500.0: (-0.002, 0.10, True), 22550.0: (0.002, 0.20, True), 22450.0: (-0.006, 0.0, True)}
    iv, lo, hi = IV.atm_iv(pts)
    assert iv == pytest.approx(0.15) and (lo, hi) == (22500.0, 22550.0)
    assert IV.atm_iv({22500.0: (-0.002, 0.1, True), 22550.0: (0.002, 0.2, False)}) == (None, None, None)          # an unusable side
    assert IV.atm_iv({22500.0: (-0.002, 0.1, True), 22650.0: (0.006, 0.2, True)}) == (None, None, None)          # bracket 150 > 100 pts
    assert IV.atm_iv({22500.0: (-0.002, 0.1, True)}) == (None, None, None) and IV.atm_iv({}) == (None, None, None)
    assert IV.atm_iv({22500.0: (-0.002, None, True), 22550.0: (0.002, 0.2, True)}) == (None, None, None)
    assert IV.atm_iv({22500.0: (-0.002, 0.1, True), 22600.0: (0.002, 0.2, True)}, 100.0)[0] == pytest.approx(0.15) and IV.atm_iv({22500.0: (-0.002, 0.1, True), 22600.01: (0.002, 0.2, True)}, 100.0)[0] is None      # 100 pts allowed, 100.01 not
    iv0, lo0, hi0 = IV.atm_iv({22450.0: (-0.002, 0.30, True), 22500.0: (0.0, 0.10, True), 22550.0: (0.002, 0.20, True)})
    assert (iv0, lo0, hi0) == (pytest.approx(0.10), 22450.0, 22500.0)                                          # a strike exactly at the forward belongs to the upper side
    iv2, _, _ = IV.atm_iv({22500.0: (-0.003, 0.10, True), 22550.0: (0.001, 0.20, True)})
    assert iv2 == pytest.approx(0.10 + 0.75 * 0.10)                                                          # forward nearer the upper strike


def test_solve_iv_guards():
    S = K = 22500.0
    T = 8 / 365
    assert IV.solve_iv(None, S, K, T, R, 0.0, OptionType.CALL) == (None, False) and IV.solve_iv(0.0, S, K, T, R, 0.0, OptionType.CALL) == (None, False)
    assert IV.solve_iv(1e-9, S, K, T, R, 0.0, OptionType.CALL)[1] is False                                  # below the tick resolution: never "usable"
    iv_t, ok_t = IV.solve_iv(0.05, 22500.0, 22900.0, 0.3 / 365, R, 0.0, OptionType.CALL)                     # converges, but a 5-paise price on this contract is tick-resolution-limited
    assert iv_t == pytest.approx(0.204, abs=1e-3) and ok_t is False
    iv, ok = IV.solve_iv(bsm_price(S, K, T, R, 0.0, 0.2, OptionType.CALL), S, K, T, R, 0.0, OptionType.CALL)
    assert iv == pytest.approx(0.2, abs=1e-8) and ok


# ================================================================================================ row eligibility
def test_exclusion_reasons_and_order():
    base = dict(data_source="fyers:ws-full", bid=99.9, ask=100.1, capture_minus_feed_s=2.0, volume=10)
    f = lambda **k: AN.exclusion_reason(pd.Series(dict(base, **k)), 1.5, CFG)
    assert f(bid=100.1, ask=100.1) is None                                          # a locked quote (bid == ask) is eligible; only a crossed one is dropped
    assert f() is None and f(data_source="fyers:options-chain-v3") == "not_websocket_row" and f(bid=0) == "no_bid_or_ask" and f(ask=np.nan) == "no_bid_or_ask" and f(bid=100.2) == "crossed"
    assert f(capture_minus_feed_s=301.6) == "stale_quote" and f(capture_minus_feed_s=301.5) is None and f(capture_minus_feed_s=np.nan) == "stale_quote" and f(volume=0) == "never_traded_today" and f(volume=np.nan) == "never_traded_today"
    assert f(data_source="x", bid=0, volume=0) == "not_websocket_row" and f(bid=0, volume=0) == "no_bid_or_ask"            # first matching reason wins


# ================================================================================================ known-truth recovery
def test_recovers_forward_iv_spread_and_vol_point_half_spread(tmp_path):
    out, truth, _ = analyzed(tmp_path, sigma=0.15, half_spread=0.10)
    atm, rows = out["atm"].iloc[0], out["rows"]
    t = next(iter(truth.values()))
    assert atm["forward_status"] == "ok" and atm["forward"] == pytest.approx(t["F"], rel=1e-9) and atm["dte_bucket"] == "7-14d" and atm["dte_days"] == pytest.approx(8.0) and atm["n_eligible"] == 50
    assert atm["atm_iv_mid"] == pytest.approx(0.15, abs=1e-8) and atm["atm_strike_low"] == 22550.0 and atm["atm_strike_high"] == 22600.0
    # independent re-solve of the bid / ask ATM IVs (scipy, Black-76) at the two bracketing strikes
    F, T = t["F"], t["T"]
    ivs = {}
    for K, typ in ((22550.0, "CE"), (22600.0, "CE")):
        mid = truth[("11:01", "2026-10-13", K, typ)]["mid"]
        ivs[K] = (black76_iv(mid - 0.10, F, K, T, R, True), black76_iv(mid + 0.10, F, K, T, R, True))
    x_lo, x_hi = math.log(22550.0 / F), math.log(22600.0 / F)
    w = (0 - x_lo) / (x_hi - x_lo)
    exp_bid = ivs[22550.0][0] + w * (ivs[22600.0][0] - ivs[22550.0][0])
    exp_ask = ivs[22550.0][1] + w * (ivs[22600.0][1] - ivs[22550.0][1])
    assert atm["atm_iv_bid"] == pytest.approx(exp_bid, abs=1e-7) and atm["atm_iv_ask"] == pytest.approx(exp_ask, abs=1e-7)
    assert atm["atm_spread_vol_pts"] == pytest.approx(100 * (exp_ask - exp_bid), abs=1e-5) and atm["atm_half_spread_vol_pts"] == pytest.approx(atm["atm_spread_vol_pts"] / 2)
    assert atm["sell_iv_cross1"] == pytest.approx(atm["atm_iv_bid"]) and atm["buy_iv_cross1"] == pytest.approx(atm["atm_iv_ask"]) and atm["sell_iv_cross0"] == pytest.approx(atm["atm_iv_mid"])
    assert atm["sell_iv_cross0.5"] == pytest.approx(atm["atm_iv_mid"] - 0.5 * (atm["atm_iv_mid"] - atm["atm_iv_bid"])) and atm["buy_iv_cross0.5"] == pytest.approx(atm["atm_iv_mid"] + 0.5 * (atm["atm_iv_ask"] - atm["atm_iv_mid"]))
    assert atm["sell_minus_ltp_vol_pts_cross1"] == pytest.approx(100 * (atm["atm_iv_bid"] - atm["atm_iv_ltp"])) and atm["sell_minus_ltp_vol_pts_cross1"] < 0
    # near the money: the vega-based and the IV-based half-spreads agree
    near = rows[(rows.offset_strikes.abs() <= 2) & rows.otm_side]
    assert (near.iv_bid < near.iv_mid).all() and (near.iv_mid < near.iv_ask).all() and np.allclose(near.half_spread_vol_pts_vega, near.half_spread_vol_pts_iv, rtol=0.03)
    r0 = rows[(rows.strike == 22550.0) & (rows.option_type == "CE")].iloc[0]
    assert r0["mid"] == pytest.approx(truth[("11:01", "2026-10-13", 22550.0, "CE")]["mid"]) and r0["spread_rs"] == pytest.approx(0.2) and r0["half_spread_rs"] == pytest.approx(0.1)
    assert r0["spread_pct"] == pytest.approx(100 * 0.2 / r0["mid"]) and r0["vega_rs_per_volpt"] == pytest.approx(IV.vega_per_vol_point(F, 22550.0, T, R, 0.15))
    assert r0["half_spread_vol_pts_vega"] == pytest.approx(0.1 / r0["vega_rs_per_volpt"]) and r0["offset_strikes"] == 1 and r0["moneyness_bucket"] == "1-2"


def test_skewed_surface_atm_is_the_interpolated_value(tmp_path):
    out, truth, _ = analyzed(tmp_path, sigma=0.15, smile=-0.8)
    t = next(iter(truth.values()))
    atm = out["atm"].iloc[0]
    k1, k2 = 22550.0, 22600.0
    i1, i2 = 0.15 - 0.8 * math.log(k1 / t["F"]), 0.15 - 0.8 * math.log(k2 / t["F"])
    x1, x2 = math.log(k1 / t["F"]), math.log(k2 / t["F"])
    assert atm["atm_iv_mid"] == pytest.approx(i1 + (0 - x1) / (x2 - x1) * (i2 - i1), abs=1e-8)


def test_last_trade_bias_is_recovered(tmp_path):
    for bias, at in ((0.3, "inside"), (1.0, "ask"), (-1.0, "bid"), (-0.5, "inside")):
        out, truth, _ = analyzed(tmp_path / f"b{bias}", half_spread=0.10, ltp_bias_hs=bias)
        rows = out["rows"]
        assert np.allclose(rows.ltp_minus_mid_halfspreads, bias) and np.allclose(rows.ltp_minus_mid_rs, 0.10 * bias) and (rows.ltp_at == at).all()
        atm = out["atm"].iloc[0]
        near = rows[(rows.offset_strikes.abs() <= 1) & rows.iv_ltp_usable & rows.iv_mid_usable]
        assert len(near) > 0 and np.allclose(near.ltp_minus_mid_vol_pts, np.sign(bias) * abs(bias) * near.half_spread_vol_pts_iv, rtol=0.05)
        assert atm["atm_ltp_minus_mid_vol_pts"] == pytest.approx(bias * atm["atm_half_spread_vol_pts"], rel=0.05)


def test_stale_last_trade_is_excluded_from_the_ltp_variant_only(tmp_path):
    out, _, _ = analyzed(tmp_path, ltp_age=400.0)                                  # 400 s old (after removing the 1.5 s skew): older than 300 s
    rows, atm = out["rows"], out["atm"].iloc[0]
    assert not rows.ltp_fresh.any() and rows.iv_ltp.isna().all() and math.isnan(atm["atm_iv_ltp"]) and rows.iv_mid.notna().sum() > 0 and atm["atm_iv_mid"] == pytest.approx(0.15, abs=1e-8)
    assert rows.ltp_age_bucket.isna().all() and math.isnan(atm["atm_ltp_minus_mid_vol_pts"]) and math.isnan(atm["sell_minus_ltp_vol_pts_cross1"]) and atm["sell_iv_cross1"] == atm["sell_iv_cross1"]
    out2, _, _ = analyzed(tmp_path / "b", ltp_age=298.0)                          # 298 + skew(1.5) - skew = 298 s: still fresh
    assert out2["rows"].ltp_fresh.all() and out2["rows"].ltp_age_s.iloc[0] == pytest.approx(298.0)
    out3, _, _ = analyzed(tmp_path / "c", ltp_age=299.0)                          # recorded 300.5 s, but 299.0 s after removing the skew: fresh
    assert out3["rows"].ltp_fresh.all()


def test_ineligible_rows_are_dropped_and_counted(tmp_path):
    out, _, p = analyzed(tmp_path)
    import sqlite3
    d2 = tmp_path / "bad"
    pth, _ = make_db(str(d2))
    con = sqlite3.connect(pth)
    con.execute("DROP TRIGGER quotes_no_update")
    con.execute("UPDATE quotes SET bid=0 WHERE symbol LIKE '%22700CE'")
    con.execute("UPDATE quotes SET bid=ask+1 WHERE symbol LIKE '%22700PE'")
    con.execute("UPDATE quotes SET capture_minus_feed_s=400 WHERE symbol LIKE '%22750CE'")
    con.execute("UPDATE quotes SET volume=0 WHERE symbol LIKE '%22750PE'")
    con.execute("UPDATE quotes SET data_source='fyers:options-chain-v3' WHERE symbol LIKE '%22800CE'")
    con.commit()
    con.close()
    cyc, q = LD.load_db(pth)
    res = AN.analyze_frames(cyc, q, CFG)
    ex = res["exclusions"].groupby("reason").n.sum().to_dict()
    assert ex == dict(no_bid_or_ask=1, crossed=1, stale_quote=1, never_traded_today=1, not_websocket_row=1)
    assert len(res["rows"]) == 45 and res["atm"].iloc[0]["n_eligible"] == 45 and res["atm"].iloc[0]["n_option_rows"] == 50
    assert not res["rows"].strike.isin([22700.0]).any() or len(res["rows"][res["rows"].strike == 22700.0]) == 0
    assert SM.exclusion_summary(res["exclusions"], res["rows"], res["atm"]).set_index("reason").loc["crossed", "share_of_option_rows"] == pytest.approx(1 / 50)


def test_forward_unavailable_keeps_liquidity_but_no_ivs(tmp_path):
    import sqlite3
    pth, _ = make_db(str(tmp_path))
    con = sqlite3.connect(pth)
    con.execute("DROP TRIGGER quotes_no_update")
    con.execute("UPDATE quotes SET volume=0 WHERE option_type='PE'")                  # no put eligible -> no parity pairs
    con.commit()
    con.close()
    cyc, q = LD.load_db(pth)
    res = AN.analyze_frames(cyc, q, CFG)
    atm = res["atm"].iloc[0]
    assert atm["forward_status"] == "fewer_than_2_pairs" and math.isnan(atm["atm_iv_mid"]) and math.isnan(atm["forward"])
    assert len(res["rows"]) == 25 and res["rows"].iv_mid.isna().all() and res["rows"].spread_rs.notna().all() and not res["rows"].otm_side.any()


def test_otm_only_ivs_and_itm_siblings_use_the_strike_vega(tmp_path):
    out, truth, _ = analyzed(tmp_path)
    rows = out["rows"]
    F = next(iter(truth.values()))["F"]
    for _, r in rows.iterrows():
        otm = (r.option_type == "CE") if r.strike >= F else (r.option_type == "PE")
        assert bool(r.otm_side) == otm and (r.iv_mid == r.iv_mid) == otm
    sib = rows[(rows.strike == 22450.0)]
    assert len(sib) == 2 and sib.vega_rs_per_volpt.nunique() == 1 and sib.vega_rs_per_volpt.iloc[0] > 0                    # call and put share the strike's vega


def test_mid_atm_equals_the_stage2a_smile_builder(tmp_path):
    """Consistency with the research engine: same forward gate, same OTM-only IVs, same strict ATM -> identical ATM IV when fed the same prices."""
    out, truth, _ = analyzed(tmp_path, smile=-0.5, sigma=0.16)
    t0 = next(iter(truth.values()))
    quotes = [OptionQuote(K, OptionType.CALL if typ == "CE" else OptionType.PUT, v["mid"], 0.0) for (h, e, K, typ), v in truth.items()]
    sm = build_smile(quotes, 22500.0, t0["T"], R)
    assert sm.atm is not None and sm.forward.status.value == "ok"
    atm = out["atm"].iloc[0]
    assert atm["forward"] == pytest.approx(sm.forward_used, rel=1e-12) and atm["atm_iv_mid"] == pytest.approx(sm.atm.iv, abs=1e-12) and (atm["atm_strike_low"], atm["atm_strike_high"]) == (sm.atm.strike_low, sm.atm.strike_high)


def test_skew_estimate_changes_T_not_the_data(tmp_path):
    a, _, _ = analyzed(tmp_path / "a", skew=0.0)
    b, _, _ = analyzed(tmp_path / "b", skew=3.0)
    assert a["atm"].iloc[0]["dte_days"] == pytest.approx(b["atm"].iloc[0]["dte_days"])                 # the fake world shifts capture with the skew, so true T is identical


def test_missed_cycles_and_other_expiries(tmp_path):
    p, _ = make_db(str(tmp_path), cycles=(("11:01", "2026-10-13"), ("11:06", "2026-10-13")), status="ok")
    cyc, q = LD.load_db(p)
    cyc2 = cyc.copy()
    cyc2.loc[cyc2.cycle_id.str.endswith("11:06"), "status"] = "missed"
    res = AN.analyze_frames(cyc2, q, CFG)
    assert res["atm"].cycle_id.tolist() == ["2026-10-05T11:01"] and set(res["rows"].cycle_id) == {"2026-10-05T11:01"}


# ================================================================================================ summaries by hand
def hand_rows():
    def row(**k):
        base = dict(day="20261005", cycle_id="c1", dte_bucket="1-3d", moneyness_bucket="ATM", time_bucket="midday", offset_strikes=0, spread_rs=0.2, half_spread_rs=0.1, spread_pct=0.2, bid_size=100, ask_size=200,
                    half_spread_vol_pts_vega=0.02, half_spread_vol_pts_iv=0.03, ltp_at="ask", ltp_fresh=True, ltp_age_bucket="<=10s", ltp_minus_mid_halfspreads=1.0, ltp_minus_mid_rs=0.1, ltp_minus_mid_vol_pts=0.03)
        base.update(k)
        return base
    return pd.DataFrame([row(), row(spread_rs=0.4, half_spread_rs=0.2, spread_pct=0.4, ltp_at="bid", ltp_minus_mid_halfspreads=-1.0, ltp_minus_mid_rs=-0.2, ltp_minus_mid_vol_pts=np.nan),
                         row(ltp_at="inside", ltp_minus_mid_halfspreads=0.0, ltp_minus_mid_rs=0.0, ltp_minus_mid_vol_pts=0.0, day="20261006"), row(dte_bucket="3-7d", spread_rs=1.0, half_spread_rs=0.5, spread_pct=0.6)])


def test_liquidity_summary_by_hand():
    L = SM.liquidity_summary(hand_rows()).set_index(["dte_bucket", "moneyness_bucket", "time_bucket"])
    a = L.loc[("1-3d", "ATM", "midday")]
    assert a.n_rows == 3 and a.n_days == 2 and a.spread_rs_median == pytest.approx(0.2) and a.half_spread_rs_median == pytest.approx(0.1) and a.spread_pct_median == pytest.approx(0.2)
    assert a.share_ltp_at_ask == pytest.approx(1 / 3) and a.share_ltp_at_bid == pytest.approx(1 / 3) and a.share_ltp_inside == pytest.approx(1 / 3) and a.half_spread_vol_pts_iv_median == pytest.approx(0.03)
    assert L.loc[("3-7d", "ATM", "midday")].spread_rs_median == pytest.approx(1.0) and SM.liquidity_summary(pd.DataFrame()).empty


def test_liquidity_shares_are_asymmetric_aware():
    h = hand_rows().iloc[:4].copy()
    h["dte_bucket"], h["ltp_at"] = "1-3d", ["ask", "ask", "bid", "inside"]
    a = SM.liquidity_summary(h).iloc[0]
    assert (a.share_ltp_at_ask, a.share_ltp_at_bid, a.share_ltp_inside) == (0.5, 0.25, 0.25)
    b = SM.last_trade_bias(h).iloc[0]
    assert (b.share_at_ask, b.share_at_bid) == (0.5, 0.25)


def test_last_trade_bias_by_hand():
    B = SM.last_trade_bias(hand_rows()).set_index(["dte_bucket", "moneyness_bucket", "ltp_age_bucket"])
    b = B.loc[("1-3d", "ATM", "<=10s")]
    assert b.n_rows == 3 and b.mean_halfspreads == pytest.approx(0.0) and b.median_halfspreads == pytest.approx(0.0) and b.mean_rs == pytest.approx(-0.1 / 3) and b.n_with_vol_pts == 2
    assert b.median_vol_pts == pytest.approx(0.015) and b.share_at_ask == pytest.approx(1 / 3) and b.p10_halfspreads == pytest.approx(-0.8) and b.p90_halfspreads == pytest.approx(0.8)
    D = SM.last_trade_bias_by_day(hand_rows()).set_index(["day", "dte_bucket"])
    assert D.loc[("20261005", "1-3d")].mean_halfspreads == pytest.approx(0.0) and D.loc[("20261006", "1-3d")].n_rows == 1 and D.loc[("20261005", "3-7d")].n_rows == 1
    far = hand_rows()
    far.loc[len(far)] = far.iloc[0].to_dict() | dict(offset_strikes=5, ltp_minus_mid_halfspreads=9.0)
    assert SM.last_trade_bias_by_day(far).set_index(["day", "dte_bucket"]).loc[("20261005", "1-3d")].n_rows == 2             # the |offset| 5 row is outside the near-the-money window
    stale = hand_rows().assign(ltp_fresh=False)
    assert SM.last_trade_bias(stale).empty


def hand_atm():
    base = dict(day="20261005", dte_bucket="1-3d", primary=False, atm_iv_mid=0.20, atm_iv_bid=0.1995, atm_iv_ask=0.2005, atm_iv_ltp=0.2003, atm_spread_vol_pts=0.10, atm_half_spread_vol_pts=0.05, atm_ltp_minus_mid_vol_pts=0.03)
    out = []
    for p, ltp in ((False, 0.2003), (True, 0.2010), (True, 0.1990)):
        r = dict(base, primary=p, atm_iv_ltp=ltp, atm_ltp_minus_mid_vol_pts=100 * (ltp - 0.20), time="10:01" if p else "11:01", expiry_date="2026-10-06", dte_days=1.2, spot=22500.0, forward=22560.0,
                 forward_status="ok", atm_strike_low=22550.0, atm_strike_high=22600.0, n_eligible=50, india_vix=15.0, skew_est_s=1.5)
        for f in CFG.cross_fractions:
            sell = 0.20 - f * (0.20 - 0.1995)
            r[f"sell_iv_cross{f:g}"] = sell
            r[f"buy_iv_cross{f:g}"] = 0.20 + f * (0.2005 - 0.20)
            r[f"sell_minus_ltp_vol_pts_cross{f:g}"] = 100 * (sell - ltp)
        out.append(r)
    return pd.DataFrame(out)


def test_cost_scenarios_by_hand():
    C_ = SM.cost_scenarios(hand_atm(), CFG).set_index(["dte_bucket", "primary_instant"])
    p = C_.loc[("1-3d", True)]
    assert p.n_cycle_expiries == 2 and p.atm_iv_mid_median == pytest.approx(20.0) and p.atm_half_spread_vol_pts_median == pytest.approx(0.05)
    assert p.sell_minus_ltp_vol_pts_cross0_median == pytest.approx(np.median([100 * (0.20 - 0.2010), 100 * (0.20 - 0.1990)])) and p.sell_minus_mid_vol_pts_cross0_median == pytest.approx(0.0)
    assert p["sell_minus_mid_vol_pts_cross1_median"] == pytest.approx(-0.05) and p["sell_minus_mid_vol_pts_cross0.5_median"] == pytest.approx(-0.025)
    assert p["sell_minus_ltp_vol_pts_cross1_median"] == pytest.approx(np.median([100 * (0.1995 - 0.2010), 100 * (0.1995 - 0.1990)]))
    assert C_.loc[("1-3d", False)].n_cycle_expiries == 1
    assert SM.primary_instants(hand_atm()).time.tolist() == ["10:01", "10:01"] and SM.cost_scenarios(hand_atm().iloc[0:0], CFG).empty


def test_cost_scenarios_need_all_three_ivs():
    a = hand_atm()
    a.loc[0, "atm_iv_bid"] = np.nan
    assert SM.cost_scenarios(a, CFG).set_index(["dte_bucket", "primary_instant"]).loc[("1-3d", False)].n_cycle_expiries if ("1-3d", False) in SM.cost_scenarios(a, CFG).set_index(["dte_bucket", "primary_instant"]).index else True
    assert ("1-3d", False) not in SM.cost_scenarios(a, CFG).set_index(["dte_bucket", "primary_instant"]).index


# ================================================================================================ CLI / reproducibility / safety
def test_run_end_to_end_multi_day_deterministic_and_read_only(tmp_path):
    p1, _ = make_db(str(tmp_path / "d1"), day=date(2026, 10, 5), cycles=(("10:01", "2026-10-06"), ("10:01", "2026-10-13"), ("11:01", "2026-10-13")), dte_days=1.2, ltp_bias_hs=0.2)
    p2, _ = make_db(str(tmp_path / "d2"), day=date(2026, 10, 6), cycles=(("13:01", "2026-10-13"), ("15:01", "2026-10-13")), dte_days=7.2, ltp_bias_hs=-0.2)
    before = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (p1, p2)]
    m1 = RUN.run([p1, p2], str(tmp_path / "o1"))
    m2 = RUN.run([p1, p2], str(tmp_path / "o2"))
    for f in ("row_metrics", "atm_by_cycle_expiry", "primary_instants", "liquidity_summary", "last_trade_bias", "last_trade_bias_by_day", "cost_scenarios", "exclusions"):
        assert open(tmp_path / "o1" / f"{f}.csv", "rb").read() == open(tmp_path / "o2" / f"{f}.csv", "rb").read(), f
    assert [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (p1, p2)] == before                       # inputs untouched
    assert [i["sha256"] for i in m1["inputs"]] == before and m1["n_cycles"] == 4 and m1["config"]["rate"] == 0.065 and "not forecasts" in m1["note"]
    atm = pd.read_csv(tmp_path / "o1" / "atm_by_cycle_expiry.csv", dtype={"day": str})
    assert sorted(atm.day.unique()) == ["20261005", "20261006"] and len(atm) == 5
    assert pd.read_csv(tmp_path / "o1" / "primary_instants.csv").shape[0] == 4                                   # 10:01 x2, 13:01, 15:01
    lt = pd.read_csv(tmp_path / "o1" / "last_trade_bias_by_day.csv", dtype={"day": str})
    assert lt[lt.day == "20261005"].mean_halfspreads.max() == pytest.approx(0.2) and lt[lt.day == "20261006"].mean_halfspreads.max() == pytest.approx(-0.2)
    meta = json.load(open(tmp_path / "o1" / "run_metadata.json"))
    assert meta["tables"]["atm_by_cycle_expiry"] == 5 and RUN.main([p1, "--out", str(tmp_path / "o3"), "--rate", "0.07"]) == 0
    assert json.load(open(tmp_path / "o3" / "run_metadata.json"))["config"]["rate"] == 0.07
    with pytest.raises(ValueError):
        RUN.run([], str(tmp_path / "o4"))


def test_loader_day_and_sha(tmp_path):
    p, _ = make_db(str(tmp_path))
    c, q = LD.load_db(p)
    assert set(c.day) == {"20261005"} and set(q.day) == {"20261005"} and LD.sha256(p) == hashlib.sha256(open(p, "rb").read()).hexdigest() and len(q) == 51
    c2, q2 = LD.load_many([p, p])
    assert len(c2) == 2 and len(q2) == 102


def test_package_is_offline_read_only_and_not_imported_by_the_app():
    d = os.path.join(os.path.dirname(__file__), "..", "micro_analysis")
    for fn in os.listdir(d):
        if fn.endswith(".py"):
            src = open(os.path.join(d, fn), encoding="utf-8").read()
            assert not any(w in src for w in ("import requests", "import socket", "import fyers", "fyers_auth", ".post(", "collector.recorder", "collector.sources", "INSERT", "UPDATE ", "DELETE ")), fn
    root = os.path.join(os.path.dirname(__file__), "..")
    for fn in os.listdir(root):
        if fn.endswith(".py"):
            assert "micro_analysis" not in open(os.path.join(root, fn), encoding="utf-8", errors="ignore").read(), fn
