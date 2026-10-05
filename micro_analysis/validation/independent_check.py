"""Independent verification of micro_analysis output (sqlite3 + pandas + scipy only; no micro_analysis / collector / optionsengine imports).

    python micro_analysis/validation/independent_check.py <micro_YYYYMMDD.sqlite> <analysis_output_dir>

* Re-derives time to expiry from the stored timestamps and the skew.
* Re-solves every bid / mid / ask / last-trade IV of the OTM option at the two ATM-bracketing strikes with a Black-76 brentq solver and rebuilds the ATM IVs and the scenario IVs.
* Rechecks the parity forward against a plain median of K + e^{rT}(C - P) over the nearest strikes (agreement within 1 index point).
* Recomputes every row-level quantity (mid, spreads, last-trade bias, vega, vega-based half-spread) from the raw database rows.
* Recomputes liquidity / last-trade-bias / cost-scenario summaries from row_metrics.csv with plain pandas.
"""
import math
import sqlite3
import sys

import numpy as np
import pandas as pd
import scipy.optimize as so

R = 0.065
YEAR = 365.0 * 86400.0
bad = 0


def check(name, cond, detail=""):
    global bad
    bad += not cond
    print(f"  {'ok ' if cond else 'BAD'} {name} {detail}")


def b76(price, F, K, T, call):
    def f(s):
        d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
        d2 = d1 - s * math.sqrt(T)
        n = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
        df = math.exp(-R * T)
        return (df * (F * n(d1) - K * n(d2)) if call else df * (K * n(-d2) - F * n(-d1))) - price
    try:
        return so.brentq(f, 1e-4, 5.0, xtol=1e-14)
    except ValueError:
        return float("nan")


def vega(F, K, T, s):
    d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return math.exp(-R * T) * F * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T) / 100.0


def main(db_path, out):
    con = sqlite3.connect("file:" + db_path + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    atm = pd.read_csv(f"{out}/atm_by_cycle_expiry.csv", dtype={"day": str})
    rows = pd.read_csv(f"{out}/row_metrics.csv", dtype={"day": str})
    cycles = {r["cycle_id"]: r for r in con.execute("SELECT * FROM cycles")}
    print(f"{db_path}: {len(cycles)} cycles; analysis: {len(atm)} ATM records, {len(rows)} row records")
    # ---- ATM records
    tol = dict(forward=0, t=0, iv=0, scen=0, fwd_plain=0)
    worst = dict(t=0.0, iv=0.0, scen=0.0, fwd=0.0, n=0)
    for _, a in atm.iterrows():
        c = cycles[a.cycle_id]
        q = con.execute("SELECT * FROM quotes WHERE cycle_id=? AND expiry_date=? AND kind='option'", (a.cycle_id, a.expiry_date)).fetchall()
        T = (q[0]["expiry_ts"] - (c["capture_start_ts"] - (c["skew_est_s"] or 0.0))) / YEAR
        worst["t"] = max(worst["t"], abs(T * 365 - a.dte_days))
        if a.forward_status != "ok" or not (a.forward == a.forward):
            continue
        F = a.forward
        by = {(r["strike"], r["option_type"]): r for r in q}
        # plain parity median over the 6 strikes nearest the index
        spot = c["spot"]
        ks = sorted({k for k, _ in by if (k, "CE") in by and (k, "PE") in by}, key=lambda k: abs(k - spot))[:6]
        fw = []
        for k in ks:
            ce, pe = by[(k, "CE")], by[(k, "PE")]
            if ce["bid"] and ce["ask"] and pe["bid"] and pe["ask"]:
                fw.append(k + math.exp(R * T) * ((ce["bid"] + ce["ask"]) / 2 - (pe["bid"] + pe["ask"]) / 2))
        if fw:
            worst["fwd"] = max(worst["fwd"], abs(float(np.median(fw)) - F))
        lo, hi = a.atm_strike_low, a.atm_strike_high
        if not (lo == lo):
            continue
        ivs = {}
        for v in ("mid", "bid", "ask", "ltp"):
            pts = {}
            for k in (lo, hi):
                call = k >= F
                r = by[(k, "CE" if call else "PE")]
                price = {"mid": (r["bid"] + r["ask"]) / 2, "bid": r["bid"], "ask": r["ask"], "ltp": r["ltp"]}[v]
                pts[k] = (math.log(k / F), b76(price, F, k, T, call))
            if any(p[1] != p[1] for p in pts.values()):
                continue
            w = (0 - pts[lo][0]) / (pts[hi][0] - pts[lo][0])
            ivs[v] = pts[lo][1] + w * (pts[hi][1] - pts[lo][1])
        for v in ("mid", "bid", "ask"):
            got = a[f"atm_iv_{v}"]
            worst["iv"] = max(worst["iv"], abs(got - ivs[v]))
        if a.atm_iv_ltp == a.atm_iv_ltp and "ltp" in ivs:
            worst["iv"] = max(worst["iv"], abs(a.atm_iv_ltp - ivs["ltp"]))
        for f in (0.0, 0.5, 1.0):
            sell = ivs["mid"] - f * (ivs["mid"] - ivs["bid"])
            worst["scen"] = max(worst["scen"], abs(a[f"sell_iv_cross{f:g}"] - sell))
        worst["n"] += 1
    check("time to expiry equals (expiry - (capture - skew)) / 365d for every ATM record", worst["t"] < 1e-9, f"(max diff {worst['t']:.2e} days)")
    check(f"bid / mid / ask / last-trade ATM IVs equal an independent Black-76 re-solve ({worst['n']} records)", worst["iv"] < 2e-6, f"(max diff {worst['iv']:.2e})")
    check("sell-side scenario IVs equal mid - f*(mid - bid)", worst["scen"] < 2e-6, f"(max diff {worst['scen']:.2e})")
    check("parity forward within 1 index point of a plain median parity estimate", worst["fwd"] < 1.0, f"(max diff {worst['fwd']:.3f} pts)")
    # ---- row records vs the raw database
    raw = pd.read_sql_query("SELECT cycle_id, symbol, bid, ask, ltp, capture_minus_last_trade_s, strike, option_type, expiry_date FROM quotes WHERE kind='option'", con)
    m = rows.merge(raw, on=["cycle_id", "expiry_date", "strike", "option_type"], suffixes=("", "_raw"))
    check("every analysed row exists in the database", len(m) == len(rows))
    check("mid / spread / half-spread / spread% recomputed", np.allclose(m["mid"], (m.bid_raw + m.ask_raw) / 2) and np.allclose(m.spread_rs, m.ask_raw - m.bid_raw) and np.allclose(m.half_spread_rs, (m.ask_raw - m.bid_raw) / 2)
          and np.allclose(m.spread_pct, 100 * (m.ask_raw - m.bid_raw) / ((m.ask_raw + m.bid_raw) / 2)))
    d = m.ltp_raw - (m.bid_raw + m.ask_raw) / 2
    check("last-trade minus mid (rupees and half-spreads) recomputed", np.allclose(m.ltp_minus_mid_rs, d) and np.allclose(m.ltp_minus_mid_halfspreads.dropna(), (d / m.half_spread_rs)[m.ltp_minus_mid_halfspreads.notna()]))
    at = np.where(m.ltp_raw >= m.ask_raw, "ask", np.where(m.ltp_raw <= m.bid_raw, "bid", "inside"))
    check("ltp_at (ask / bid / inside) recomputed", (m.ltp_at.values == at).all())
    # vega and the vega-based half spread
    a_idx = atm.set_index(["cycle_id", "expiry_date"])
    vbad = 0
    for _, r in m[m.vega_rs_per_volpt.notna()].iterrows():
        a = a_idx.loc[(r.cycle_id, r.expiry_date)]
        T = a.dte_days / 365.0
        v = vega(a.forward, r.strike, T, r.iv_mid) if r.iv_mid == r.iv_mid else None
        if v is None:                                                  # ITM sibling: vega taken from its OTM twin
            twin = m[(m.cycle_id == r.cycle_id) & (m.expiry_date == r.expiry_date) & (m.strike == r.strike) & m.iv_mid.notna()]
            v = vega(a.forward, r.strike, T, twin.iv_mid.iloc[0])
        vbad += abs(v - r.vega_rs_per_volpt) > 1e-6 * max(1, v) or abs(r.half_spread_rs / v - r.half_spread_vol_pts_vega) > 1e-9
    check("Black-76 vega (rupees per vol point) and the vega-based half-spread recomputed", vbad == 0, f"({vbad} mismatches)")
    # IV spread rows
    both = m[m.iv_spread_vol_pts.notna()]
    ibad = 0
    for _, r in both.iterrows():
        a = a_idx.loc[(r.cycle_id, r.expiry_date)]
        T = a.dte_days / 365.0
        call = r.option_type == "CE"
        ib, ia = b76(r.bid_raw, a.forward, r.strike, T, call), b76(r.ask_raw, a.forward, r.strike, T, call)
        ibad += abs(100 * (ia - ib) - r.iv_spread_vol_pts) > 1e-5
    check(f"IV spread (vol points) re-solved for {len(both)} rows", ibad == 0, f"({ibad} mismatches)")
    # ---- summaries
    liq = pd.read_csv(f"{out}/liquidity_summary.csv")
    g = rows.groupby(["dte_bucket", "moneyness_bucket", "time_bucket"])
    exp = g.agg(n=("spread_rs", "size"), med=("half_spread_rs", "median"), pct=("spread_pct", "median")).reset_index()
    j = liq.merge(exp, on=["dte_bucket", "moneyness_bucket", "time_bucket"])
    check("liquidity_summary n_rows / median half-spread / median spread% recomputed", len(j) == len(liq) and (j.n_rows == j.n).all() and np.allclose(j.half_spread_rs_median, j.med) and np.allclose(j.spread_pct_median, j.pct))
    ltb = pd.read_csv(f"{out}/last_trade_bias.csv")
    fresh = rows[rows.ltp_fresh & rows.ltp_age_bucket.notna()]
    e2 = fresh.groupby(["dte_bucket", "moneyness_bucket", "ltp_age_bucket"]).agg(n=("spread_rs", "size"), mean=("ltp_minus_mid_halfspreads", "mean")).reset_index()
    j2 = ltb.merge(e2, on=["dte_bucket", "moneyness_bucket", "ltp_age_bucket"])
    check("last_trade_bias counts and mean (half-spreads) recomputed", len(j2) == len(ltb) and (j2.n_rows == j2.n).all() and np.allclose(j2.mean_halfspreads, j2["mean"], equal_nan=True))
    cs = pd.read_csv(f"{out}/cost_scenarios.csv")
    a3 = atm[atm.atm_iv_mid.notna() & atm.atm_iv_bid.notna() & atm.atm_iv_ask.notna()]
    ok = True
    for _, r in cs.iterrows():
        sub = a3[(a3.dte_bucket == r.dte_bucket) & (a3.primary == r.primary_instant)]
        ok &= len(sub) == r.n_cycle_expiries and abs(100 * ((sub.atm_iv_ask - sub.atm_iv_bid) / 2).median() - r.atm_half_spread_vol_pts_median) < 1e-9
        ok &= abs((100 * (sub.atm_iv_mid - 1.0 * (sub.atm_iv_mid - sub.atm_iv_bid) - sub.atm_iv_mid)).median() - r["sell_minus_mid_vol_pts_cross1_median"]) < 1e-9
    check("cost_scenarios counts, median half-spread (vol pts) and the cross-1 sell-minus-mid medians recomputed", ok)
    print("\nRESULT:", "ALL CHECKS PASSED" if bad == 0 else f"{bad} CHECKS FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
