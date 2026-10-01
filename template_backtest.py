"""template_backtest.py -- 5-year track record of every Strategy Ideas
template, replayed on the downloaded 1-min option history (data/hist1m).

For every weekly expiry, every trading day of its own week AND the week
before (days to expiry 9..0 -- so next week's expiry can be matched too)
and three entry times (09:30, 12:00, 14:30) the SAME rules the live
Strategy Ideas table uses (strategy_ideas._candidates) pick the strikes --
deltas computed from that moment's historical prices, range from that day's
volume leaders -- and each strategy is then followed minute by minute:

  hold      held to expiry, settled at intrinsic vs NIFTY's expiry close
  managed   credit strategies: book at +50% of the credit, stop at -2x the
            credit; debit strategies: book at +100% of the debit, stop at
            -50% of it (checked on 1-min closes, else held to expiry)

Prices are the 1-min closes (bid/ask aren't in the history) with 0.5 pt
slippage per order; rupees are for 1 lot of the CURRENT lot size (so years
compare fairly), after round-trip charges.

Output: results/template_bt/trades.parquet (one row per trade)
        results/template_bt/summary.parquet (per template x days-to-expiry x entry time)
Usage:  python template_backtest.py [--years 5]
"""
import math
import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import charges
import greeks as g
import manual_trades
import market_calendar as mc
import paths
import simulator as S
from intraday_range_bt import Day
from strategy_ideas import _candidates

OUT = os.path.join(paths.BASE_DIR, "results", "template_bt")
SLOTS = ("09:30", "12:00", "14:30")
SLIP = 0.5
LOT = manual_trades.LOT_SIZE
MANAGE = {"credit": (0.5, 2.0), "debit": (1.0, 0.5)}   # (take profit x premium, stop x premium)
F_O_CLOSE_CHANGE = date(2026, 8, 3)
WEEKS = 2       # entries from this many expiry weeks (1 = the expiry's own week only)
MAX_DTE = 9


def _week(exp, prev):
    df = S._load_options(exp)
    lo = pd.Timestamp(prev).date() if prev else S._week_of(exp)[0]
    df = df[(df["day"] > lo) & (df["day"] <= pd.Timestamp(exp).date())]
    mins = df["date"].dt.hour * 60 + df["date"].dt.minute
    return df[mins <= 15 * 60 + 40]


def _matrix(df):
    """Minute index + forward-filled close per contract + last-trade minute."""
    ts = np.sort(df["ts"].unique())
    n = len(ts)
    px, last = {}, {}
    for (typ, k), gdf in df.groupby(["type", "strike"], sort=False):
        idx = np.searchsorted(ts, gdf["ts"].to_numpy())
        c = np.full(n, np.nan)
        c[idx] = gdf["close"].to_numpy()
        lt = np.full(n, -1)
        lt[idx] = idx
        px[(int(k), typ)] = pd.Series(c).ffill().to_numpy()
        last[(int(k), typ)] = np.maximum.accumulate(lt)
    return ts, px, last


def run(years=5, verbose=True):
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") == "done")
    cutoff = (date.today() - timedelta(days=int(365.25 * years))).isoformat()
    sp = S._load_spot()
    st, sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
    allk = sorted(man)  # includes failed/live expiries for the "previous expiry" boundary
    rows = []
    t0 = time.time()
    for n_e, exp in enumerate(exps):
        if exp < cutoff:
            continue
        i = allk.index(exp)
        # entries from the expiry's own week AND the week before it (days to
        # expiry ~0-9), so NEXT week's expiry can be matched too
        prev = allk[i - WEEKS] if i >= WEEKS else (allk[0] if i else None)
        df = _week(exp, prev)
        if df.empty:
            continue
        ts, px, last = _matrix(df)
        exp_d = pd.Timestamp(exp).date()
        close_t = "15:40" if exp_d >= F_O_CLOSE_CHANGE else "15:30"
        exp_close_ts = int(pd.Timestamp(f"{exp} {close_t}", tz=S.IST).timestamp())
        j = np.searchsorted(st, int(pd.Timestamp(f"{exp} 23:59", tz=S.IST).timestamp()), side="right") - 1
        settle = float(sc[j])
        days = sorted(df["day"].unique())
        tdays = [d for d in days]
        for day in days:
            # trading days left after `day` (NSE calendar -- the same count the live table uses)
            dte, dd_ = 0, day + timedelta(days=1)
            while dd_ <= exp_d:
                dte += mc.is_trading_day(dd_)
                dd_ += timedelta(days=1)
            if dte > MAX_DTE:
                continue
            dd = df[df["day"] == day]
            D = Day(dd, 100, 5, st, sc)
            for slot in SLOTS:
                t_ent = int(pd.Timestamp(f"{day} {slot}", tz=S.IST).timestamp())
                e = int(np.searchsorted(ts, t_ent))
                if e >= len(ts) or ts[e] - t_ent > 300:
                    continue
                day_start = int(np.searchsorted(ts, int(pd.Timestamp(f"{day} 09:15", tz=S.IST).timestamp())))
                # quotes at entry: contracts that traded today, near the money
                q = {}
                spot_now = float(sc[max(0, np.searchsorted(st, ts[e], side="right") - 1)])
                near = []
                for (k, typ), arr in px.items():
                    p_ = arr[e]
                    if not np.isfinite(p_) or last[(k, typ)][e] < day_start or abs(k - spot_now) > 1600:
                        continue
                    q[(k, typ)] = {"close": float(p_)}
                for k in {k for (k, t) in q}:
                    if (k, "CE") in q and (k, "PE") in q:
                        near.append((k, q[(k, "CE")]["close"], q[(k, "PE")]["close"]))
                near = sorted(near, key=lambda x: abs(x[0] - spot_now))[:6]
                fwd = g.synthetic_forward(near) if near else None
                if not fwd:
                    continue
                T = (exp_close_ts - ts[e]) / (365 * 86400)
                if T <= 0:
                    continue
                for (k, typ), v in q.items():
                    iv = g.implied_vol_fwd(v["close"], fwd, k, T, g.RISK_FREE_RATE, typ == "CE")
                    v["g"] = g.compute_greeks_fwd(fwd, k, T, g.RISK_FREE_RATE, iv, typ == "CE") if iv else None
                # regime at entry: ATM implied vol (average of the ATM call and put)
                atm_k = int(round(fwd / 50) * 50)
                ivs = []
                for t_ in ("CE", "PE"):
                    v_ = q.get((atm_k, t_))
                    if v_ and v_.get("g"):
                        iv_ = g.implied_vol_fwd(v_["close"], fwd, atm_k, T, g.RISK_FREE_RATE, t_ == "CE")
                        if iv_:
                            ivs.append(iv_)
                atm_iv = round(sum(ivs) / len(ivs) * 100, 2) if ivs else None
                # the day's running range at this minute (volume leaders + first-candle highs)
                di = next((ii for ii, h in enumerate(D.hhmm) if h >= slot), None)
                rng = D.rng(D.leaders(di - 1)) if di and D.leaders(di - 1) else None
                for c in _candidates(q, fwd, rng):
                    legs = c["legs"]
                    if any((k, t) not in q for _, t, k in legs):
                        continue
                    sgn = np.array([1 if s_ == "SELL" else -1 for s_, _, _ in legs])
                    ent = np.array([q[(k, t)]["close"] for _, t, k in legs])
                    fill = ent - sgn * SLIP                          # sell lower / buy higher
                    prem = float((sgn * fill).sum())                 # + credit / - debit (per unit)
                    credit = prem > 0
                    if abs(prem) < 0.5:
                        continue
                    path = np.vstack([px[(k, t)][e + 1:] for _, t, k in legs])       # future closes
                    if path.shape[1] == 0:
                        continue
                    path = np.where(np.isfinite(path), path, ent[:, None])
                    mtm = prem - (sgn[:, None] * (path + sgn[:, None] * SLIP)).sum(axis=0)  # P&L per unit if closed now
                    intr = np.array([max(0.0, settle - k) if t == "CE" else max(0.0, k - settle) for _, t, k in legs])
                    hold = prem - float((sgn * intr).sum())
                    tp_x, sl_x = MANAGE["credit" if credit else "debit"]
                    base = abs(prem)
                    hit_tp = np.flatnonzero(mtm >= tp_x * base)
                    hit_sl = np.flatnonzero(mtm <= -sl_x * base)
                    first_tp = hit_tp[0] if len(hit_tp) else None
                    first_sl = hit_sl[0] if len(hit_sl) else None
                    if first_tp is not None and (first_sl is None or first_tp < first_sl):
                        managed, why, xi = float(mtm[first_tp]), "TP", first_tp
                    elif first_sl is not None:
                        managed, why, xi = float(mtm[first_sl]), "SL", first_sl
                    else:
                        managed, why, xi = hold, "EXP", None
                    units = LOT
                    ch_in = sum(charges.order_charges("SELL" if s_ > 0 else "BUY", f_, units) for s_, f_ in zip(sgn, fill))
                    ch_hold = ch_in
                    if why != "EXP":
                        exit_px = path[:, xi] + sgn * SLIP
                        ch_man = ch_in + sum(charges.order_charges("BUY" if s_ > 0 else "SELL", x_, units) for s_, x_ in zip(sgn, exit_px))
                    else:
                        ch_man = ch_in
                    rows.append({"expiry": exp, "day": day.isoformat(), "dte": dte, "slot": slot, "tpl": c["tpl"], "atm_iv": atm_iv,
                                 "name": c["name"].split(" · ")[0], "direction": c["direction"], "credit": credit,
                                 "premium_pts": round(prem, 2), "hold_pts": round(hold, 2), "managed_pts": round(managed, 2),
                                 "managed_exit": why, "mae_pts": round(float(mtm.min()), 2),
                                 "hold_rs": round(hold * units - ch_hold, 2), "managed_rs": round(managed * units - ch_man, 2)})
        if verbose:
            print(f"\r{exp}: {len(rows):,} trades · {time.time() - t0:.0f}s", end="", flush=True)
    if verbose:
        print()
    tr = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tr.to_parquet(os.path.join(OUT, "trades.parquet"), compression="zstd", index=False)
    summ = summarize(tr)
    summ.to_parquet(os.path.join(OUT, "summary.parquet"), index=False)
    return tr, summ


def summarize(tr):
    def agg(x):
        h, m = x.hold_rs, x.managed_rs
        return pd.Series({"n": len(x), "win_hold": (h > 0).mean() * 100, "avg_hold": h.mean(), "med_hold": h.median(),
                          "worst_hold": h.min(), "p5_hold": h.quantile(0.05),
                          "win_man": (m > 0).mean() * 100, "avg_man": m.mean(), "worst_man": m.min(),
                          "sl_rate": (x.managed_exit == "SL").mean() * 100, "tp_rate": (x.managed_exit == "TP").mean() * 100,
                          "avg_premium_rs": x.premium_pts.mean() * LOT})
    return tr.groupby(["tpl", "dte", "slot"]).apply(agg, include_groups=False).reset_index().round(2)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    yrs = float(sys.argv[sys.argv.index("--years") + 1]) if "--years" in sys.argv else 5
    tr, sm = run(yrs)
    print(len(tr), "trades;", len(sm), "template x dte x slot cells")
