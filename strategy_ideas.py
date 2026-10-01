"""strategy_ideas.py -- auto-generated option strategies from the live chain,
each with risk/reward and probability of profit, for the Strategy Builder's
"Strategy Ideas" panel.

For one expiry it builds every standard single-expiry strategy (credit and
debit spreads, strangles/straddles, condors, flies, butterflies, jade lizards,
single legs), with strikes chosen by delta, by the volume-leader range and by
ATM, then for 1 lot:

  price       SELL legs at the bid, BUY legs at the ask (what you'd really get)
  payoff      at expiry on a NIFTY grid (+/-25% of the forward), after
              round-trip charges -> max profit / max loss (or unlimited),
              breakevens, reward : risk
  POP (model) probability NIFTY expires in the profit zone under a lognormal
              with today's ATM implied vol; EV = expected P&L under it
  POP (hist)  the same, but with NIFTY's real moves over the same number of
              trading minutes, from 5 years of 1-min history (data/hist1m)
  greeks      net delta / theta per day / vega per vol point, in rupees
  fit         whether it suits the current view (Bullish / Bearish / Neutral,
              from market_view) and IV regime (IV rank)
"""
import math
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import charges
import greeks as g
import market_calendar as mc
import paths

STEP = 50
DELTAS = (0.10, 0.15, 0.20, 0.25, 0.30)
WIDTHS = (100, 200, 300)
GRID_PCT = 0.25
GRID_STEP = 5.0
MIN_PER_DAY = 375            # NIFTY index session 09:15-15:30 (settlement is on the index close)
_hist_cache = {}
TRACK_PATH = os.path.join(paths.BASE_DIR, "results", "template_bt", "summary.parquet")
TRACK_MIN_N = 20             # fewer past trades than this -> shown as "too few"
_track = {"mtime": None, "df": None}
TRADES_PATH = os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet")
SIMILAR_MIN, SIMILAR_FRAC = 30, 0.25   # similar-IV sample: the nearest quarter of past trades, at least 30
_trades = {"mtime": None, "groups": None}


def _trade_groups():
    """template_backtest trades grouped by (template, days to expiry, entry slot),
    as arrays for fast similar-IV look-ups."""
    try:
        m = os.path.getmtime(TRADES_PATH)
    except OSError:
        return None
    if _trades["mtime"] != m:
        t = pd.read_parquet(TRADES_PATH, columns=["tpl", "dte", "slot", "atm_iv", "hold_rs", "managed_rs", "managed_exit"])
        t = t.dropna(subset=["atm_iv"])
        _trades["groups"] = {k: (x.atm_iv.to_numpy(), x.hold_rs.to_numpy(), x.managed_rs.to_numpy(), (x.managed_exit == "SL").to_numpy())
                             for k, x in t.groupby(["tpl", "dte", "slot"])}
        _trades["mtime"] = m
    return _trades["groups"]


def _similar(tpl, dte, slot, iv_now):
    """Past trades of this template/dte/slot whose entry ATM IV was closest to today's."""
    gr = _trade_groups()
    if not gr or iv_now is None or (tpl, dte, slot) not in gr:
        return None
    iv, hold, man, sl = gr[(tpl, dte, slot)]
    if len(iv) < SIMILAR_MIN:
        return None
    k = max(SIMILAR_MIN, int(len(iv) * SIMILAR_FRAC))
    idx = np.argsort(np.abs(iv - iv_now))[:k]
    h, m = hold[idx], man[idx]
    return {"n": int(k), "iv_lo": round(float(iv[idx].min()), 1), "iv_hi": round(float(iv[idx].max()), 1),
            "win_hold": round(float((h > 0).mean() * 100), 1), "avg_hold": round(float(h.mean()), 1),
            "worst_hold": round(float(h.min()), 1), "p5_hold": round(float(np.percentile(h, 5)), 1),
            "win_man": round(float((m > 0).mean() * 100), 1), "avg_man": round(float(m.mean()), 1),
            "sl_rate": round(float(sl[idx].mean() * 100), 1),
            # where today's IV sits among this template's past entries (0 = lowest, 100 = highest)
            "iv_pct": round(float((iv < iv_now).mean() * 100), 0)}


def _track_table():
    """5-year template track record (template_backtest.py), re-read when the file changes."""
    try:
        m = os.path.getmtime(TRACK_PATH)
    except OSError:
        return None
    if _track["mtime"] != m:
        _track["df"] = pd.read_parquet(TRACK_PATH).set_index(["tpl", "dte", "slot"])
        _track["mtime"] = m
    return _track["df"]


def track_context(expiry_close_ts, now=None):
    """(days to expiry in trading days, nearest entry slot) the backtest is
    matched on. Outside market hours: the next session's 09:30."""
    now = now or mc.now_ist()
    day = now.date()
    hm = now.hour * 60 + now.minute
    if not mc.is_trading_day(day) or hm >= 15 * 60 + 30:
        day = day + timedelta(days=1)
        while not mc.is_trading_day(day):
            day += timedelta(days=1)
        slot = "09:30"
    else:
        slot = "09:30" if hm < 10 * 60 + 45 else "12:00" if hm < 13 * 60 + 15 else "14:30"
    exp = datetime.fromtimestamp(expiry_close_ts, mc.IST).date()
    dte, d = 0, day + timedelta(days=1)
    while d <= exp:
        dte += mc.is_trading_day(d)
        d += timedelta(days=1)
    return dte, slot


def _track_for(tpl, dte, slot):
    t = _track_table()
    if t is None:
        return None
    try:
        r = t.loc[(tpl, dte, slot)]
    except KeyError:
        return None
    out = {k: (None if pd.isna(v) else round(float(v), 1)) for k, v in r.items()}
    out.update(dte=dte, slot=slot, enough=out["n"] >= TRACK_MIN_N)
    return out


# ---------------------------------------------------------------------------
# market inputs
def _quotes(chain, fwd, T):
    q = {}
    for r in chain["strikes"]:
        k = int(r["strike"])
        for key, typ in (("ce", "CE"), ("pe", "PE")):
            leg = r.get(key) or {}
            bid, ask, ltp = leg.get("bid") or 0, leg.get("ask") or 0, leg.get("ltp")
            mid = (bid + ask) / 2 if bid and ask else ltp
            if not mid or not leg.get("symbol"):
                continue
            iv = g.implied_vol_fwd(mid, fwd, k, T, g.RISK_FREE_RATE, typ == "CE") if T > 0 else None
            gk = g.compute_greeks_fwd(fwd, k, T, g.RISK_FREE_RATE, iv, typ == "CE") if iv else None
            q[(k, typ)] = {"symbol": leg["symbol"], "bid": bid or ltp, "ask": ask or ltp, "mid": mid, "ltp": ltp,
                           "iv": iv, "g": gk, "oi": leg.get("oi"), "volume": leg.get("volume")}
    return q


def trading_minutes_left(expiry_close_ts, now=None):
    now = now or mc.now_ist()
    exp = datetime.fromtimestamp(expiry_close_ts, mc.IST)
    total, d = 0, now.date()
    while d <= exp.date():
        if mc.is_trading_day(d):
            o = datetime.combine(d, datetime.min.time(), mc.IST).replace(hour=9, minute=15)
            c = o + timedelta(minutes=MIN_PER_DAY)
            s = max(o, now)
            if s < c:
                total += (c - s).total_seconds() / 60
        d += timedelta(days=1)
    return max(1, int(total))


def _hist_returns(h):
    """NIFTY log-returns over h trading minutes (1-min bars, sessions joined
    back to back so overnight gaps are included), sampled every 5 bars."""
    key = int(round(h / 5.0) * 5) or 1
    if key in _hist_cache:
        return _hist_cache[key]
    p = os.path.join(paths.BASE_DIR, "data", "hist1m", "NIFTY50_1m.parquet")
    if not os.path.exists(p):
        return None
    c = pd.read_parquet(p, columns=["ts", "close"]).sort_values("ts")["close"].to_numpy(dtype=float)
    if len(c) <= key + 10:
        return None
    r = np.log(c[key:] / c[:-key])[::5]
    _hist_cache[key] = r
    return r


# ---------------------------------------------------------------------------
# strategy construction
def _near_delta(q, typ, target, fwd):
    """OTM strike (100-pt or 50-pt) whose |delta| is closest to target."""
    best = None
    for (k, t), v in q.items():
        if t != typ or not v["g"]:
            continue
        if (typ == "CE" and k <= fwd) or (typ == "PE" and k >= fwd):
            continue
        d = abs(abs(v["g"]["delta"]) - target)
        if best is None or d < best[0]:
            best = (d, k)
    return best[1] if best else None


def _candidates(q, fwd, rng):
    """Every strategy to evaluate. Each carries `tpl`: a strike-independent
    template key (the same key in the live table and in the 5-year template
    backtest -- template_backtest.py), e.g. "Iron condor|Δ0.15|w100"."""
    atm = int(round(fwd / STEP) * STEP)
    cands = []

    def add(name, cat, direction, legs, tag, tpl):
        cands.append({"name": name + (f" · {tag}" if tag else ""), "category": cat, "direction": direction,
                      "legs": legs, "tpl": f"{name}|{tpl}"})

    short_ce = {f"Δ{d:.2f}": _near_delta(q, "CE", d, fwd) for d in DELTAS}
    short_pe = {f"Δ{d:.2f}": _near_delta(q, "PE", d, fwd) for d in DELTAS}
    if rng and rng.get("upper") and rng.get("lower"):
        short_ce["range"] = int(round(rng["upper"] / STEP) * STEP)
        short_pe["range"] = int(round(rng["lower"] / STEP) * STEP)
    pairs = [(t, short_ce[t], short_pe[t]) for t in short_ce if short_ce[t] and short_pe.get(t)]
    off = lambda k: "ATM" if k == atm else f"ATM{k - atm:+d}"

    # ---- neutral, credit
    add("Short straddle", "Neutral · credit", "neutral", [("SELL", "CE", atm), ("SELL", "PE", atm)], f"ATM {atm}", "ATM")
    for tag, c, p in pairs:
        add("Short strangle", "Neutral · credit", "neutral", [("SELL", "CE", c), ("SELL", "PE", p)], tag, tag)
        for w in WIDTHS:
            add("Iron condor", "Neutral · credit", "neutral",
                [("SELL", "CE", c), ("BUY", "CE", c + w), ("SELL", "PE", p), ("BUY", "PE", p - w)],
                f"{tag}, wings {w}", f"{tag}|w{w}")
    for w in (100, 200, 300, 400):
        add("Iron butterfly", "Neutral · credit", "neutral",
            [("SELL", "CE", atm), ("SELL", "PE", atm), ("BUY", "CE", atm + w), ("BUY", "PE", atm - w)],
            f"ATM {atm}, wings {w}", f"ATM|w{w}")
    # ---- directional, credit
    for d in ("Δ0.20", "Δ0.30", "range"):
        p, c = short_pe.get(d), short_ce.get(d)
        for w in WIDTHS:
            if p:
                add("Bull put spread", "Bullish · credit", "bullish", [("SELL", "PE", p), ("BUY", "PE", p - w)], f"{d}, width {w}", f"{d}|w{w}")
            if c:
                add("Bear call spread", "Bearish · credit", "bearish", [("SELL", "CE", c), ("BUY", "CE", c + w)], f"{d}, width {w}", f"{d}|w{w}")
        if p:
            add("Short put", "Bullish · credit", "bullish", [("SELL", "PE", p)], d, d)
        if c:
            add("Short call", "Bearish · credit", "bearish", [("SELL", "CE", c)], d, d)
    for d in ("Δ0.20", "Δ0.30"):
        p, c = short_pe.get(d), short_ce.get(d)
        if p and c:
            add("Jade lizard", "Neutral-bullish · credit", "bullish", [("SELL", "PE", p), ("SELL", "CE", c), ("BUY", "CE", c + 100)], d, d)
            add("Reverse jade lizard", "Neutral-bearish · credit", "bearish", [("SELL", "CE", c), ("SELL", "PE", p), ("BUY", "PE", p - 100)], d, d)
    # ---- directional, debit
    for o in (0, -100, 100):
        k = atm + o
        for w in WIDTHS:
            add("Bull call spread", "Bullish · debit", "bullish", [("BUY", "CE", k), ("SELL", "CE", k + w)], f"buy {k}, width {w}", f"buy {off(k)}|w{w}")
            add("Bear put spread", "Bearish · debit", "bearish", [("BUY", "PE", k), ("SELL", "PE", k - w)], f"buy {k}, width {w}", f"buy {off(k)}|w{w}")
    add("Long call", "Bullish · debit", "bullish", [("BUY", "CE", atm)], f"ATM {atm}", "ATM")
    add("Long put", "Bearish · debit", "bearish", [("BUY", "PE", atm)], f"ATM {atm}", "ATM")
    # ---- neutral, debit (pin) and long volatility
    centres = {atm: "ATM"}
    if rng and rng.get("upper") and rng.get("lower"):
        centres.setdefault(int(round((rng["upper"] + rng["lower"]) / 2 / STEP) * STEP), "range mid")
    for cK, lab in centres.items():
        for w in (100, 200):
            add("Long call butterfly", "Neutral · debit", "neutral",
                [("BUY", "CE", cK - w), ("SELL", "CE", cK), ("SELL", "CE", cK), ("BUY", "CE", cK + w)],
                f"centre {cK}, wings {w}", f"{lab}|w{w}")
    add("Long straddle", "Volatility · debit", "volatile", [("BUY", "CE", atm), ("BUY", "PE", atm)], f"ATM {atm}", "ATM")
    for d in ("Δ0.20", "Δ0.30"):
        if short_ce.get(d) and short_pe.get(d):
            add("Long strangle", "Volatility · debit", "volatile", [("BUY", "CE", short_ce[d]), ("BUY", "PE", short_pe[d])], d, d)
    # de-duplicate identical leg sets (first template wins)
    seen, out = set(), []
    for c in cands:
        key = tuple(sorted(c["legs"]))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# evaluation
def _evaluate(c, q, fwd, T, lot, grid, dens_model, dens_hist, sigma, mid=False):
    """mid=True prices every leg at the bid/ask mid (for comparing adjustment
    choices on equal terms; the switching cost is added separately)."""
    legs, credit, chg = [], 0.0, 0.0
    delta = theta = vega = 0.0
    for side, typ, k in c["legs"]:
        v = q.get((k, typ))
        if not v:
            return None
        px = v["mid"] if mid else (v["bid"] if side == "SELL" else v["ask"])
        if not px or px <= 0:
            return None
        sgn = 1 if side == "SELL" else -1
        credit += sgn * px
        chg += charges.round_trip(side, px, px, lot)
        if v["g"]:
            delta -= sgn * v["g"]["delta"]
            theta -= sgn * v["g"]["theta"]
            vega -= sgn * v["g"]["vega"]
        legs.append({"side": side, "type": typ, "strike": k, "price": round(px, 2), "symbol": v["symbol"],
                     "iv": round(v["iv"] * 100, 1) if v["iv"] else None})
    pay = np.full_like(grid, credit)
    for side, typ, k in c["legs"]:
        intr = np.maximum(grid - k, 0) if typ == "CE" else np.maximum(k - grid, 0)
        pay += -intr if side == "SELL" else intr
    pnl = pay * lot - chg
    # unlimited: still falling / rising at the grid's edges
    slope_lo, slope_hi = pnl[1] - pnl[0], pnl[-1] - pnl[-2]
    unl_loss = bool(slope_lo > 1e-6 or slope_hi < -1e-6)
    unl_profit = bool(slope_lo < -1e-6 or slope_hi > 1e-6)
    max_p = None if unl_profit else float(pnl.max())
    max_l = None if unl_loss else float(pnl.min())
    s = np.sign(pnl)
    idx = np.flatnonzero(np.diff(s) != 0)
    bes = [round(float(grid[i] - pnl[i] * (grid[i + 1] - grid[i]) / (pnl[i + 1] - pnl[i])), 1) for i in idx]
    win = pnl > 0
    pop_m = float(dens_model[win].sum())
    ev_m = float((pnl * dens_model).sum())
    pop_h = float(dens_hist[win].sum()) if dens_hist is not None else None
    ev_h = float((pnl * dens_hist).sum()) if dens_hist is not None else None
    rr = (max_p / -max_l) if (max_p and max_l and max_l < 0) else None
    # CVaR 5% on NIFTY's real moves: the average P&L of the worst 5% of outcomes
    cvar = None
    dd = dens_hist if dens_hist is not None else dens_model
    order = np.argsort(pnl)
    cut = int(np.searchsorted(np.cumsum(dd[order]), 0.05)) + 1
    w_ = dd[order][:cut]
    if w_.sum() > 0:
        cvar = float((pnl[order][:cut] * w_).sum() / w_.sum())
    em = fwd * sigma * math.sqrt(T)            # 1 s.d. expected move
    sel = (grid >= fwd - 3 * em) & (grid <= fwd + 3 * em)
    gi = np.flatnonzero(sel)
    gi = gi[:: max(1, len(gi) // 70)]
    curve = [[round(float(grid[i]), 1), round(float(pnl[i]), 0)] for i in gi]
    return {**{k: c[k] for k in ("name", "category", "direction", "tpl")}, "legs": legs,
            "net_premium": round(credit * lot, 2), "credit": bool(credit > 0), "charges": round(chg, 2),
            "max_profit": None if max_p is None else round(max_p, 2), "max_loss": None if max_l is None else round(max_l, 2),
            "unlimited_loss": unl_loss, "unlimited_profit": unl_profit, "breakevens": bes,
            "reward_risk": None if rr is None else round(rr, 2),
            "pop_model": round(pop_m * 100, 1), "ev_model": round(ev_m, 2),
            "pop_hist": None if pop_h is None else round(pop_h * 100, 1), "ev_hist": None if ev_h is None else round(ev_h, 2),
            "cvar5": None if cvar is None else round(cvar, 2),
            "greeks": {"delta": round(delta * lot, 1), "theta": round(theta * lot, 1), "vega": round(vega * lot, 1)},
            "curve": curve}


def context(chain, lot=None):
    """Market inputs shared by every evaluation on this chain: forward, time
    to expiry, per-strike quotes + IV/greeks, and the two outcome
    distributions (lognormal on ATM IV, and NIFTY's real moves)."""
    import manual_trades
    lot = lot or manual_trades.LOT_SIZE
    rows = chain["strikes"]
    spot, exp_ts = chain["spot"], chain.get("resolved_expiry_ts")
    T = (exp_ts - time.time()) / (365 * 86400) if exp_ts else 0
    if T <= 0:
        raise ValueError("expiry already over")
    near = sorted(rows, key=lambda r: abs(r["strike"] - spot))[:6]
    fwd = g.synthetic_forward([(r["strike"], (r["ce"] or {}).get("ltp"), (r["pe"] or {}).get("ltp")) for r in near]) or spot
    q = _quotes(chain, fwd, T)
    atm_k = int(round(fwd / STEP) * STEP)
    ivs = [q[(atm_k, t)]["iv"] for t in ("CE", "PE") if (atm_k, t) in q and q[(atm_k, t)]["iv"]]
    sigma = sum(ivs) / len(ivs) if ivs else 0.15
    grid = np.arange(math.floor(fwd * (1 - GRID_PCT) / GRID_STEP) * GRID_STEP, fwd * (1 + GRID_PCT), GRID_STEP)
    # lognormal (risk-neutral, forward-centred) probability of each grid cell
    edges = np.concatenate([[grid[0] - GRID_STEP / 2], grid + GRID_STEP / 2])
    sd = sigma * math.sqrt(T)
    z = (np.log(edges / fwd) + 0.5 * sd * sd) / sd
    cdf = 0.5 * (1 + np.array([math.erf(x / math.sqrt(2)) for x in z]))
    dens_model = np.diff(cdf)
    dens_model[0] += cdf[0]
    dens_model[-1] += 1 - cdf[-1]
    # historical: NIFTY's real moves over the same trading minutes
    mins = trading_minutes_left(exp_ts)
    r = _hist_returns(mins)
    dens_hist = None
    if r is not None and len(r):
        vals = np.clip(fwd * np.exp(r), grid[0], grid[-1])
        dens_hist = np.bincount(np.clip(np.searchsorted(edges, vals) - 1, 0, len(grid) - 1), minlength=len(grid)) / len(vals)
    return {"q": q, "fwd": fwd, "T": T, "lot": lot, "grid": grid, "dens_model": dens_model, "dens_hist": dens_hist,
            "sigma": sigma, "spot": spot, "exp_ts": exp_ts, "atm_k": atm_k, "sd": sd, "mins": mins, "r": r}


def evaluate_legs(ctx, legs, name="Position", mid=True):
    """Metrics of any position on this chain (1 lot per leg): legs = [(side, 'CE'/'PE', strike), ...]."""
    c = {"name": name, "category": "", "direction": "", "tpl": "", "legs": legs}
    return _evaluate(c, ctx["q"], ctx["fwd"], ctx["T"], ctx["lot"], ctx["grid"], ctx["dens_model"], ctx["dens_hist"],
                     ctx["sigma"], mid=mid)


def _trade_dates(exp_ts):
    """NSE trading days from today (while it is still open) up to the day
    before expiry -- the "by" choices for My View."""
    from datetime import datetime, timedelta
    import market_calendar as mc
    now = mc.now_ist()
    exp_d = datetime.fromtimestamp(float(exp_ts), tz=now.tzinfo).date()
    d = now.date() if now.time() < mc.MARKET_CLOSE else now.date() + timedelta(days=1)
    out = []
    while d < exp_d:
        if mc.is_trading_day(d):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def build(chain, stats=None, rng=None, view=None, lot=None):
    ctx = context(chain, lot)
    q, fwd, T, lot, grid = ctx["q"], ctx["fwd"], ctx["T"], ctx["lot"], ctx["grid"]
    dens_model, dens_hist, sigma, spot, exp_ts = ctx["dens_model"], ctx["dens_hist"], ctx["sigma"], ctx["spot"], ctx["exp_ts"]
    atm_k, sd, mins, r = ctx["atm_k"], ctx["sd"], ctx["mins"], ctx["r"]
    ideas = []
    tdte, tslot = track_context(exp_ts)
    for c in _candidates(q, fwd, rng):
        e = _evaluate(c, q, fwd, T, lot, grid, dens_model, dens_hist, sigma)
        if e:
            e["track"] = _track_for(c["tpl"], tdte, tslot)
            if e["track"]:
                e["track"]["similar"] = _similar(c["tpl"], tdte, tslot, sigma * 100)
            ideas.append(e)
    # environment + fit
    iv_rank = (stats or {}).get("iv_rank")
    verdict = (view or {}).get("verdict") or "NEUTRAL"
    vdir = "bullish" if "BULL" in verdict else "bearish" if "BEAR" in verdict else "neutral"
    regime = "high" if iv_rank is not None and iv_rank >= 50 else "low" if iv_rank is not None and iv_rank < 25 else "normal"
    for e in ideas:
        dir_ok = e["direction"] == vdir or (e["direction"] == "neutral" and vdir == "neutral")
        vol_ok = (e["credit"] and regime != "low") or (not e["credit"] and regime != "high")
        e["fit"] = {"direction": dir_ok, "iv": vol_ok, "both": dir_ok and vol_ok}
        risk = -e["max_loss"] if e["max_loss"] is not None else None
        e["ev_per_risk"] = round(e["ev_model"] / risk, 3) if risk else None
    env = {"spot": round(spot, 2), "forward": round(fwd, 2), "atm": atm_k, "atm_iv": round(sigma * 100, 2),
           "iv_rank": iv_rank, "iv_regime": regime, "verdict": verdict, "view_direction": vdir,
           "expiry_ts": exp_ts, "days_to_expiry": round(T * 365, 2), "trading_minutes_left": mins,
           "expected_move": round(fwd * sd, 1), "lot": lot, "trade_dates": _trade_dates(exp_ts),
           "range": {k: rng.get(k) for k in ("upper", "lower", "source", "ce_leader", "pe_leader")} if rng else None,
           "hist_samples": int(len(r)) if r is not None else 0,
           "track_dte": tdte, "track_slot": tslot, "track_ready": _track_table() is not None}
    return {"env": env, "ideas": ideas, "ctx": ctx}
