"""ashish_rules_bt.py -- tests of the volatility-selling rules from the
Groww "Trading ki Baat" interview (Ashish Gupta, tastytrade style) on NIFTY
weekly options (data/hist1m), next to this project's IV rule.

 1. Exits: profit target (straddle 20%, strangle 35% of the credit), stop at
    a loss of 2x the credit, both, vs the IV-normal exit and hold-to-expiry.
 2. Delta-roll adjustment (16-delta strangle): when |call delta| and |put
    delta| differ by more than 0.25, roll the untested short to the strike
    that halves the gap (never past ATM); no extra lots.
 3. Tail hedge: buy one ~4-delta weekly put (3 and 5 delta too) at the start
    of every expiry week, held to expiry -- cost per year and what it paid in
    the worst weeks.
 4. Filters: IV vs realized vol (HV10 / HV30), and IV percentile >= 70, next
    to the IV rule (>= 60th percentile of the past 500 days).

Positions are marked every 15 minutes (09:15 ... 15:30); targets, stops, IV
checks and rolls act on those marks, filled at that minute's close +/- 0.5 pt.
Entries: 09:30. Set A = the current weekly expiry, 1-4 trading days left,
every day (green / normal kept apart). Set B = the next weekly expiry on days
the current one has 1 trading day left.
Output: results/ashish_rules/report.html
"""
import json
import math
import os
import pickle
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

import charges
import greeks as g
import manual_trades
import paths
import sell_rules as SR
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "ashish_rules")
LOT = manual_trades.LOT_SIZE
SLIP = 0.5
YR = 365 * 86400
STRATS = {"Short straddle ATM": None, "Short strangle 16Δ": 0.16}
TARGET = {"Short straddle ATM": 0.20, "Short strangle 16Δ": 0.35}
VARIANTS = {
    "hold": dict(),
    "target": dict(target=True),
    "stop2x": dict(stop=2.0),
    "target+stop": dict(target=True, stop=2.0),
    "ivexit": dict(ivexit=True),
    "ivexit+stop": dict(ivexit=True, stop=2.0),
    "roll": dict(roll=True),
    "roll+stop": dict(roll=True, stop=2.0),
}
LABEL = {"hold": "Hold to expiry", "target": "Profit target (straddle 20% / strangle 35%)", "stop2x": "Stop: loss = 2× credit",
         "target+stop": "Target + 2× stop (Ashish)", "ivexit": "Exit when IV turns normal (09:30 check)",
         "ivexit+stop": "IV-normal exit + 2× stop", "roll": "Delta-roll adjustment, hold", "roll+stop": "Delta-roll + 2× stop"}


def _ts(day, hm):
    return int(pd.Timestamp(f"{day} {hm}", tz=S.IST).timestamp())


class Book:
    """1-min forward-filled closes for one expiry's contracts."""

    def __init__(self, exp):
        df = S._load_options(exp)
        mins = df["date"].dt.hour * 60 + df["date"].dt.minute
        exp_d = pd.Timestamp(exp).date()
        df = df[(mins <= 15 * 60 + 40) & (df["day"] >= exp_d - timedelta(days=16))]
        self.exp, self.exp_d = exp, exp_d
        self.ts = np.sort(df["ts"].unique())
        self.px, self.last = {}, {}
        n = len(self.ts)
        for (t, k), gdf in df.groupby(["type", "strike"], sort=False):
            idx = np.searchsorted(self.ts, gdf["ts"].to_numpy())
            c = np.full(n, np.nan)
            c[idx] = gdf["close"].to_numpy()
            lt = np.full(n, -1)
            lt[idx] = idx
            self.px[(int(k), t)] = pd.Series(c).ffill().to_numpy()
            self.last[(int(k), t)] = np.maximum.accumulate(lt)
        self.close_ts = SR._exp_close_ts(exp)
        loc = self.ts + 19800                                   # IST seconds
        self.hhmm = ((loc // 3600) % 24) * 100 + (loc // 60) % 60
        self.date = pd.to_datetime(loc, unit="s").strftime("%Y-%m-%d").to_numpy()
        self.grid15 = np.flatnonzero((self.hhmm % 100) % 15 == 0)

    def idx(self, tsv):
        return int(np.searchsorted(self.ts, tsv, side="right") - 1)

    def quotes(self, i, spot, since=None, width=1600):
        out = {}
        for key, arr in self.px.items():
            p = arr[i]
            if not np.isfinite(p) or abs(key[0] - spot) > width:
                continue
            if since is not None and self.last[key][i] < since:
                continue
            out[key] = float(p)
        return out

    def fwd(self, q, spot):
        ks = sorted({k for k, _ in q})
        near = sorted([(k, q[(k, "CE")], q[(k, "PE")]) for k in ks if (k, "CE") in q and (k, "PE") in q], key=lambda x: abs(x[0] - spot))[:6]
        return g.synthetic_forward(near) if near else None


class Spot:
    def __init__(self):
        sp = S._load_spot()
        self.st, self.sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
        d = sp.groupby(sp.index.date).close.last()
        self.daily = pd.Series(d.to_numpy(), index=[x.isoformat() for x in d.index])

    def at(self, tsv):
        return float(self.sc[max(0, np.searchsorted(self.st, tsv, side="right") - 1)])

    def settle(self, exp):
        return float(self.sc[np.searchsorted(self.st, _ts(exp, "23:59"), side="right") - 1])


def pick(book, q, fwd, T, target):
    if target is None:
        atm = int(round(fwd / 50) * 50)
        return [("CE", atm), ("PE", atm)] if (atm, "CE") in q and (atm, "PE") in q else None
    legs = []
    for t in ("CE", "PE"):
        best = None
        for (k, ty), p in q.items():
            if ty != t or (t == "CE" and k <= fwd) or (t == "PE" and k >= fwd):
                continue
            _, d = SR._iv_delta(p, fwd, k, T, t == "CE")
            if d is not None and (best is None or abs(abs(d) - target) < best[0]):
                best = (abs(abs(d) - target), k)
        if best is None:
            return None
        legs.append((t, best[1]))
    return legs


def strike_for_delta(q, fwd, T, t, want):
    best = None
    for (k, ty), p in q.items():
        if ty != t or (t == "CE" and k < fwd - 25) or (t == "PE" and k > fwd + 25):
            continue
        _, d = SR._iv_delta(p, fwd, k, T, t == "CE")
        if d is not None and (best is None or abs(abs(d) - want) < best[0]):
            best = (abs(abs(d) - want), k)
    return None if best is None else best[1]


def simulate(book, spot, entry_ts, legs, strat, state, v):
    """One position under one variant. Returns (rs, days held, reason, n_rolls)."""
    i0 = book.idx(entry_ts)
    ent = {(k, t): book.px[(k, t)][i0] for t, k in legs}
    credit = sum(ent.values()) - SLIP * len(legs)                  # per unit, net of slippage
    cash = credit
    ch = sum(charges.order_charges("SELL", ent[(k, t)], LOT) for t, k in legs)
    open_legs = list(legs)
    rolls = 0
    g0 = np.searchsorted(book.grid15, i0 + 1)
    grid = [i for i in book.grid15[g0:] if book.ts[i] < book.close_ts]
    for i in grid:
        tsv = int(book.ts[i])
        val = sum(book.px[(k, t)][i] for t, k in open_legs)
        pnl = cash - val - SLIP * len(open_legs)                   # if closed now
        close_now = None
        if v.get("stop") and pnl <= -v["stop"] * credit:
            close_now = "stop"
        elif v.get("target") and pnl >= TARGET[strat] * credit:
            close_now = "target"
        elif v.get("ivexit") and book.hhmm[i] == 930 and state.get((book.date[i], "09:30")) is True:
            close_now = "ivexit"
        if close_now:
            for t, k in open_legs:
                p = book.px[(k, t)][i] + SLIP
                ch += charges.order_charges("BUY", p, LOT)
            return (pnl * LOT - ch, (tsv - entry_ts) / 86400, close_now, rolls)
        if v.get("roll") and len(open_legs) == 2:
            sp = spot.at(tsv)
            q = book.quotes(i, sp, width=2000)
            f = book.fwd(q, sp)
            T = (book.close_ts - tsv) / YR
            if f and T > 0:
                d = {}
                for t, k in open_legs:
                    _, dl = SR._iv_delta(book.px[(k, t)][i], f, k, T, t == "CE")
                    d[t] = abs(dl) if dl is not None else None
                if d.get("CE") is not None and d.get("PE") is not None and abs(d["CE"] - d["PE"]) > 0.25:
                    tested = "CE" if d["CE"] > d["PE"] else "PE"
                    untested = "PE" if tested == "CE" else "CE"
                    want = min(0.5, d[tested] - (d[tested] - d[untested]) / 2)
                    old_k = next(k for t, k in open_legs if t == untested)
                    new_k = strike_for_delta(q, f, T, untested, want)
                    if new_k and new_k != old_k and (new_k, untested) in book.px:
                        buy = book.px[(old_k, untested)][i] + SLIP
                        sell = book.px[(new_k, untested)][i] - SLIP
                        cash += sell - buy
                        ch += charges.order_charges("BUY", buy, LOT) + charges.order_charges("SELL", sell, LOT)
                        open_legs = [(t, k) for t, k in open_legs if t != untested] + [(untested, new_k)]
                        rolls += 1
    settle = spot.settle(book.exp)
    intr = sum(max(0.0, settle - k) if t == "CE" else max(0.0, k - settle) for t, k in open_legs)
    return ((cash - intr) * LOT - ch, (book.close_ts - entry_ts) / 86400, "expiry", rolls)


def collect():
    H = SR.history()
    h = H.dropna(subset=["thr"]).copy()
    h["normal"] = h.atm_iv < h.thr
    state = {(r.day, r.slot): bool(r.normal) for r in h.itertuples()}
    rows930 = h[h.slot == "09:30"]
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live"))
    spot = Spot()
    entries = []
    for r in rows930.itertuples():
        entries.append(("A", r.day, r.expiry, not r.normal, int(r.dte)))
        if r.dte == 1:
            nxt = [x for x in exps if x > r.expiry]
            if nxt and (pd.Timestamp(nxt[0]) - pd.Timestamp(r.day)).days <= 12:
                entries.append(("B", r.day, nxt[0], not r.normal, None))
    entries.sort(key=lambda x: (x[2], x[1]))
    out, book = [], None
    for n, (st_, day, exp, green, dte) in enumerate(entries):
        if book is None or book.exp != exp:
            book = Book(exp)
        t_ent = _ts(day, "09:30")
        i = book.idx(t_ent)
        if i < 0:
            continue
        sp = spot.at(t_ent)
        day_start = int(np.searchsorted(book.ts, _ts(day, "09:15")))
        q = book.quotes(i, sp, since=day_start)
        f = book.fwd(q, sp)
        T = (book.close_ts - t_ent) / YR
        if not f or T <= 0:
            continue
        for strat, target in STRATS.items():
            legs = pick(book, q, f, T, target)
            if not legs:
                continue
            rec = {"set": st_, "day": day, "expiry": exp, "green": green, "dte": dte, "strat": strat}
            for vn, v in VARIANTS.items():
                if v.get("roll") and target is None:
                    continue
                rs, held, why, rolls = simulate(book, spot, t_ent, legs, strat, state, v)
                rec.update({f"{vn}_rs": rs, f"{vn}_held": held, f"{vn}_why": why, f"{vn}_rolls": rolls})
            out.append(rec)
        if n % 100 == 0:
            print(f"\r{n}/{len(entries)} {exp}", end="", flush=True)
    print()
    return pd.DataFrame(out)


def hedge():
    """Buy one ~3/4/5-delta put of the current weekly expiry at 09:30 on the
    first trading day of each expiry week; hold to expiry."""
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live"))
    spot = Spot()
    days = list(spot.daily.index)
    rows = []
    for a, b in zip(exps, exps[1:]):
        if b < "2021-10-01":
            continue
        d0 = next((d for d in days if d > a), None)
        if not d0 or d0 >= b:
            continue
        book = Book(b)
        t = _ts(d0, "09:30")
        i = book.idx(t)
        if i < 0:
            continue
        sp = spot.at(t)
        q = book.quotes(i, sp, since=int(np.searchsorted(book.ts, _ts(d0, "09:15"))), width=3000)
        f = book.fwd(q, sp)
        T = (book.close_ts - t) / YR
        if not f or T <= 0:
            continue
        settle = spot.settle(b)
        rec = {"week": b, "start": d0, "spot": sp, "settle": settle, "move": (settle / sp - 1) * 100}
        for dl in (0.03, 0.04, 0.05):
            k = strike_for_delta(q, f, T, "PE", dl)
            if not k:
                continue
            p = q[(k, "PE")] + SLIP
            pay = max(0.0, k - settle) - SLIP if k > settle else 0.0
            ch = charges.order_charges("BUY", p, LOT) + (charges.order_charges("SELL", pay, LOT) if pay > 0 else 0)
            rec[f"d{int(dl * 100)}_k"] = k
            rec[f"d{int(dl * 100)}_cost"] = p * LOT
            rec[f"d{int(dl * 100)}_rs"] = (pay - p) * LOT - ch
            # the same put's best mark before expiry (a crash can be cashed in early)
            j0, j1 = i + 1, book.idx(book.close_ts)
            if j1 > j0:
                rec[f"d{int(dl * 100)}_peak"] = float(np.nanmax(book.px[(k, "PE")][j0:j1 + 1])) * LOT
        rows.append(rec)
        print(f"\rhedge {b}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def filters():
    """IV rule vs IV/HV and IVP>=70 on the template backtest (09:30, 1-4 days left, held)."""
    H = SR.history()
    h = H[H.slot == "09:30"].copy().sort_values("day")
    sp = Spot()
    r = np.log(sp.daily).diff()
    hv10 = (r.rolling(10).std() * math.sqrt(252) * 100).shift(1)
    hv30 = (r.rolling(30).std() * math.sqrt(252) * 100).shift(1)
    h["hv10"], h["hv30"] = h.day.map(hv10), h.day.map(hv30)
    h["r10"], h["r30"] = h.atm_iv / h.hv10, h.atm_iv / h.hv30
    roll = lambda s, q: s.shift(1).rolling(500, min_periods=120).quantile(q)
    h["thr60"], h["thr70"] = roll(h.atm_iv, 0.6), roll(h.atm_iv, 0.7)
    h["r30thr"], h["r10thr"] = roll(h.r30, 0.6), roll(h.r10, 0.6)
    h = h.dropna(subset=["thr60", "r30thr", "r10thr"])
    t = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet"))
    t = t[(t.slot == "09:30") & (t.dte >= 1) & (t.dte <= 4)].drop(columns=["atm_iv"])
    t = t.merge(h[["day", "expiry", "atm_iv", "thr60", "thr70", "r10", "r30", "r10thr", "r30thr"]], on=["day", "expiry"])
    rules = {
        "Every day": lambda x: np.ones(len(x), bool),
        "IV rule (IV ≥ 60th pct)": lambda x: x.atm_iv >= x.thr60,
        "IVP ≥ 70": lambda x: x.atm_iv >= x.thr70,
        "IV/HV30 high (≥ 60th pct)": lambda x: x.r30 >= x.r30thr,
        "IV/HV10 high (≥ 60th pct)": lambda x: x.r10 >= x.r10thr,
        "IV/HV30 ≥ 1.2": lambda x: x.r30 >= 1.2,
        "IV rule AND IV/HV30 high": lambda x: (x.atm_iv >= x.thr60) & (x.r30 >= x.r30thr),
        "IV rule AND IV/HV10 high": lambda x: (x.atm_iv >= x.thr60) & (x.r10 >= x.r10thr),
        "IV rule but IV/HV30 low": lambda x: (x.atm_iv >= x.thr60) & (x.r30 < x.r30thr),
    }
    out = []
    for tpl in ("Short straddle|ATM", "Short strangle|Δ0.15", "Short strangle|Δ0.20", "Iron condor|Δ0.15|w200"):
        x = t[t.tpl == tpl]
        for name, f in rules.items():
            y = x[np.asarray(f(x))]
            v = y.hold_rs
            out.append({"tpl": tpl, "rule": name, "n": int(len(y)), "share": round(len(y) / len(x) * 100),
                        "avg": round(float(v.mean())) if len(y) else None, "p5": round(float(v.quantile(0.05))) if len(y) > 5 else None,
                        "a": round(float(v[y.expiry < "2024"].mean())) if (y.expiry < "2024").any() else None,
                        "b": round(float(v[y.expiry >= "2024"].mean())) if (y.expiry >= "2024").any() else None,
                        "total": round(float(v.sum()))})
    return out


def summarize(d):
    out = []
    for st_ in ("A", "B"):
        for strat in STRATS:
            for green in (True, False):
                x = d[(d.set == st_) & (d.strat == strat) & (d.green == green)]
                if x.empty:
                    continue
                for vn in VARIANTS:
                    if f"{vn}_rs" not in x or x[f"{vn}_rs"].isna().all():
                        continue
                    v = x[f"{vn}_rs"].dropna()
                    y = x.loc[v.index]
                    eq = v.cumsum()
                    out.append({"set": st_, "strat": strat, "green": green, "v": vn, "n": int(len(v)), "avg": round(float(v.mean())),
                                "win": round(float((v > 0).mean() * 100), 1), "p5": round(float(v.quantile(0.05))), "worst": round(float(v.min())),
                                "dd": round(float((eq - eq.cummax()).min())), "held": round(float(y[f"{vn}_held"].mean()), 1),
                                "early": round(float((y[f"{vn}_why"] != "expiry").mean() * 100)),
                                "rolls": round(float(y[f"{vn}_rolls"].mean()), 2),
                                "a": round(float(v[y.day < "2024"].mean())), "b": round(float(v[y.day >= "2024"].mean()))})
    return out


def hedge_summary(hd, d):
    out = {}
    yrs = hd.week.str[:4]
    for dl in (3, 4, 5):
        c = f"d{dl}_rs"
        if c not in hd:
            continue
        x = hd.dropna(subset=[c])
        per_year = x.groupby(x.week.str[:4])[c].sum()
        notional = (x.spot * LOT).mean()
        out[dl] = {"weeks": int(len(x)), "avg_cost": round(float(x[f"d{dl}_cost"].mean())), "total": round(float(x[c].sum())),
                   "per_year": {k: round(float(v)) for k, v in per_year.items()},
                   "cost_pct_notional_yr": round(float(-x[c].sum() / (len(x) / 52) / notional * 100), 2),
                   "paid_weeks": int((x[c] > 0).sum()),
                   "best": x.nlargest(5, c)[["week", "move", c, f"d{dl}_peak"]].round(1).to_dict("records")}
    # worst weeks of the green-day straddle (set A) and what 1 hedge per short lot paid that week
    s = d[(d.set == "A") & (d.strat == "Short straddle ATM") & d.green]
    wk = s.groupby("expiry").agg(pnl=("hold_rs", "sum"), lots=("hold_rs", "size")).reset_index()
    wk = wk.merge(hd[["week", "move", "d4_rs"]], left_on="expiry", right_on="week", how="left")
    wk["hedged"] = wk.pnl + wk.d4_rs.fillna(0) * wk.lots
    worst = wk.nsmallest(10, "pnl")[["expiry", "lots", "move", "pnl", "d4_rs", "hedged"]].round(1).to_dict("records")
    tot = {"unhedged": round(float(wk.pnl.sum())), "hedged": round(float(wk.hedged.sum())),
           "worst_unhedged": round(float(wk.pnl.min())), "worst_hedged": round(float(wk.hedged.min()))}
    return out, worst, tot


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    cp = os.path.join(OUT, "positions.parquet")
    if os.path.exists(cp) and "--fresh" not in sys.argv:
        d = pd.read_parquet(cp)
    else:
        d = collect()
        d.to_parquet(cp, index=False)
    hp = os.path.join(OUT, "hedge.parquet")
    if os.path.exists(hp) and "--fresh" not in sys.argv:
        hd = pd.read_parquet(hp)
    else:
        hd = hedge()
        hd.to_parquet(hp, index=False)
    summ = summarize(d)
    for r in summ:
        print(f"{r['set']} {r['strat']:20s} {'G' if r['green'] else 'n'} {r['v']:12s} n{r['n']:4d} avg {r['avg']:6d} win {r['win']:5.1f} p5 {r['p5']:7d} dd {r['dd']:8d} held {r['held']:4.1f} early {r['early']:3d}% rolls {r['rolls']:4.2f} A {r['a']:6d} B {r['b']:6d}")
    hs, worst, tot = hedge_summary(hd, d)
    print(json.dumps(hs, indent=1, default=str)[:3000])
    print(worst, tot)
    fl = filters()
    for r in fl:
        print(f"{r['tpl']:24s} {r['rule']:28s} n{r['n']:4d} ({r['share']:3d}%) avg {r['avg']} p5 {r['p5']} A {r['a']} B {r['b']}")
    pickle.dump({"summ": summ, "hedge": hs, "worst": worst, "tot": tot, "filters": fl}, open(os.path.join(OUT, "results.pkl"), "wb"))
