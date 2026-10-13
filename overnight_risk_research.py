"""overnight_risk_research.py -- the plan's A/B trades held overnight near their
break-even: does (X) closing at 15:25 when spot is near / past a break-even, or
(Y) buying a cheap far-OTM wing for the night (sold at 09:20 next day), beat
plain holding? (user's question, 13 Oct 2026: on 12 Oct all four trades sat near
their lower break-even into the close)

Trades: every IV-rule sell day (09:30 ATM IV >= threshold) --
  A  this week's expiry, 1-4 trading days left, held to expiry
  B  on the day before expiry, next week's expiry, closed one minute after the
     first check (09:30 / 12:00 / 14:30) where IV reads normal, else expiry
  straddle ATM / strangle 0.15 delta, 09:30 entry, 1 lot, charges, 0.5 pt slip per leg.
Every night the position is held (15:25 -> next 09:20):
  X_b    close at 15:25 if spot is within b% of a break-even or past it (b = 0, 0.25, 0.5)
  Y_put  buy the put ~w% below spot (w = 1.5, 2.5) at 15:25, sell it 09:20 next day
  Y_both put and call ~2% away
Output: results/overnight_risk/trades.parquet + summary printed.

Usage:  python overnight_risk_research.py
"""
import os
import sys

import numpy as np
import pandas as pd

import ashish_rules_bt as AR
import charges
import paths
import sell_rules as SR

OUT = os.path.join(paths.BASE_DIR, "results", "overnight_risk")
LOT, SLIP, YR = AR.LOT, AR.SLIP, AR.YR
BANDS = (0.0, 0.25, 0.5)
WINGS = {"Y_put1.5": [("PE", -0.015)], "Y_put2.5": [("PE", -0.025)], "Y_both2": [("PE", -0.02), ("CE", 0.02)]}


def _ch(side, p):
    return charges.order_charges(side, p, LOT) if p > 0 else 0.0


class Books:
    def __init__(self):
        self.c = {}

    def get(self, exp):
        if exp not in self.c:
            if len(self.c) >= 3:
                self.c.pop(min(self.c))
            self.c[exp] = AR.Book(exp)
        return self.c[exp]


def entries():
    h = SR.history().dropna(subset=["thr"]).copy()
    h["normal"] = h.atm_iv < h.thr
    state = {(r.day, r.slot): bool(r.normal) for r in h.itertuples()}
    man = AR.S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live"))
    out = []
    for r in h[(h.slot == "09:30") & ~h.normal].itertuples():
        if 1 <= r.dte <= 4 and r.expiry in exps:
            out.append(("A", r.day, r.expiry))
        if r.dte == 1:
            nxt = [x for x in exps if x > r.expiry]
            if nxt and (pd.Timestamp(nxt[0]) - pd.Timestamp(r.day)).days <= 12:
                out.append(("B", r.day, nxt[0]))
    return sorted(out, key=lambda x: (x[2], x[1])), state


def b_exit_ts(day, exp, state, days, close_ts):
    """B: one minute after the first check after entry where IV reads normal."""
    for d in [x for x in days if day <= x <= exp]:
        for slot in SR.SLOTS:
            t = AR._ts(d, slot)
            if t > AR._ts(day, "09:30") and t < close_ts and state.get((d, slot)) is True:
                return t + 60
    return None


def run():
    ents, state = entries()
    spot = AR.Spot()
    days = list(spot.daily.index)
    books = Books()
    rows = []
    for n, (setup, day, exp) in enumerate(ents):
        b = books.get(exp)
        t0 = AR._ts(day, "09:30")
        i0 = b.idx(t0)
        if i0 < 0:
            continue
        sp0 = spot.at(t0)
        q = b.quotes(i0, sp0, since=int(np.searchsorted(b.ts, AR._ts(day, "09:15"))))
        f = b.fwd(q, sp0)
        T = (b.close_ts - t0) / YR
        if not f or T <= 0:
            continue
        exit_t = b_exit_ts(day, exp, state, days, b.close_ts) if setup == "B" else None
        settle = spot.settle(exp)
        for strat, target in (("straddle", None), ("strangle", 0.15)):
            legs = AR.pick(b, q, f, T, target)
            if not legs:
                continue
            keys = [(k, t) for t, k in legs]
            ent = [b.px[x][i0] for x in keys]
            credit = sum(ent) - SLIP * len(keys)
            ch_in = sum(_ch("SELL", p) for p in ent)
            kp = next(k for t, k in legs if t == "PE")
            kc = next(k for t, k in legs if t == "CE")
            be_lo, be_hi = kp - credit, kc + credit

            def close_at(i):
                px = [b.px[x][i] for x in keys]
                return (credit - sum(px) - SLIP * len(keys)) * LOT - ch_in - sum(_ch("BUY", p + SLIP) for p in px)

            intr = sum(max(0.0, settle - k) if t == "CE" else max(0.0, k - settle) for t, k in legs)
            base = close_at(b.idx(exit_t)) if exit_t else (credit - intr) * LOT - ch_in
            # nights the position is held under the rule
            nights = []
            for d in [x for x in days if day <= x < exp]:
                t1525 = AR._ts(d, "15:25")
                if exit_t and exit_t <= t1525:
                    break
                k = days.index(d)
                if k + 1 >= len(days):
                    break
                nights.append((d, days[k + 1], b.idx(t1525), b.idx(AR._ts(days[k + 1], "09:20"))))
            rec = {"setup": setup, "strat": strat, "day": day, "expiry": exp, "credit_rs": credit * LOT, "base": base, "nights": len(nights)}
            gaps, near_flags = [], []
            for d, d2, i1, i2 in nights:
                v1 = sum(b.px[x][i1] for x in keys)
                v2 = sum(b.px[x][i2] for x in keys)
                gaps.append((v2 - v1) * LOT)
                s1 = spot.at(AR._ts(d, "15:25"))
                near_flags.append(min(s1 - be_lo, be_hi - s1) / s1 * 100)
            rec["worst_gap"] = max(gaps) if gaps else 0.0          # biggest overnight jump in what it costs to buy back
            rec["min_be_dist"] = min(near_flags) if near_flags else np.nan
            for bd in BANDS:
                rs = base
                for (d, d2, i1, i2), dist in zip(nights, near_flags):
                    if dist <= bd:
                        rs = close_at(i1)
                        break
                rec[f"X_{bd}"] = rs
            for name, spec in WINGS.items():
                cost = 0.0
                for d, d2, i1, i2 in nights:
                    s1 = spot.at(AR._ts(d, "15:25"))
                    qq = b.quotes(i1, s1, since=int(np.searchsorted(b.ts, AR._ts(d, "09:15"))), width=3000)
                    for typ, w in spec:
                        want = s1 * (1 + w)
                        ks = [k for (k, t) in qq if t == typ]
                        if not ks:
                            continue
                        kk = min(ks, key=lambda k: abs(k - want))
                        buy = b.px[(kk, typ)][i1] + SLIP
                        sell = max(0.0, b.px[(kk, typ)][i2] - SLIP)
                        cost += (sell - buy) * LOT - _ch("BUY", buy) - _ch("SELL", sell)
                rec[name] = base + cost
                rec[name + "_cost"] = cost
            rows.append(rec)
        if n % 40 == 0:
            print(f"\r{n}/{len(ents)} {exp}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def _dd(x):
    c = np.cumsum(x)
    return float((c - np.maximum.accumulate(np.r_[0.0, c][1:])).min()) if len(c) else 0.0


def summarize(df):
    cols = ["base"] + [f"X_{b}" for b in BANDS] + list(WINGS)
    out = []
    for (setup, strat), g in df.sort_values("day").groupby(["setup", "strat"]):
        for c in cols:
            x = g[c].to_numpy()
            h1, h2 = g[g.day < "2024-04-01"][c], g[g.day >= "2024-04-01"][c]
            out.append({"trade": f"{setup} {strat}", "variant": c, "n": len(x), "avg": x.mean(), "win%": (x > 0).mean() * 100,
                        "worst": x.min(), "total": x.sum(), "max_dd": _dd(x), "avg_1st": h1.mean(), "avg_2nd": h2.mean()})
    return pd.DataFrame(out).round(0)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    df = run()
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, "trades.parquet"), index=False)
    s = summarize(df)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(df.day.min(), df.day.max(), len(df))
    print(s.to_string(index=False))
    held = df[df.nights > 0]
    print("\ntrades held >= 1 night:", len(held), " of", len(df))
    print("worst overnight jump (Rs, buy-back cost up):", held.groupby(["setup", "strat"]).worst_gap.max().round(0).to_dict())
    print("X triggers (share of trades closed early):", {b: round((df[f'X_{b}'] != df.base).mean() * 100, 1) for b in BANDS})
