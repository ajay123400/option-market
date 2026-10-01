"""be_approaches.py -- breakeven-based positional approaches, replayed on one
expiry week (step-by-step walkthrough; not yet a 5-year backtest).

  A  centred straddle: sell a straddle at the 50-pt strike nearest the range
     middle; when a recalculated range's middle is 50+ pts from the straddle
     strike, move both legs to the strike nearest the new middle.
  B  breakeven guard: start at the strikes nearest UPPER / LOWER. On a range
     change: (1) usual roll TOWARD the market (call down to the strike nearest
     the new UPPER / put up to the strike nearest the new LOWER); (2) if a
     range edge is within `guard` pts of that side's breakeven, roll that leg
     AWAY to the nearest strike that puts the breakeven `guard` pts beyond the
     edge again.
  C  balanced strangle: the CE/PE pair (CE above PE) whose breakevens are the
     same distance outside the range, closest to it; re-balance on every range
     change (roll only the legs whose strike changes).

Breakeven includes booked P&L from earlier rolls:
  credit = open legs' entry premium + booked points
  BE up  = CE strike + credit,  BE down = PE strike - credit
Leaders / range as in positional_bt.py (5-min checks, range recalculated only
on a leader change). Fills at the next 1-min bar's open, 0.5 pt slippage.
"""
import sys

import numpy as np
import pandas as pd

import simulator as S
from intraday_range_bt import Day, _nearest

SLIP = 0.5
STEP = 50


class Pos:
    def __init__(self):
        self.legs = {}      # "CE"/"PE" -> {"strike", "entry"}
        self.booked = 0.0
        self.log = []

    def credit(self):
        return sum(l["entry"] for l in self.legs.values()) + self.booked

    def be(self):
        c = self.credit()
        return self.legs["PE"]["strike"] - c, self.legs["CE"]["strike"] + c

    def open_pts(self, D, i):
        return sum(l["entry"] - (D.price(t, l["strike"], i, "close") or l["entry"]) for t, l in self.legs.items())

    def sell(self, t, k, D, i):
        p = D.price(t, k, i)
        self.legs[t] = {"strike": k, "entry": round(p - SLIP, 2)}
        return self.legs[t]["entry"]

    def buy(self, t, D, i):
        l = self.legs.pop(t)
        x = round(D.price(t, l["strike"], i) + SLIP, 2)
        self.booked += l["entry"] - x
        return l, x

    def roll(self, t, k, D, i, when, why):
        if D.price(t, k, i) is None or self.legs[t]["strike"] == k:
            return False
        old, x = self.buy(t, D, i)
        e = self.sell(t, k, D, i)
        self.log.append({"time": when, "action": f"{t} {old['strike']} -> {k}", "why": why,
                         "detail": f"bought back @{x} ({old['entry'] - x:+.2f}), sold @{e}"})
        return True


def balanced(D, i, U, L):
    best = None
    lo, hi = int(L // STEP) * STEP - 400, int(U // STEP) * STEP + 450
    for kc in range(lo, hi, STEP):
        c = D.price("CE", kc, i)
        if c is None:
            continue
        for kp in range(lo, kc, STEP):
            p = D.price("PE", kp, i)
            if p is None:
                continue
            cr = c + p - 2 * SLIP
            ou, od = kc + cr - U, L - (kp - cr)
            if ou < 0 or od < 0:
                continue
            s = abs(ou - od) * 3 + ou + od
            if best is None or s < best[0]:
                best = (s, kc, kp)
    return (best[1], best[2]) if best else None


def replay(exp, approach, guard=50):
    exps = sorted(e for e, r in S._manifest().items() if r.get("status") == "done")
    prev = exps[exps.index(exp) - 1]
    df = S._load_options(exp)
    df = df[df["date"].dt.hour * 60 + df["date"].dt.minute <= 930]
    days = [d for d in sorted(df.day.unique()) if pd.Timestamp(prev).date() < d <= pd.Timestamp(exp).date()]
    sp = S._load_spot()
    st, sc = sp.ts.to_numpy(), sp.close.to_numpy()
    P, L_, R = Pos(), None, None
    daily = []
    for day in days:
        D = Day(df[df.day == day], 100, 5, st, sc)
        for i in range(D.n - 1):
            if (D.mins[i] + 1) % 5 or D.mins[i] < 4:
                continue
            ld = D.leaders(i)
            if not ld:
                continue
            if L_ is not None and ld == L_:
                continue
            nR = D.rng(ld)
            if not nR:
                continue
            L_, R = ld, nR
            U, Lo, mid = R["upper"], R["lower"], (R["upper"] + R["lower"]) / 2
            when = f"{day} {D.hhmm[i + 1]}"
            e = i + 1
            spot = round(float(D.spot[i]), 1)
            if not P.legs:  # entry
                if approach == "A":
                    k = int(round(mid / STEP) * STEP)
                    kc = kp = k
                elif approach == "B":
                    kc, kp = _nearest(U, STEP, "CE"), _nearest(Lo, STEP, "PE")
                else:
                    kc, kp = balanced(D, e, U, Lo)
                P.sell("CE", kc, D, e)
                P.sell("PE", kp, D, e)
                P.log.append({"time": when, "action": f"SELL {kc} CE @{P.legs['CE']['entry']} + {kp} PE @{P.legs['PE']['entry']}",
                              "why": f"entry · range {Lo:.0f}–{U:.0f}", "detail": ""})
            else:
                before = len(P.log)
                if approach == "A":
                    k = P.legs["CE"]["strike"]
                    if abs(mid - k) >= 50:
                        nk = int(round(mid / STEP) * STEP)
                        P.roll("CE", nk, D, e, when, f"range middle {mid:.0f} is {mid - k:+.0f} from the straddle")
                        P.roll("PE", nk, D, e, when, "straddle moved with it")
                elif approach == "B":
                    ce, pe = P.legs["CE"]["strike"], P.legs["PE"]["strike"]
                    tc, tp = _nearest(U, STEP, "CE"), _nearest(Lo, STEP, "PE")
                    if tc < ce:
                        P.roll("CE", max(tc, pe - 100), D, e, when, "range shifted down -> call toward the market")
                    if tp > pe:
                        P.roll("PE", min(tp, P.legs["CE"]["strike"] + 100), D, e, when, "range shifted up -> put toward the market")
                    bd, bu = P.be()
                    if U > bu - guard:
                        k = P.legs["CE"]["strike"]
                        while k < U + 1000:
                            k += STEP
                            p = D.price("CE", k, e)
                            if p is None:
                                continue
                            # breakeven after this roll: booked changes by (entry - buyback), new entry added
                            cur = P.legs["CE"]
                            bb = D.price("CE", cur["strike"], e) + SLIP
                            cr = P.credit() - cur["entry"] + (cur["entry"] - bb) + (p - SLIP)
                            if k + cr >= U + guard:
                                P.roll("CE", k, D, e, when, f"UPPER {U:.0f} within {guard} of breakeven {bu:.0f} -> call away")
                                break
                    bd, bu = P.be()
                    if Lo < bd + guard:
                        k = P.legs["PE"]["strike"]
                        while k > Lo - 1000:
                            k -= STEP
                            p = D.price("PE", k, e)
                            if p is None:
                                continue
                            cur = P.legs["PE"]
                            bb = D.price("PE", cur["strike"], e) + SLIP
                            cr = P.credit() - cur["entry"] + (cur["entry"] - bb) + (p - SLIP)
                            if k - cr <= Lo - guard:
                                P.roll("PE", k, D, e, when, f"LOWER {Lo:.0f} within {guard} of breakeven {bd:.0f} -> put away")
                                break
                else:
                    b = balanced(D, e, U, Lo)
                    if b:
                        kc, kp = b
                        P.roll("CE", kc, D, e, when, "re-balance to the new range")
                        P.roll("PE", kp, D, e, when, "re-balance to the new range")
                if len(P.log) == before:
                    P.log.append({"time": when, "action": "—", "why": f"range {Lo:.0f}–{U:.0f} (leaders {ld[0]}/{ld[1]}) · no adjustment", "detail": ""})
            bd, bu = P.be()
            P.log[-1].update(range=f"{Lo:.0f}–{U:.0f}", nifty=spot, be=f"{bd:.0f}–{bu:.0f}",
                             pnl=round(P.booked + P.open_pts(D, e), 1))
        last = D.n - 1
        daily.append({"day": str(day), "close": round(float(D.spot[last]) if np.isfinite(D.spot[last]) else float("nan"), 1),
                      "pnl": round(P.booked + P.open_pts(D, last), 1),
                      "legs": f"{P.legs['CE']['strike']} CE / {P.legs['PE']['strike']} PE", "be": "{:.0f}–{:.0f}".format(*P.be())})
    # settle at expiry close
    e_end = int(pd.Timestamp(f"{exp} 23:59", tz=S.IST).timestamp())
    settle = float(sc[np.searchsorted(st, e_end, side="right") - 1])
    fin = P.booked + sum(l["entry"] - (max(0, settle - l["strike"]) if t == "CE" else max(0, l["strike"] - settle)) for t, l in P.legs.items())
    return P.log, daily, settle, round(fin, 2), S.lot_size_for(exp), sum(1 for x in P.log if "->" in x["action"])


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    exp = sys.argv[1] if len(sys.argv) > 1 else "2026-08-11"
    for ap in (sys.argv[2:] or ["A", "B", "C"]):
        log, daily, settle, fin, lot, n = replay(exp, ap)
        print(f"\n===== {ap} · expiry {exp} · lot {lot} =====")
        for x in log:
            print(f"{x['time']}  {x['action']:<28} {x.get('detail', ''):<42} | {x['why']} | NIFTY {x.get('nifty')} | range {x.get('range')} | BE {x.get('be')} | P&L {x.get('pnl')} pts")
        for d in daily:
            print(f"   end of {d['day']}: NIFTY {d['close']} · {d['legs']} · BE {d['be']} · P&L {d['pnl']} pts")
        print(f"   EXPIRY close {settle} -> final {fin} pts = Rs {fin * lot:,.0f} (before charges) · {n} leg rolls")
