"""range_strategy.py -- the user's own weekly "leader range" option-selling
method, automated as a paper strategy (no real orders).

The method (as the user described it, 2026-09-24, walked through step by
step on the 09-Sep -> 15-Sep 2026 expiry):

 1. Cycle start = first trading day after the previous weekly expiry
    (normally Wednesday). After the first 5-min candle (09:15-09:20) the
    highest-volume CALL and PUT strikes are the leaders (100-pt strikes only).
 2. Range: UPPER = CE leader strike + PE leader's first-candle high
           LOWER = PE leader strike - CE leader's first-candle high
 3. Entry: SELL the round strike nearest UPPER (call) and nearest LOWER
    (put), 1 lot each, no hedge.
 4. Leaders are tracked on each DAY's own volume (reset at 09:15). Whenever
    a leader CHANGES (intraday, or at a new day's first candle) the range is
    recomputed from the new leaders' first-candle highs of THAT day. No
    rolls on expiry day (only reached when the day before is a holiday):
    there the position is only closed (target / stop / time).
 5. Adjustments (roll = buy back the leg, sell the new strike):
      - call: roll DOWN when the new range's call strike is lower (range
        shifted down -> "market may fall, take more call premium"); roll UP
        only if the call has gone in the money.
      - put: roll UP when the new range's put strike is higher; roll DOWN
        only if the put has gone in the money.
 6. Stop-loss: whole trade (booked + open) loses HALF the premium collected
    at entry -> close everything, wait for next week.
 7. Profit target: net P&L (after charges) reaches 70% of the entry premium
    -> close everything.
 8. Time exit: 15:00 on the trading day before expiry (Monday). If that day
    is a holiday, the position is carried into expiry day and closed by
    15:00 there (the target/stop still apply before that).

Everything is recomputed by REPLAYING exchange 5-min candles from the cycle
start, so a restart or a late start reaches exactly the same state, and the
live runner and the backtest run the identical code.
"""
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta

import charges as charges_mod
import greeks as greeks_mod
import market_calendar as mc
import paths

BASE = paths.BASE_DIR
STATE_PATH = os.path.join(BASE, "results", "range_state.json")
DIARY_DIR = os.path.join(BASE, "results", "range_diary")          # one JSON per expiry cycle
DIARY_SENT_PATH = os.path.join(BASE, "results", "range_diary_sent.json")  # Telegram dedupe
LOG_PATH = os.path.join(BASE, "results", "range_trade_log.jsonl")

CONFIG = {
    "lot_size": 65,
    "lots": 1,
    "strike_step": 100,            # leaders and traded strikes: round 100-pt strikes only
    "slippage": 0.5,               # Rs/unit against us on every paper fill
    # research rules (positional_bt "w400_late_t90", adopted 2026-09-29): no stop,
    # target 90%, and enter only on a range at least `min_width` wide
    "sl_frac": None,               # stop at -X of entry premium (booked + open); None = no stop
    "target_frac": 0.9,            # exit at +90% of entry premium ...
    "target_on_net": True,         # ... measured on NET P&L (after charges)
    "exit_time": dtime(15, 0),     # time exit on the day before expiry (or expiry day)
    "leader_min_lead_pct": 0.0,    # optional: new leader must beat the old by this % (0 = user's manual behaviour)
    # Every day at 09:20 the range is computed FRESH from that day's leaders
    # + first-candle highs -- "our range" for the day -- even if the leaders
    # are the same as yesterday's (user, 2026-09-25). Intraday shifts are then
    # measured from it. False = the old behaviour (keep yesterday's range
    # until a leader changes).
    "daily_reset": True,
    # entry only when the range is at least this wide: the Day-1 09:20 range,
    # or (late entry) the first later range recalculation that is -- never on
    # expiry day. None = the old rule (always enter at Day-1 09:20).
    "min_width": 400,
}
FIRST_CANDLE = dtime(9, 15)


def _nearest(x, step):
    return int(round(x / step) * step)


def cycle_dates(expiry_date):
    """(cycle_start, exit_day) for a weekly expiry. Start = first trading day
    after the previous week's expiry (expiry - 7 days); exit day = the
    trading day right before expiry, or expiry day itself if that day is a
    holiday (the user's rule: 'Monday 15:00, but a Monday holiday means we
    go to expiry')."""
    d = expiry_date - timedelta(days=7) + timedelta(days=1)
    while not mc.is_trading_day(d):
        d += timedelta(days=1)
    before = expiry_date - timedelta(days=1)
    exit_day = before if mc.is_trading_day(before) else expiry_date
    return d, exit_day


@dataclass
class Leg:
    side: str      # CE / PE
    strike: int
    entry: float
    entry_time: str


@dataclass
class RangeEngine:
    cycle_start: date
    expiry: date
    cfg: dict = field(default_factory=lambda: dict(CONFIG))

    def __post_init__(self):
        self.exit_day = cycle_dates(self.expiry)[1]
        self.status = "waiting"          # waiting -> open -> closed
        self.legs = {}                   # "CE"/"PE" -> Leg
        self.realized = 0.0
        self.charges = 0.0
        self.premium0 = None             # Rs collected at entry
        self.events = []
        self.range = None                # {"upper","lower","ce_leader","pe_leader","time"}
        self.leaders = None
        self.last_ts = None
        self.exit_reason = None
        self.pnl_path = []               # (ts, total, net)
        self.range_history = []          # every range computed this cycle (leader changes)
        self.first_high_today, self.day = {}, None
        self.diary = []                  # day-start / leader-change / day-end snapshots (Range Diary page)

    # ---- helpers ---------------------------------------------------------
    @property
    def units(self):
        return self.cfg["lot_size"] * self.cfg["lots"]

    @staticmethod
    def _lbl(ts):
        """Events are stamped with the candle's CLOSE time: the 09:15-09:20
        candle's data is only known at 09:20 -- labelling it "09:15" made it
        look like a snapshot taken in the noisy first minutes."""
        return (ts + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")

    def _ev(self, ts, kind, **kw):
        self.events.append({"time": self._lbl(ts), "type": kind, **kw})

    def _sell(self, side, strike, px, ts):
        fill = round(max(0.05, px - self.cfg["slippage"]), 2)
        self.legs[side] = Leg(side, strike, fill, ts.strftime("%Y-%m-%dT%H:%M"))
        self.charges += charges_mod.order_charges("SELL", fill, self.units)
        return fill

    def _buy_back(self, side, px):
        leg = self.legs.pop(side)
        fill = round(px + self.cfg["slippage"], 2)
        pnl = (leg.entry - fill) * self.units
        self.realized += pnl
        self.charges += charges_mod.order_charges("BUY", fill, self.units)
        return leg, fill, pnl

    def _open_pnl(self, price):
        return sum((l.entry - price(l.side, l.strike)) * self.units for l in self.legs.values())

    def _est_exit_charges(self, price):
        return sum(charges_mod.order_charges("BUY", price(l.side, l.strike), self.units) for l in self.legs.values())

    def _close_all(self, ts, price, reason):
        booked = []
        for side in list(self.legs):
            leg, fill, pnl = self._buy_back(side, price(side, self.legs[side].strike))
            booked.append({"side": side, "strike": leg.strike, "entry": leg.entry, "exit": fill, "pnl": round(pnl, 2)})
        self.status, self.exit_reason = "closed", reason
        self._ev(ts, "EXIT", reason=reason, legs=booked, realized=round(self.realized, 2),
                 charges=round(self.charges, 2), net=round(self.realized - self.charges, 2))

    # ---- main replay -------------------------------------------------------
    def run(self, data, until, index=None):
        """data: {("CE"|"PE", strike): {ts(naive IST datetime): candle dict}}.
        Processes every candle with ts <= until, in time order. Idempotent
        only if called on a FRESH engine -- the live runner always rebuilds.
        index: optional {ts: candle} for NIFTY 50 itself (diary spot/OHLC);
        without it the diary uses the put-call-parity forward as spot."""
        self._index = index or {}
        step = self.cfg["strike_step"]
        data = {k: v for k, v in data.items() if k[1] % step == 0}
        all_ts = sorted({ts for d in data.values() for ts in d
                         if self.cycle_start <= ts.date() <= self.expiry and ts <= until})
        last_px = {}
        last_oi, prev_day_oi = {}, {}
        cum, first_high, day = {}, {}, None
        prev_ts, day_changes = None, 0

        def price(side, strike):
            return last_px.get((side, strike))
        self.last_px = last_px

        self._diary_ctx = (last_px, last_oi)
        # Previous trading day's close (prices + OI) before the cycle starts:
        # the Day-1 baseline for the option-data comparison.
        before = sorted({t for d in data.values() for t in d if t.date() < self.cycle_start})
        if before:
            pday = before[-1].date()
            for k, d in data.items():
                rows = [t for t in d if t.date() == pday]
                if rows:
                    c = d[max(rows)]
                    last_px[k] = c["close"]
                    if c.get("oi") is not None:
                        last_oi[k] = c["oi"]
            prev_ts0 = before[-1]
        else:
            prev_ts0 = None
        self._prev_state, self._open_state, self._last_row_state = None, None, None
        for ts in all_ts:
            new_day = ts.date() != day
            if new_day:
                if day is not None and prev_ts is not None:
                    self._diary_add(prev_ts, "DAY_END", cum, prev_day_oi, day_changes=day_changes)
                prev_day_oi = dict(last_oi)
                # the previous day's close, frozen as today's comparison base
                base_ts = prev_ts if prev_ts is not None else prev_ts0
                self._prev_state = self._state(base_ts) if base_ts is not None and last_px else None
                self._open_state = None
                day, cum, first_high, day_changes = ts.date(), {}, {}, 0
            for k, d in data.items():
                c = d.get(ts)
                if c is None:
                    continue
                cum[k] = cum.get(k, 0) + (c["volume"] or 0)
                last_px[k] = c["close"]
                if c.get("oi") is not None:
                    last_oi[k] = c["oi"]
                first_high.setdefault(k, c["high"])  # first candle of the day this strike traded
            self.last_ts = ts
            self.first_high_today, self.day = first_high, day
            leaders = self._leaders(cum)
            if leaders is None:
                continue
            # The range is recomputed only when a leader actually CHANGES
            # (a new day with the same leaders keeps the running range --
            # the user's rule from the 10-Sep walk-through).
            changed = (self.leaders is None or leaders["CE"] != self.leaders["CE"] or leaders["PE"] != self.leaders["PE"]
                       or (new_day and self.cfg.get("daily_reset")))
            old_leaders = dict(self.leaders) if self.leaders else None
            leaders["day"] = day
            self.leaders = leaders
            # The range is tracked all cycle long -- also before entry and
            # after an exit -- so the Option Chain can always show it and
            # the direction it has been shifting.
            prev_leaders = old_leaders if changed else None
            if changed and self._range(leaders, first_high, ts):
                self.range_history.append({k: self.range[k] for k in ("time", "upper", "lower", "ce_leader", "pe_leader")})
            prev_ts = ts
            if new_day or changed:
                if changed and not new_day:
                    day_changes += 1
                self._diary_add(ts, "DAY_START" if new_day else "LEADER_CHANGE", cum, prev_day_oi,
                                prev_leaders=prev_leaders, leaders=leaders, changed=changed)
            if self.status == "closed":
                continue

            # entry: after the cycle's first candle
            if self.status == "waiting":
                mw = self.cfg.get("min_width")
                if mw:
                    rng = self.range
                    late_ok = day < self.expiry and not (day == self.exit_day and ts.time() >= self.cfg["exit_time"])
                    if not late_ok:
                        self.status, self.exit_reason = "closed", f"NO_ENTRY (range never {mw}+ pts)"
                        continue
                    if not (changed and rng) or rng["upper"] - rng["lower"] < mw:
                        continue
                    ok_entry = True
                else:
                    ok_entry = day == self.cycle_start and ts.time() >= FIRST_CANDLE
                if ok_entry:
                    rng = self.range
                    if rng is None:
                        continue
                    ce_k, pe_k = _nearest(rng["upper"], step), _nearest(rng["lower"], step)
                    if price("CE", ce_k) is None or price("PE", pe_k) is None:
                        continue
                    ce_fill = self._sell("CE", ce_k, price("CE", ce_k), ts)
                    pe_fill = self._sell("PE", pe_k, price("PE", pe_k), ts)
                    self.premium0 = round((ce_fill + pe_fill) * self.units, 2)
                    self.status = "open"
                    self._ev(ts, "ENTRY", range=rng, legs=[{"side": "CE", "strike": ce_k, "price": ce_fill},
                                                            {"side": "PE", "strike": pe_k, "price": pe_fill}],
                             premium=self.premium0,
                             stop=round(-self.cfg["sl_frac"] * self.premium0, 2) if self.cfg["sl_frac"] else None,
                             target=round(self.cfg["target_frac"] * self.premium0, 2))
                elif not mw and day > self.cycle_start:
                    self.status = "closed"
                    self.exit_reason = "NO_ENTRY"
                continue

            # open position: risk checks on this candle's close
            if any(price(l.side, l.strike) is None for l in self.legs.values()):
                continue
            total = self.realized + self._open_pnl(price)
            net = total - self.charges - self._est_exit_charges(price)
            self.pnl_path.append((ts.strftime("%Y-%m-%dT%H:%M"), round(total, 2), round(net, 2)))
            if self.cfg["sl_frac"] and total <= -self.cfg["sl_frac"] * self.premium0:
                self._close_all(ts, price, "STOP_LOSS")
                continue
            if (net if self.cfg["target_on_net"] else total) >= self.cfg["target_frac"] * self.premium0:
                self._close_all(ts, price, "TARGET")
                continue
            if day == self.exit_day and ts.time() >= self.cfg["exit_time"]:
                self._close_all(ts, price, "TIME_EXIT")
                continue

            # range recompute + rolls (none on expiry day -- there the user
            # only looks to close: target / stop / time exit above)
            if changed and day != self.expiry:
                rng = self.range
                if rng is None:
                    continue
                want = {"CE": _nearest(rng["upper"], step), "PE": _nearest(rng["lower"], step)}
                rolls = []
                for side in ("CE", "PE"):
                    cur = self.legs[side].strike
                    tgt = want[side]
                    if tgt == cur or price(side, tgt) is None:
                        continue
                    other = "PE" if side == "CE" else "CE"
                    itm = (price(side, cur) or 0) > (price(other, cur) or 0) if price(other, cur) is not None else False
                    toward = tgt < cur if side == "CE" else tgt > cur
                    if toward or itm:
                        leg, bfill, pnl = self._buy_back(side, price(side, cur))
                        sfill = self._sell(side, tgt, price(side, tgt), ts)
                        rolls.append({"side": side, "from": cur, "buy_back": bfill, "booked": round(pnl, 2),
                                      "to": tgt, "sell": sfill, "why": "toward range" if toward else "leg in the money"})
                self._ev(ts, "RANGE", range=rng, wanted=want, rolls=rolls,
                         legs={s: {"strike": l.strike, "entry": l.entry} for s, l in self.legs.items()},
                         total=round(total, 2))
        if self.last_ts is not None and self.last_ts.time() >= dtime(15, 35):
            self._diary_add(self.last_ts, "DAY_END", cum, prev_day_oi, day_changes=day_changes)
        return self

    # ---- Range Diary ---------------------------------------------------------
    NEAR_WINDOW = 500  # "near the money" = strikes within +/-500 pts of ATM

    def _spot_fwd(self, ts):
        """(spot, forward, T) at ts: NIFTY's own candle (or its latest one
        earlier that day), the near-ATM put-call-parity forward, and years
        to expiry."""
        last_px, _ = self._diary_ctx
        strikes = sorted({k[1] for k in last_px})
        idx = self._index.get(ts)
        if idx is None and self._index:
            prior = [t for t in self._index if t.date() == ts.date() and t <= ts]
            idx = self._index[max(prior)] if prior else None
        quotes = [(k, last_px.get(("CE", k)), last_px.get(("PE", k))) for k in strikes]
        quotes = [q for q in quotes if q[1] and q[2]]
        fwd = greeks_mod.synthetic_forward(quotes)
        spot = idx["close"] if idx else fwd
        if spot and quotes:
            fwd = greeks_mod.synthetic_forward(sorted(quotes, key=lambda q: abs(q[0] - spot))[:6]) or fwd
        exp_dt = datetime.combine(self.expiry, dtime(15, 40))
        T = max((exp_dt - ts).total_seconds(), 60) / (365 * 86400)
        return spot, fwd, T, bool(idx)

    def _state(self, ts):
        """Frozen copy of the chain at ts -- prices, OI, spot, ATM IV -- used
        as a comparison base (previous day's close / today's 09:20 / the
        previous event)."""
        last_px, last_oi = self._diary_ctx
        spot, fwd, T, real = self._spot_fwd(ts)
        st = {"ts": ts, "px": dict(last_px), "oi": dict(last_oi), "spot": spot, "fwd": fwd, "T": T, "real_spot": real}
        strikes = sorted({k[1] for k in last_px})
        st["atm"] = min(strikes, key=lambda k: abs(k - spot)) if spot and strikes else None
        st["iv"] = None
        if st["atm"] and fwd and T * 365 * 24 >= 1:
            c, p = last_px.get(("CE", st["atm"])), last_px.get(("PE", st["atm"]))
            ivs = [v for v in (greeks_mod.implied_vol_fwd(c, fwd, st["atm"], T, greeks_mod.RISK_FREE_RATE, True) if c else None,
                               greeks_mod.implied_vol_fwd(p, fwd, st["atm"], T, greeks_mod.RISK_FREE_RATE, False) if p else None) if v]
            st["iv"] = round(sum(ivs) / len(ivs) * 100, 2) if ivs else None
        ce_tot = sum(v for k, v in last_oi.items() if k[0] == "CE")
        pe_tot = sum(v for k, v in last_oi.items() if k[0] == "PE")
        st["pcr"] = round(pe_tot / ce_tot, 3) if ce_tot else None
        import oi_walls
        ce_oi = {k[1]: v for k, v in last_oi.items() if k[0] == "CE"}
        pe_oi = {k[1]: v for k, v in last_oi.items() if k[0] == "PE"}
        straddle = None
        if st["atm"]:
            c, p = last_px.get(("CE", st["atm"])), last_px.get(("PE", st["atm"]))
            straddle = (c + p) if c and p else None
        w = oi_walls.walls(ce_oi, pe_oi, spot, fwd=fwd, T=T, iv_pct=st["iv"], straddle=straddle)
        st["resistance"], st["support"], st["walls"] = w["resistance"], w["support"], w
        return st

    @staticmethod
    def _wall_txt(st, side):
        top = ((st.get("walls") or {}).get(side + "_detail")) or []
        if not top:
            return ""
        t = top[0]
        return f" · {t['oi'] / 1e5:,.0f} L · {t['distance']} pts away"

    @staticmethod
    def _lakh(v):
        return f"{v / 1e5:+,.1f} L"

    def _oi_view(self, base, now, label):
        """Institutional-style read of the option chain between two moments:
        price, who is writing near the money, PCR, where the put (support) and
        call (resistance) walls moved, ATM call-vs-put premium strength, and
        IV with price. Each factor: Before / Now / Change / Reading and a
        +1 / 0 / -1 signal -- shown as a table on the Range Diary."""
        if not base or not now or not base.get("spot") or not now.get("spot"):
            return None
        f = []
        L = lambda v: f"{v / 1e5:,.0f} L"

        def add(name, score, before, now_v, change, reading):
            f.append({"name": name, "score": score, "before": before, "now": now_v, "change": change,
                      "reading": reading, "detail": f"{before} → {now_v} ({change}) — {reading}"})

        d = now["spot"] - base["spot"]
        pct = d / base["spot"] * 100
        add("NIFTY", 1 if pct >= 0.2 else -1 if pct <= -0.2 else 0, f"{base['spot']:,.1f}", f"{now['spot']:,.1f}",
            f"{d:+.0f} pts ({pct:+.2f}%)", "up" if d > 0 else "down" if d < 0 else "flat")
        a = base["atm"]
        win = sorted(k for k in {k[1] for k in now["oi"]} if a is not None and abs(k - a) <= self.NEAR_WINDOW)
        ce0 = sum(base["oi"].get(("CE", k), now["oi"].get(("CE", k), 0)) for k in win)
        pe0 = sum(base["oi"].get(("PE", k), now["oi"].get(("PE", k), 0)) for k in win)
        ce1 = sum(now["oi"].get(("CE", k), 0) for k in win)
        pe1 = sum(now["oi"].get(("PE", k), 0) for k in win)
        dce, dpe = ce1 - ce0, pe1 - pe0
        if dpe > 0 and dce > 0:
            sc = 1 if dpe > 1.25 * dce else -1 if dce > 1.25 * dpe else 0
            why = "put writing stronger → support building" if sc > 0 else "call writing stronger → resistance building" if sc < 0 else "both sides writing equally"
        elif dpe >= 0 >= dce and (dpe or dce):
            sc, why = 1, "put writing + call unwinding"
        elif dce >= 0 >= dpe and (dpe or dce):
            sc, why = -1, "call writing + put unwinding"
        elif dpe < 0 and dce < 0:
            sc = 1 if -dce > 1.25 * -dpe else -1 if -dpe > 1.25 * -dce else 0
            why = "call unwinding stronger" if sc > 0 else "put unwinding stronger" if sc < 0 else "both sides unwinding"
        else:
            sc, why = 0, "no change"
        rng = f"{win[0]}–{win[-1]}" if win else "near ATM"
        add(f"OI writing ({rng})", sc, f"PE {L(pe0)} · CE {L(ce0)}", f"PE {L(pe1)} · CE {L(ce1)}",
            f"PE {self._lakh(dpe)} · CE {self._lakh(dce)}", why)
        if base.get("pcr") and now.get("pcr"):
            dp = now["pcr"] - base["pcr"]
            add("PCR", 1 if dp >= 0.05 else -1 if dp <= -0.05 else 0, f"{base['pcr']:.2f}", f"{now['pcr']:.2f}", f"{dp:+.2f}",
                "puts gaining vs calls" if dp >= 0.05 else "calls gaining vs puts" if dp <= -0.05 else "little change")
        if base.get("support") and now.get("support"):
            ds = now["support"] - base["support"]
            add("Support (put OI wall)", 1 if ds > 0 else -1 if ds < 0 else 0, f"{base['support']}",
                f"{now['support']}" + self._wall_txt(now, "support"), f"{ds:+.0f}",
                "support moved up" if ds > 0 else "support moved down" if ds < 0 else "unchanged")
        if base.get("resistance") and now.get("resistance"):
            dr = now["resistance"] - base["resistance"]
            add("Resistance (call OI wall)", 1 if dr > 0 else -1 if dr < 0 else 0, f"{base['resistance']}",
                f"{now['resistance']}" + self._wall_txt(now, "resistance"), f"{dr:+.0f}",
                "resistance moved up" if dr > 0 else "resistance moved down" if dr < 0 else "unchanged")
        if a is not None:
            c0, c1 = base["px"].get(("CE", a)), now["px"].get(("CE", a))
            p0, p1 = base["px"].get(("PE", a)), now["px"].get(("PE", a))
            if c0 and c1 and p0 and p1:
                cp, pp = (c1 / c0 - 1) * 100, (p1 / p0 - 1) * 100
                diff = cp - pp
                add(f"ATM {a} CE vs PE", 1 if diff >= 10 else -1 if diff <= -10 else 0, f"CE {c0:.1f} · PE {p0:.1f}",
                    f"CE {c1:.1f} · PE {p1:.1f}", f"CE {cp:+.0f}% · PE {pp:+.0f}%",
                    "calls stronger" if diff >= 10 else "puts stronger" if diff <= -10 else "balanced")
        if base.get("iv") is not None and now.get("iv") is not None:
            di = now["iv"] - base["iv"]
            sc = -1 if d < 0 and di >= 0.3 else 1 if d > 0 and di <= -0.3 else 0
            add("ATM IV with price", sc, f"{base['iv']:.2f}%", f"{now['iv']:.2f}%", f"{di:+.2f}",
                "fear on the fall" if sc < 0 else "calm rally" if sc > 0 else "no signal")
        score = sum(x["score"] for x in f)
        verdict = ("STRONG BULLISH" if score >= 4 else "BULLISH" if score >= 2 else "MILD BULLISH" if score == 1 else
                   "STRONG BEARISH" if score <= -4 else "BEARISH" if score <= -2 else "MILD BEARISH" if score == -1 else "NEUTRAL")
        return {"vs": label, "since": self._lbl(base["ts"]) if base.get("ts") else None,
                "score": score, "verdict": verdict, "factors": f}

    def _confirm(self, shift, base, now):
        """Does price + option data agree with the direction the range just
        shifted? Compared with the previous event."""
        if not shift or shift["direction"] == "FLAT" or not base or not now or not base.get("spot") or not now.get("spot"):
            return None
        up = shift["direction"] == "UP"
        items = []
        d = now["spot"] - base["spot"]
        items.append({"name": "NIFTY", "value": f"{d:+.0f} pts", "ok": (d > 0) == up and abs(d) >= 5})
        a = base["atm"]
        win = [k for k in {k[1] for k in now["oi"]} if a is not None and abs(k - a) <= self.NEAR_WINDOW]
        dce = sum(now["oi"].get(("CE", k), 0) - base["oi"].get(("CE", k), now["oi"].get(("CE", k), 0)) for k in win)
        dpe = sum(now["oi"].get(("PE", k), 0) - base["oi"].get(("PE", k), now["oi"].get(("PE", k), 0)) for k in win)
        net = dpe - dce
        items.append({"name": "Net writing (put − call OI)", "value": self._lakh(net), "ok": (net > 0) == up and net != 0})
        if base.get("pcr") and now.get("pcr"):
            dp = now["pcr"] - base["pcr"]
            items.append({"name": "PCR", "value": f"{dp:+.3f}", "ok": (dp > 0) == up and dp != 0})
        return {"direction": shift["direction"], "aligned": sum(1 for x in items if x["ok"]), "of": len(items), "items": items}

    def _diary_add(self, ts, kind, cum, prev_day_oi, prev_leaders=None, leaders=None, changed=False, day_changes=None):
        """One diary row: who leads (and by how much), the range and its
        shift (intraday view), the best strikes to sell by the user's rule
        plus a safer one, the market at that moment, the option-data view vs
        the previous day's close and vs today's 09:20, and whether price +
        option data confirm the range's shift. Built only from candles, so a
        restart or a past week produces exactly the same rows."""
        last_px, last_oi = self._diary_ctx
        step = self.cfg["strike_step"]
        leaders = leaders or self.leaders or {}
        rng = self.range
        now = self._state(ts)
        spot, fwd, T = now["spot"], now["fwd"], now["T"]
        r = greeks_mod.RISK_FREE_RATE

        def strike_info(side, k):
            px = last_px.get((side, k))
            if px is None:
                return None
            info = {"strike": k, "premium": round(px, 2),
                    "distance": round((k - spot) if side == "CE" else (spot - k), 1) if spot else None}
            if fwd and T * 365 * 24 >= 1:  # IV/probability are meaningless in expiry's last hour
                g = greeks_mod.option_greeks_fwd(px, fwd, k, T, r, side == "CE")
                if g:
                    info.update(iv=g["iv"], delta=g["delta"], pop=round((1 - abs(g["delta"])) * 100, 1))
            return info

        row = {"time": self._lbl(ts), "day": ts.date().isoformat(), "kind": kind}
        if kind != "DAY_END" or leaders:
            top = {}
            for side in ("CE", "PE"):
                vols = sorted(((v, k[1]) for k, v in cum.items() if k[0] == side), reverse=True)[:3]
                top[side] = [{"strike": k, "volume": int(v)} for v, k in vols]
            lead_pct = {side: (round((top[side][0]["volume"] / top[side][1]["volume"] - 1) * 100, 1)
                               if len(top[side]) > 1 and top[side][1]["volume"] else None) for side in top}
            row.update(leaders={"CE": leaders.get("CE"), "PE": leaders.get("PE")},
                       prev_leaders=({"CE": prev_leaders.get("CE"), "PE": prev_leaders.get("PE")} if prev_leaders else None),
                       leader_changed=bool(changed and prev_leaders), top=top, lead_pct=lead_pct)
        if rng:
            row["range"] = {"lower": rng["lower"], "upper": rng["upper"], "width": round(rng["upper"] - rng["lower"], 1),
                            "ce_first_high": rng["ce_first_high"], "pe_first_high": rng["pe_first_high"], "since": rng["time"]}
            if len(self.range_history) >= 2 and changed:
                a, b = self.range_history[-2], self.range_history[-1]
                d = round((b["upper"] + b["lower"]) / 2 - (a["upper"] + a["lower"]) / 2, 1)
                row["shift"] = {"direction": "UP" if d > 0 else "DOWN" if d < 0 else "FLAT", "points": d,
                                "upper_move": round(b["upper"] - a["upper"], 1), "lower_move": round(b["lower"] - a["lower"], 1),
                                "from": [round(a["lower"]), round(a["upper"])], "to": [round(b["lower"]), round(b["upper"])]}
                if kind == "LEADER_CHANGE" and row["shift"]["direction"] != "FLAT":
                    row["intraday_view"] = "BULLISH" if row["shift"]["direction"] == "UP" else "BEARISH"
            ce_rule, pe_rule = _nearest(rng["upper"], step), _nearest(rng["lower"], step)
            ce_safe = -(-rng["upper"] // step) * step            # first strike at/above the upper
            pe_safe = (rng["lower"] // step) * step              # first strike at/below the lower
            ce_safe = int(ce_safe if ce_safe != ce_rule else ce_rule + step)
            pe_safe = int(pe_safe if pe_safe != pe_rule else pe_rule - step)
            row["best"] = {"CE": {"rule": strike_info("CE", ce_rule), "safer": strike_info("CE", ce_safe)},
                           "PE": {"rule": strike_info("PE", pe_rule), "safer": strike_info("PE", pe_safe)}}
        # market snapshot
        ce_chg = sum(v - prev_day_oi.get(k, v) for k, v in last_oi.items() if k[0] == "CE")
        pe_chg = sum(v - prev_day_oi.get(k, v) for k, v in last_oi.items() if k[0] == "PE")
        mkt = {"spot": round(spot, 2) if spot else None, "spot_is_forward": not now["real_spot"],
               "forward": round(fwd, 2) if fwd else None,
               "pcr_oi": round(now["pcr"], 2) if now["pcr"] else None,
               "pcr_chg_oi": round(pe_chg / ce_chg, 2) if prev_day_oi and ce_chg > 0 and pe_chg > 0 else None,
               "support": now["support"], "resistance": now["resistance"], "atm_iv": now["iv"],
               "walls": now.get("walls")}
        if now["atm"]:
            atm = now["atm"]
            c, p = last_px.get(("CE", atm)), last_px.get(("PE", atm))
            mkt["atm_strike"] = atm
            if c and p:
                mkt["atm_straddle"] = round(c + p, 2)
            oi_strikes = sorted({k[1] for k in last_oi})
            if oi_strikes:
                pay = lambda S: sum(max(0, S - k) * last_oi.get(("CE", k), 0) + max(0, k - S) * last_oi.get(("PE", k), 0)
                                    for k in oi_strikes)
                mkt["max_pain"] = min(oi_strikes, key=pay)
        row["market"] = mkt
        # option-data views: vs previous day's close, and vs today's 09:20
        if kind == "DAY_START":
            self._open_state = now
            # TODAY's range, freshly from today's leaders + today's first-candle
            # highs -- for the daily view (vs the previous day's final range),
            # even when the leaders didn't change and the running range stays.
            fh = self.first_high_today
            ce_hi, pe_hi = fh.get(("CE", leaders.get("CE"))), fh.get(("PE", leaders.get("PE")))
            if ce_hi is not None and pe_hi is not None:
                up, lo = round(leaders["CE"] + pe_hi, 2), round(leaders["PE"] - ce_hi, 2)
                row["day_range"] = {"lower": lo, "upper": up, "width": round(up - lo, 1),
                                    "ce_leader": leaders["CE"], "pe_leader": leaders["PE"],
                                    "ce_first_high": ce_hi, "pe_first_high": pe_hi}
        # ONE comparison per event: the day's first snapshot (09:20) vs the
        # previous day's close; every later snapshot vs the snapshot just
        # before it today (the first leader change vs 09:20, the next vs that).
        if kind == "DAY_START":
            row["oi_view"] = self._oi_view(self._prev_state, now, "previous day close")
        else:
            prev = self._last_row_state
            row["oi_view"] = self._oi_view(prev, now, "09:20 snapshot" if prev is self._open_state else "previous snapshot")
        if kind == "DAY_END":
            # the day as a whole, for the day / expiry summary cards only
            row["oi_view_day"] = self._oi_view(self._prev_state, now, "previous day close")
        if row.get("shift"):
            row["confirm"] = self._confirm(row["shift"], self._last_row_state, now)
        if kind == "DAY_END":
            day_idx = [c for t, c in sorted(self._index.items()) if t.date() == ts.date()]
            if day_idx:
                row["nifty_ohlc"] = [day_idx[0]["open"], max(c["high"] for c in day_idx),
                                     min(c["low"] for c in day_idx), day_idx[-1]["close"]]
            row["leader_changes"] = day_changes
            starts = [x for x in self.diary if x["day"] == row["day"] and x.get("range")]
            if starts and rng:
                a = starts[0]["range"]
                d = round((rng["upper"] + rng["lower"]) / 2 - (a["upper"] + a["lower"]) / 2, 1)
                row["day_shift"] = {"direction": "UP" if d > 0 else "DOWN" if d < 0 else "FLAT", "points": d}
        if kind != "DAY_END":
            self._last_row_state = now
        self.diary.append(row)

    def _leaders(self, cum):
        out = {}
        for side in ("CE", "PE"):
            cands = [(v, k[1]) for k, v in cum.items() if k[0] == side]
            if not cands:
                return None
            best_v, best_k = max(cands)
            prev = self.leaders.get(side) if self.leaders else None
            lead_pct = self.cfg["leader_min_lead_pct"]
            if prev is not None and best_k != prev and lead_pct and self.leaders.get("day") is not None:
                prev_v = cum.get((side, prev), 0)
                if best_v < prev_v * (1 + lead_pct / 100):
                    best_k = prev  # not a decisive enough lead -- keep the old leader
            out[side] = best_k
        return out

    def _range(self, leaders, first_high, ts):
        ce_hi = first_high.get(("CE", leaders["CE"]))
        pe_hi = first_high.get(("PE", leaders["PE"]))
        if ce_hi is None or pe_hi is None:
            return None
        self.range = {"upper": round(leaders["CE"] + pe_hi, 2), "lower": round(leaders["PE"] - ce_hi, 2),
                      "ce_leader": leaders["CE"], "pe_leader": leaders["PE"], "ce_first_high": ce_hi,
                      "pe_first_high": pe_hi, "time": self._lbl(ts)}
        return self.range

    def snapshot(self, price=None):
        return {
            "cycle_start": self.cycle_start.isoformat(), "expiry": self.expiry.isoformat(),
            "exit_day": self.exit_day.isoformat(), "status": self.status, "exit_reason": self.exit_reason,
            "range": self.range, "leaders": {k: v for k, v in (self.leaders or {}).items() if k != "day"},
            "legs": {s: {**vars(l), "ltp": getattr(self, "last_px", {}).get((s, l.strike)),
                         "pnl": round((l.entry - getattr(self, "last_px", {}).get((s, l.strike), l.entry)) * self.units, 2)}
                     for s, l in self.legs.items()},
            "pnl_path": self.pnl_path[-300:],
            "range_history": self.range_history,
            "diary": self.diary,
            # today's first-candle high of every 100-pt strike: lets the
            # Option Chain recompute the range LIVE (every 3s) the moment the
            # volume leader changes, before the 5-min candle closes
            "today": self.day.isoformat() if self.day else None,
            "today_first_high": {side: {str(k[1]): v for k, v in self.first_high_today.items() if k[0] == side}
                                 for side in ("CE", "PE")},
            "premium0": self.premium0,
            "stop": round(-self.cfg["sl_frac"] * self.premium0, 2) if self.premium0 and self.cfg["sl_frac"] else None,
            "target": round(self.cfg["target_frac"] * self.premium0, 2) if self.premium0 else None,
            "realized": round(self.realized, 2), "charges": round(self.charges, 2),
            "last_pnl": self.pnl_path[-1] if self.pnl_path else None,
            "last_candle": self.last_ts.strftime("%Y-%m-%dT%H:%M") if self.last_ts else None,
            "events": self.events, "config": {k: (v.strftime("%H:%M") if isinstance(v, dtime) else v)
                                              for k, v in self.cfg.items()},
        }


# ---- data providers ---------------------------------------------------------

def load_local_cycle(expiry_date):
    """Candles for one expiry from data/*.csv (backtest)."""
    import pandas as pd
    exps = {e["expiry"]: e for e in json.load(open(os.path.join(BASE, "expiries.json")))}
    e = exps.get(expiry_date.isoformat())
    if e is None:
        return {}
    data = {}
    for inst in json.load(open(os.path.join(BASE, e["instruments_file"]))):
        if int(inst["strike"]) % CONFIG["strike_step"]:
            continue
        path = os.path.join(BASE, "data", f"{inst['symbol']}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).drop_duplicates("date", keep="last")
        d = {}
        for r in df.itertuples(index=False):
            try:
                ts = datetime.fromisoformat(str(r.date)).replace(tzinfo=None)
            except ValueError:
                continue  # a malformed row (e.g. half-written line) -- skip it
            d[ts] = {"open": r.open, "high": r.high, "low": r.low, "close": r.close,
                     "volume": r.volume, "oi": r.oi}
        data[(inst["type"], int(inst["strike"]))] = d
    return data


def load_hist_cycle(expiry_date):
    """Candles for one expiry from the 1-min history in data/hist1m (kept
    current each evening by daily_history.py), as 5-min candles labelled by
    their start time -- the same shape load_local_cycle() returns."""
    import pandas as pd
    import simulator
    try:
        df = simulator._load_options(expiry_date.isoformat())
    except (FileNotFoundError, OSError):
        return {}
    df = df[df["strike"] % CONFIG["strike_step"] == 0]
    mins = df["date"].dt.hour * 60 + df["date"].dt.minute
    df = df[mins <= 15 * 60 + 40].copy()
    df["bar"] = simulator._bucket_labels(pd.DatetimeIndex(df["date"]), 5).tz_localize(None)
    g = (df.sort_values("ts").groupby(["type", "strike", "bar"])
           .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"), oi=("oi", "last")).reset_index())
    data = {}
    for (typ, k), x in g.groupby(["type", "strike"]):
        data[(typ, int(k))] = {ts.to_pydatetime(): {"open": float(o), "high": float(h), "low": float(l), "close": float(c),
                                                     "volume": float(v), "oi": float(oi)}
                               for ts, o, h, l, c, v, oi in zip(x.bar, x.open, x.high, x.low, x.close, x.volume, x.oi)}
    return data


def load_live_cycle(expiry_info, start, today, strikecount=30):
    """Candles for the cycle from Fyers 5-min history, for every 100-pt
    strike within +/-strikecount x50 pts of spot on this expiry."""
    import fyers_option_chain as chain_mod
    import historical_recorder
    chain = chain_mod.get_chain(strikecount=strikecount, expiry_timestamp=expiry_info["expiry"])
    data = {}
    for row in chain["strikes"]:
        if int(row["strike"]) % CONFIG["strike_step"]:
            continue
        for key, side in (("ce", "CE"), ("pe", "PE")):
            leg = row[key]
            if not leg or not leg.get("symbol"):
                continue
            rows = historical_recorder._fetch_history_candles(leg["symbol"], mc.previous_trading_day(start).isoformat(),
                                                              today.isoformat())
            d = {}
            for r in rows:
                ts = datetime.fromtimestamp(r[0], mc.IST).replace(tzinfo=None)
                d[ts] = {"open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5],
                         "oi": r[6] if len(r) > 6 else 0}
            data[(side, int(row["strike"]))] = d
            time.sleep(0.13)  # stay under Fyers' 10 req/s
    return data


def load_index(start, end):
    """NIFTY 50 5-min candles {ts: candle} from Fyers (the index never
    expires, so past weeks work too). {} on any failure -- the diary then
    falls back to the put-call-parity forward as its spot."""
    try:
        import fyers_option_chain as chain_mod
        import historical_recorder
        rows = historical_recorder._fetch_history_candles(chain_mod.INDEX_SYMBOL, start.isoformat(), end.isoformat())
    except Exception:
        return {}
    return {datetime.fromtimestamp(r[0], mc.IST).replace(tzinfo=None):
            {"open": r[1], "high": r[2], "low": r[3], "close": r[4]} for r in rows}


def save_diary(eng, updated_at=None):
    os.makedirs(DIARY_DIR, exist_ok=True)
    paths.atomic_write_json(os.path.join(DIARY_DIR, f"{eng.expiry.isoformat()}.json"), {
        "cycle_start": eng.cycle_start.isoformat(), "expiry": eng.expiry.isoformat(),
        "updated_at": updated_at or mc.now_ist().isoformat(timespec="seconds"),
        "status": eng.status, "exit_reason": eng.exit_reason, "diary": eng.diary,
        # strategy events (entry / rolls / exit) and NIFTY's own 5-min closes
        # per day, for the day view's timeline and its range-vs-NIFTY chart
        "events": eng.events,
        "nifty": {d: [[t.strftime("%H:%M"), c["close"]] for t, c in sorted(eng._index.items()) if t.date().isoformat() == d]
                  for d in sorted({t.date().isoformat() for t in eng._index})
                  if eng.cycle_start.isoformat() <= d <= eng.expiry.isoformat()},
        "premium0": eng.premium0, "realized": round(eng.realized, 2), "charges": round(eng.charges, 2)})


def build_diary(expiry_date):
    """Builds (and saves) the Range Diary for a completed week from the
    recorded candles + NIFTY's own 5-min history."""
    start, _ = cycle_dates(expiry_date)
    data = load_local_cycle(expiry_date)
    if not data:
        return None
    eng = RangeEngine(start, expiry_date).run(data, datetime.combine(expiry_date, dtime(15, 40)),
                                              index=load_index(mc.previous_trading_day(start), expiry_date))
    save_diary(eng)
    return eng


def list_diaries():
    """Every week a diary exists for, or can be built from recorded data."""
    have = set()
    if os.path.isdir(DIARY_DIR):
        have = {f[:-5] for f in os.listdir(DIARY_DIR) if f.endswith(".json")}
    try:
        recorded = {e["expiry"] for e in json.load(open(os.path.join(BASE, "expiries.json")))}
    except Exception:
        recorded = set()
    today = mc.now_ist().date().isoformat()
    out = []
    for exp in sorted(have | {e for e in recorded if e < today}, reverse=True):
        out.append({"expiry": exp, "cycle_start": cycle_dates(date.fromisoformat(exp))[0].isoformat(), "saved": exp in have})
    return out


def get_diary(expiry):
    path = os.path.join(DIARY_DIR, f"{expiry}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    eng = build_diary(date.fromisoformat(expiry))
    if eng is None:
        return None
    with open(path) as f:
        return json.load(f)


def all_days():
    """Every trading day that has a diary, newest first, with its week."""
    out = []
    for w in list_diaries():
        d = get_diary(w["expiry"])
        if not d:
            continue
        for day in sorted({r["day"] for r in d.get("diary", [])}, reverse=True):
            out.append({"date": day, "expiry": w["expiry"], "cycle_start": d["cycle_start"]})
    return sorted(out, key=lambda x: x["date"], reverse=True)


def _daily_view(today_rows, prev_rows):
    """Your method's DAILY view: today's 09:20 range (fresh, from today's
    leaders + first-candle highs) vs the previous trading day's final range.
    For Day 1 of an expiry the previous day is the previous expiry's last day."""
    start = next((r for r in today_rows if r["kind"] == "DAY_START"), None)
    if not start or not start.get("day_range") or not prev_rows:
        return None
    prev_end = next((r for r in reversed(prev_rows) if r.get("range")), None)
    if not prev_end:
        return None
    a, b = prev_end["range"], start["day_range"]
    d = round((b["upper"] + b["lower"]) / 2 - (a["upper"] + a["lower"]) / 2, 1)
    view = "BULLISH" if d >= 10 else "BEARISH" if d <= -10 else "NEUTRAL"
    return {"view": view, "points": d, "upper_move": round(b["upper"] - a["upper"], 1),
            "lower_move": round(b["lower"] - a["lower"], 1),
            "prev": {"day": prev_end["day"], "lower": a["lower"], "upper": a["upper"]},
            "today": {"lower": b["lower"], "upper": b["upper"], "ce_leader": b["ce_leader"], "pe_leader": b["pe_leader"]}}


def _rows_by_day():
    out = {}
    for w in list_diaries():
        d = get_diary(w["expiry"])
        if not d:
            continue
        for r in d.get("diary", []):
            out.setdefault(r["day"], {"expiry": w["expiry"], "cycle_start": d["cycle_start"], "rows": []})["rows"].append(r)
    return out


def expiry_view(expiry):
    """Day 1 (cycle start, normally Wednesday) ... last day, one summary each."""
    d = get_diary(expiry)
    if not d:
        return None
    by_day = _rows_by_day()
    all_dates = sorted(by_day)
    days = sorted({r["day"] for r in d["diary"]})
    out = []
    for i, day in enumerate(days, 1):
        rows = [r for r in d["diary"] if r["day"] == day]
        k = all_dates.index(day)
        prev_rows = by_day[all_dates[k - 1]]["rows"] if k > 0 else None
        end = next((r for r in rows if r["kind"] == "DAY_END"), None)
        last = end or rows[-1]
        shifts = [r.get("intraday_view") for r in rows if r.get("intraday_view")]
        confirms = [r["confirm"] for r in rows if r.get("confirm")]
        out.append({
            "day_no": i, "date": day, "daily": _daily_view(rows, prev_rows),
            "open_range": next((r.get("day_range") for r in rows if r["kind"] == "DAY_START"), None),
            "close_range": last.get("range"), "leaders_close": last.get("leaders"),
            "nifty_ohlc": (end or {}).get("nifty_ohlc"),
            "leader_changes": sum(1 for r in rows if r["kind"] == "LEADER_CHANGE"),
            "intraday": {"bullish": shifts.count("BULLISH"), "bearish": shifts.count("BEARISH"), "sequence": shifts,
                         "last": shifts[-1] if shifts else None},
            "oi_view_day": (end or {}).get("oi_view_day") or next((r.get("oi_view") for r in rows if r["kind"] == "DAY_START"), None),
            "confirm": {"aligned": sum(c["aligned"] for c in confirms), "of": sum(c["of"] for c in confirms)} if confirms else None,
            "market": last.get("market"),
            "events": [e for e in d.get("events", []) if e["time"][:10] == day],
            "in_progress": end is None,
        })
    return {"expiry": d["expiry"], "cycle_start": d["cycle_start"], "status": d.get("status"),
            "exit_reason": d.get("exit_reason"), "days": out, "updated_at": d.get("updated_at"),
            "cycles": list_diaries()}


def day_detail(day):
    """Everything the Range Diary shows for ONE date."""
    days = all_days()
    match = next((x for x in days if x["date"] == day), None)
    dates = [x["date"] for x in days]
    if match is None:
        earlier = [d for d in dates if d < day]
        later = [d for d in dates if d > day]
        return {"date": day, "found": False, "nearest_before": earlier[0] if earlier else None,
                "nearest_after": later[-1] if later else None, "available": dates}
    d = get_diary(match["expiry"])
    rows = [r for r in d["diary"] if r["day"] == day]
    # strategy ideas saved at each live snapshot (snapshot_ideas.py)
    try:
        import snapshot_ideas
        store = snapshot_ideas.load(d["expiry"])
        rows = [{**r, "ideas_block": store["rows"].get(snapshot_ideas.row_key(r))} for r in rows]
    except Exception:
        pass
    by_day = _rows_by_day()
    all_dates = sorted(by_day)
    k = all_dates.index(day)
    prev_rows = by_day[all_dates[k - 1]]["rows"] if k > 0 else None
    day_no = sorted({r["day"] for r in d["diary"]}).index(day) + 1
    events = [e for e in d.get("events", []) if e["time"][:10] == day]
    # strategy state at the end of that day
    upto = [e for e in d.get("events", []) if e["time"][:10] <= day]
    exited = next((e for e in upto if e["type"] == "EXIT"), None)
    status = "closed" if exited else ("open" if any(e["type"] == "ENTRY" for e in upto) else "waiting")
    i = dates.index(day)
    return {"date": day, "found": True, "expiry": d["expiry"], "cycle_start": d["cycle_start"],
            "rows": rows, "events": events, "nifty": (d.get("nifty") or {}).get(day, []),
            "day_no": day_no, "daily": _daily_view(rows, prev_rows),
            "strategy": {"status": status, "exit": exited, "premium0": d.get("premium0")},
            "prev_day": dates[i + 1] if i + 1 < len(dates) else None,
            "next_day": dates[i - 1] if i > 0 else None,
            "available": dates, "updated_at": d.get("updated_at")}


def fmt_diary(row, daily=None):
    """Telegram text for a leader change / day start."""
    t = row["time"][11:16]
    lead = row.get("leaders") or {}
    prev = row.get("prev_leaders") or {}
    lp = row.get("lead_pct") or {}

    def side(s):
        txt = f"{prev[s]}→{lead[s]}" if prev.get(s) not in (None, lead.get(s)) else f"{lead.get(s)}"
        return f"{s} {txt}" + (f" (lead {lp[s]:.0f}%)" if lp.get(s) is not None else "")
    head = ("🌅 Day start" if row["kind"] == "DAY_START" else "🔔 Leader change") + f" [{row['day']} {t}]"
    lines = [head]
    if daily:
        lines.append(f"DAILY VIEW: {daily['view']} ({daily['points']:+.0f} pts vs {daily['prev']['day']} final range "
                     f"{daily['prev']['lower']:.0f}-{daily['prev']['upper']:.0f} -> {daily['today']['lower']:.0f}-{daily['today']['upper']:.0f})")
    if row.get("intraday_view"):
        lines.append(f"INTRADAY VIEW: {row['intraday_view']}")
    lines.append(f"{side('CE')} · {side('PE')}")
    rng, sh = row.get("range"), row.get("shift")
    if rng:
        lines.append(f"Range {rng['lower']:.0f} – {rng['upper']:.0f} ({rng['width']:.0f} pts)"
                     + (f" {'▲ UP' if sh['direction'] == 'UP' else '▼ DOWN' if sh['direction'] == 'DOWN' else ''} {abs(sh['points']):.0f} pts"
                        + (" (bullish shift)" if sh["direction"] == "UP" else " (bearish shift)" if sh["direction"] == "DOWN" else "")
                        if sh else ""))
    b = row.get("best") or {}

    def bs(s):
        r, sf = (b.get(s) or {}).get("rule"), (b.get(s) or {}).get("safer")
        if not r:
            return f"{s} —"
        txt = f"{r['strike']} {s} @{r['premium']}" + (f" (POP {r['pop']:.0f}%)" if r.get("pop") is not None else "")
        if sf:
            txt += f" · safer {sf['strike']} @{sf['premium']}" + (f" ({sf['pop']:.0f}%)" if sf.get("pop") is not None else "")
        return txt
    if b:
        lines.append(f"Best sell: {bs('CE')} | {bs('PE')}")
    m = row.get("market") or {}
    lines.append(" · ".join(x for x in [
        f"NIFTY {m['spot']:.0f}" if m.get("spot") else None,
        f"PCR {m['pcr_oi']}" if m.get("pcr_oi") is not None else None,
        f"ATM IV {m['atm_iv']}%" if m.get("atm_iv") is not None else None,
        f"Straddle {m['atm_straddle']}" if m.get("atm_straddle") is not None else None,
        f"Max pain {m['max_pain']}" if m.get("max_pain") is not None else None] if x))
    v = row.get("oi_view")
    if v:
        lines.append(f"Option data vs {v['vs']}: {v['verdict']} ({v['score']:+d})")
        lines.extend(f"  {'+' if x['score'] > 0 else '-' if x['score'] < 0 else '·'} {x['name']}: {x['change']} ({x['reading']})"
                     for x in v["factors"] if x["score"])
    c = row.get("confirm")
    if c:
        lines.append(f"Confirm {c['aligned']}/{c['of']}: " + ", ".join(f"{x['name']} {x['value']} {'✓' if x['ok'] else '✗'}" for x in c["items"]))
    return "\n".join(lines)


def backtest(expiry_date):
    start, _ = cycle_dates(expiry_date)
    data = load_hist_cycle(expiry_date) or load_local_cycle(expiry_date)
    if not data:
        return None
    eng = RangeEngine(start, expiry_date).run(data, datetime.combine(expiry_date, dtime(15, 40)))
    return eng.snapshot()


def backtest_all(weeks=8):
    """The last `weeks` completed expiries, replayed from data/hist1m."""
    import simulator
    out = []
    today = mc.now_ist().date()
    done = [e for e in simulator.list_expiries() if e["expiry"] < today.isoformat()][:weeks]
    for e in sorted(done, key=lambda x: x["expiry"]):
        exp = date.fromisoformat(e["expiry"])
        snap = backtest(exp)
        if snap and snap["status"] == "closed":
            exit_ev = next((x for x in snap["events"] if x["type"] == "EXIT"), None)
            out.append({"expiry": e["expiry"], "cycle_start": snap["cycle_start"], "range": snap["events"][0].get("range") if snap["events"] else None,
                        "entry": snap["events"][0] if snap["events"] else None,
                        "rolls": sum(len(x.get("rolls", [])) for x in snap["events"] if x["type"] == "RANGE"),
                        "exit": exit_ev, "reason": snap["exit_reason"],
                        "gross": snap["realized"], "charges": snap["charges"],
                        "net": round(snap["realized"] - snap["charges"], 2)})
    return out


# ---- live runner -------------------------------------------------------------

def _current_cycle(today):
    import fyers_option_chain as chain_mod
    for e in chain_mod.list_expiries():
        d, m, y = e["date"].split("-")
        exp = date(int(y), int(m), int(d))
        if exp >= today:
            start, _ = cycle_dates(exp)
            return e, exp, start
    return None, None, None


def _logged_keys():
    keys = set()
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r.get("expiry"), r.get("time"), r.get("type")))
                except json.JSONDecodeError:
                    continue
    return keys


def fmt_event(ev):
    t = ev["time"].replace("T", " ")
    if ev["type"] == "ENTRY":
        r = ev["range"]
        legs = " + ".join(f"SELL {l['strike']} {l['side']} @{l['price']}" for l in ev["legs"])
        return (f"📐 [{t}] RANGE STRATEGY ENTRY\nLeaders CE {r['ce_leader']} (1st-candle high {r['ce_first_high']}) / "
                f"PE {r['pe_leader']} (high {r['pe_first_high']})\nRange {r['lower']:.0f} - {r['upper']:.0f} "
                f"({r['upper'] - r['lower']:.0f} pts)\n{legs}\nPremium Rs{ev['premium']:,.0f} | stop {'Rs' + format(ev['stop'], ',.0f') if ev.get('stop') is not None else 'none'} | "
                f"target Rs{ev['target']:,.0f}")
    if ev["type"] == "RANGE":
        r = ev["range"]
        head = f"🔁 [{t}] Leaders CE {r['ce_leader']} / PE {r['pe_leader']} -> range {r['lower']:.0f} - {r['upper']:.0f}"
        if not ev["rolls"]:
            return head + " | strikes unchanged"
        rolls = "\n".join(f"ROLL {x['side']}: buy back {x['from']} @{x['buy_back']} ({'+' if x['booked'] >= 0 else ''}{x['booked']:,.0f}), "
                          f"sell {x['to']} @{x['sell']} ({x['why']})" for x in ev["rolls"])
        return f"{head}\n{rolls}\nTotal P&L Rs{ev['total']:,.0f}"
    if ev["type"] == "EXIT":
        legs = ", ".join(f"{l['strike']}{l['side']} @{l['exit']}" for l in ev["legs"])
        return (f"🏁 [{t}] RANGE STRATEGY EXIT ({ev['reason']}): closed {legs}\n"
                f"Gross Rs{ev['realized']:,.0f} | charges Rs{ev['charges']:,.0f} | NET Rs{ev['net']:,.0f}")
    return f"[{t}] {ev['type']}"


def run_once(notify=None):
    """Rebuild the current cycle from exchange history, publish state, log +
    notify new events. Returns the snapshot (or None outside a cycle)."""
    today = mc.now_ist().date()
    expiry_info, exp, start = _current_cycle(today)
    if exp is None or today < start:
        return None
    now = mc.now_ist().replace(tzinfo=None)
    until = now - timedelta(minutes=5, seconds=30)  # only fully-closed candles
    data = load_live_cycle(expiry_info, start, today)
    eng = RangeEngine(start, exp).run(data, until, index=load_index(mc.previous_trading_day(start), today))
    snap = eng.snapshot()
    snap["updated_at"] = mc.now_ist().isoformat(timespec="seconds")
    paths.atomic_write_json(STATE_PATH, snap)
    save_diary(eng, snap["updated_at"])
    # Telegram on every leader change / day start -- whether or not the
    # strategy holds a position (dedupe across restarts via a sent-file).
    try:
        with open(DIARY_SENT_PATH) as f:
            sent = set(json.load(f))
    except (OSError, json.JSONDecodeError):
        sent = set()
    new_sent = False
    for row in eng.diary:
        if row["kind"] not in ("DAY_START", "LEADER_CHANGE"):
            continue
        key = f"{exp.isoformat()}|{row['time']}|{row['kind']}"
        if key in sent:
            continue
        sent.add(key)
        new_sent = True
        if notify and now - datetime.fromisoformat(row["time"]) <= timedelta(minutes=15):
            daily = None
            if row["kind"] == "DAY_START":
                try:
                    by_day = _rows_by_day()
                    dates = sorted(by_day)
                    k = dates.index(row["day"])
                    daily = _daily_view(by_day[row["day"]]["rows"], by_day[dates[k - 1]]["rows"] if k > 0 else None)
                except Exception:
                    daily = None
            notify(fmt_diary(row, daily))
            # Top 3 strategy ideas for this snapshot + follow-up of the earlier
            # ones (snapshot_ideas.py) -- live snapshots only, never on a replay
            try:
                import snapshot_ideas
                txt = snapshot_ideas.on_snapshot(exp.isoformat(), expiry_info["expiry"], row)
                if txt:
                    notify(txt)
            except Exception:
                traceback.print_exc()
    if new_sent:
        paths.atomic_write_json(DIARY_SENT_PATH, sorted(sent))
    # today's DAY_END: mark the day's ideas at the close / settle them on expiry
    # day (also on the after-close catch-up rebuild, which runs without notify)
    for row in eng.diary:
        if row["kind"] == "DAY_END" and row["time"][:10] == today.isoformat():
            try:
                import snapshot_ideas
                snapshot_ideas.on_day_end(exp.isoformat(), expiry_info["expiry"], row)
            except Exception:
                traceback.print_exc()
    keys = _logged_keys()
    for ev in eng.events:
        key = (exp.isoformat(), ev["time"], ev["type"])
        if key in keys:
            continue
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({"expiry": exp.isoformat(), **ev}, default=str) + "\n")
        age = now - datetime.fromisoformat(ev["time"])
        if ev["type"] == "RANGE" and not ev.get("rolls"):
            continue  # the Range Diary's leader-change message already covers it
        if notify and age <= timedelta(minutes=15):
            notify(fmt_event(ev))
    return snap


_thread = None


def _missed_close():
    """True after today's close when the saved state was last built before
    the close's last candle -- e.g. the app was closed at 15:40 -- so the
    day's DAY_END snapshot still needs one catch-up rebuild."""
    now = mc.now_ist()
    close = datetime.combine(now.date(), mc.MARKET_CLOSE) + timedelta(minutes=6)
    if not mc.is_trading_day(now.date()) or now.replace(tzinfo=None) < close:
        return False
    try:
        with open(STATE_PATH) as f:
            upd = datetime.fromisoformat(json.load(f)["updated_at"]).replace(tzinfo=None)
    except (OSError, ValueError, KeyError, TypeError):
        return True
    return upd < close


def _loop():
    import telegram_notify as tg
    while True:
        try:
            if mc.is_market_open():
                run_once(notify=tg.send)
            elif not os.path.exists(STATE_PATH) or _missed_close():
                run_once(notify=None)
        except Exception:
            traceback.print_exc()
        # wake ~40s after each 5-min candle closes
        now = mc.now_ist()
        secs = (5 - now.minute % 5) * 60 - now.second + 40
        time.sleep(max(30, secs))


def start_background():
    global _thread
    if _thread is not None and _thread.is_alive():
        return _thread
    _thread = threading.Thread(target=_loop, daemon=True, name="range-strategy")
    _thread.start()
    return _thread


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        for r in backtest_all():
            print(json.dumps(r, default=str))
    else:
        print(json.dumps(run_once(), indent=1, default=str))
