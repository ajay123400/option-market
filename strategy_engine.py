"""strategy_engine.py -- the highest-volume-strike cross-strangle strategy,
extracted from backtest.py into a reusable, candle-by-candle state machine.

Both the historical backtest AND the live paper-trader feed this the same
way: call `on_candle_close()` once per finished 5-min candle with that
candle's per-strike (high, close, volume-IN-THIS-CANDLE, current OI) for
every CE and PE strike. It returns a list of Event objects describing
whatever happened (leader changes, entries, rolls, EOD exits) -- the caller
(backtest.py's day loop, or paper_trade.py's live loop / Telegram sender)
decides what to do with them (log a row, send a notification, etc).

Rules (interactively validated against 2026-09-02 real data before being
coded here):
  1. First candle of the day: CE_leader / PE_leader = highest-volume CE/PE
     strike in that candle.
  2. Cross targets: CE_target_price = PE_leader's own first-candle high;
     PE_target_price = CE_leader's own first-candle high. These update
     whenever the SOURCE leader (the opposite side) changes.
  3. Target strike = the leader, UNLESS its current price already exceeds
     the target price -- then fall back to the nearest strike (either
     direction) whose current price is below it. This overshoot-fallback
     search only runs on a PENDING leg -- once filled, a leg is never
     re-targeted just because its own price later drifts past the level
     that got it filled.
  4. Leadership smoothing: tracked by CUMULATIVE volume since day open, a
     candidate must also show OI up since day open (fresh positioning, not
     churn), and must beat the current leader's cumulative volume by a 20%
     margin before leadership actually flips (avoids 5-min noise/whipsaw).
  5. Entry: a target leg fills (SELL) the candle its target strike's HIGH
     first touches the target price.
  6. Hedge: the instant a short leg fills, buy 200 points further OTM on
     the same side at market. Rolls/closes take the hedge with them.
  7. Rolling mid-day happens ONLY on a genuine leadership change (the
     smoothed leader in step 4 actually flips to a different strike) while
     a leg is open -- not on the overshoot-fallback in step 3, and not on
     ordinary price movement. This was an explicit correction after an
     earlier version rolled on every overshoot-fallback flip too, which
     produced several same-strike-in/out round trips in one day purely
     from the entry strike's own price oscillating around its target.
  8. EOD: any leg still open squares off (with its hedge) at the last
     candle's close.

Risk controls (added after the audit -- the original rules had none):
  9. Hedge guard: a short leg is only opened if a hedge strike at least
     HEDGE_DISTANCE further OTM exists AND has a price. Previously the hedge
     silently fell back to the furthest strike available (which on a trend
     day could be < 200pt away, or the short strike itself -> no hedge at
     all), and a missing hedge price crashed the paper-trader's Telegram
     formatting.
 10. Per-leg stop-loss: a short leg exits when its candle HIGH reaches
     sl_multiple x entry premium (fill at that level, or at the open on a
     gap through it). That side is then done for the day.
 11. Daily loss cap: once realized + open MTM P&L <= -max_daily_loss, every
     open leg is squared off at the candle close and no new entries are
     taken for the rest of the day.
 12. Fill realism: a target only fills if the candle's high trades at
     least TRADE_THROUGH beyond it (merely touching a limit often doesn't
     fill -- you're behind the queue), and every order pays SLIPPAGE against
     us (sell fills lower, buy fills higher) for entries, hedges, exits.
 Set sl_multiple=None / max_daily_loss=None / trade_through=0 / slippage=0
 to reproduce the original rules.
"""
from dataclasses import dataclass, field
from typing import Optional

LOT_SIZE = 65
LEADER_MARGIN = 1.20
HEDGE_DISTANCE = 200
SL_MULTIPLE = 2.0          # short leg stops out at 2x its entry premium
TRADE_THROUGH = 0.05       # a resting limit only counts as filled if price traded 1 tick THROUGH it
SLIPPAGE = 0.5             # Rs per unit, every order, always against us (half-spread + impact proxy)
MAX_DAILY_LOSS = 10000.0   # Rs, realized + open MTM, per day (1 lot per side)


@dataclass
class Event:
    type: str          # LEADER_CE, LEADER_PE, TARGET_CE, TARGET_PE, ENTRY_CE,
                        # ENTRY_PE, ROLL_EXIT_CE, ROLL_EXIT_PE, EOD_EXIT_CE,
                        # EOD_EXIT_PE, SL_EXIT_CE, SL_EXIT_PE, RISK_EXIT_CE,
                        # RISK_EXIT_PE, DAILY_STOP, ENTRY_SKIPPED_CE,
                        # ENTRY_SKIPPED_PE
    strike: Optional[float] = None
    price: Optional[float] = None
    hedge_strike: Optional[float] = None
    hedge_price: Optional[float] = None
    pnl: Optional[float] = None
    extra: dict = field(default_factory=dict)


class _Side:
    def __init__(self):
        self.leader = None
        self.target_price = None
        self.target_strike = None
        self.status = "pending"   # pending | open
        self.entry_price = None
        self.entry_strike = None
        self.hedge_strike = None
        self.hedge_entry_price = None
        # ITM-neighbor crossing tracker: None until the neighbor has been
        # observed at least once; then True/False for "was it above target
        # last time we looked". Reset whenever target_strike changes, since
        # the neighbor being watched changes too.
        self.itm_neighbor_strike = None
        self.itm_was_above = None
        self.stopped = False        # SL hit -- no more entries this side today
        self.skip_logged_for = None  # target strike an ENTRY_SKIPPED was already reported for


class StrategyEngine:
    def __init__(self, ce_strikes, pe_strikes, lot_size=LOT_SIZE,
                 leader_margin=LEADER_MARGIN, hedge_distance=HEDGE_DISTANCE,
                 sl_multiple=SL_MULTIPLE, max_daily_loss=MAX_DAILY_LOSS,
                 trade_through=TRADE_THROUGH, slippage=SLIPPAGE):
        """ce_strikes / pe_strikes: sorted list of available strike prices
        (float) for the side -- used for the hedge lookup and the
        overshoot-fallback nearest-strike search."""
        self.ce_strikes = sorted(ce_strikes)
        self.pe_strikes = sorted(pe_strikes)
        # Leadership (and its overshoot-fallback search) only ever looks at
        # round, 100-point strikes -- a 50-point "half strike" (23950, say)
        # spiking in volume for one candle is often a one-off/algo print
        # rather than genuine crowd positioning, and round strikes are where
        # institutional liquidity actually concentrates. The hedge leg still
        # searches the FULL strike list (it's risk management, not a signal,
        # so it shouldn't be forced worse just to stay on a round number).
        self.ce_leader_strikes = [s for s in self.ce_strikes if s % 100 == 0]
        self.pe_leader_strikes = [s for s in self.pe_strikes if s % 100 == 0]
        self.lot_size = lot_size
        self.leader_margin = leader_margin
        self.hedge_distance = hedge_distance
        self.sl_multiple = sl_multiple
        self.max_daily_loss = max_daily_loss
        self.halted = False  # daily loss cap hit -- no new entries today
        self.trade_through = trade_through or 0.0
        self.slippage = slippage or 0.0

        # Separate dicts per side -- a CE strike and a PE strike can share
        # the same numeric strike price (e.g. 23800 CE and 23800 PE both
        # exist), so a single strike-keyed dict would silently let one side
        # overwrite the other's volume/OI/close. Every lookup below always
        # knows which side it's on, so this stays simple.
        self.cum_vol = {"CE": {}, "PE": {}}
        self.first_oi = {"CE": {}, "PE": {}}
        self.first_high = {"CE": {}, "PE": {}}
        self.last_oi = {"CE": {}, "PE": {}}
        self.last_close = {"CE": {}, "PE": {}}

        self.ce = _Side()
        self.pe = _Side()
        self.day_pnl = 0.0
        self.num_rolls = 0

    # ---- strike universe ---------------------------------------------------

    def update_strikes(self, ce_strikes=(), pe_strikes=(), seeds=None):
        """Adds strikes that came into range mid-day (the chain window
        follows spot, so a trend day brings new strikes in). `seeds`:
        {("CE"|"PE", strike): {"first_high":, "first_oi":, "cum_vol":,
        "last_close":, "last_oi":}} computed from that strike's own history
        up to now, so a late-added strike's "first candle high" / OI-since-
        open / cumulative volume are the real day's values, not whatever
        candle it happened to first be seen on."""
        self.ce_strikes = sorted(set(self.ce_strikes) | set(ce_strikes))
        self.pe_strikes = sorted(set(self.pe_strikes) | set(pe_strikes))
        self.ce_leader_strikes = [s for s in self.ce_strikes if s % 100 == 0]
        self.pe_leader_strikes = [s for s in self.pe_strikes if s % 100 == 0]
        for (opt_side, strike), sd in (seeds or {}).items():
            if strike in self.first_high[opt_side]:
                continue
            self.first_high[opt_side][strike] = sd["first_high"]
            self.first_oi[opt_side][strike] = sd["first_oi"]
            self.cum_vol[opt_side][strike] = sd["cum_vol"]
            self.last_close[opt_side][strike] = sd["last_close"]
            self.last_oi[opt_side][strike] = sd["last_oi"]

    # ---- internal helpers -------------------------------------------------

    def _hedge_strike_for(self, short_strike, side):
        """Nearest strike at least hedge_distance further OTM, or None if
        the known strike list doesn't reach that far (the caller then
        refuses the entry rather than trading a fake/too-close hedge)."""
        if side == "CE":
            cands = [s for s in self.ce_strikes if s >= short_strike + self.hedge_distance]
            return min(cands) if cands else None
        cands = [s for s in self.pe_strikes if s <= short_strike - self.hedge_distance]
        return max(cands) if cands else None

    def _leg_mtm(self, side: _Side, opt_side):
        if side.status != "open":
            return 0.0
        lc = self.last_close[opt_side]
        pnl = (side.entry_price - lc.get(side.entry_strike, side.entry_price)) * self.lot_size
        if side.hedge_strike is not None and side.hedge_entry_price is not None:
            pnl += (lc.get(side.hedge_strike, side.hedge_entry_price) - side.hedge_entry_price) * self.lot_size
        return pnl

    def _check_stops(self, ce_candles, pe_candles, events):
        """Rules 10-11: per-leg SL, then the daily loss cap."""
        if self.sl_multiple:
            for side, candles, opt_side in ((self.ce, ce_candles, "CE"), (self.pe, pe_candles, "PE")):
                if side.status != "open" or side.entry_strike not in candles:
                    continue
                c = candles[side.entry_strike]
                sl_level = round(side.entry_price * self.sl_multiple, 2)
                if c["high"] >= sl_level:
                    fill = c["open"] if c.get("open") is not None and c["open"] >= sl_level else sl_level
                    self._close_leg(side, f"SL_EXIT_{opt_side}", events, opt_side, exit_price=fill)
                    side.stopped = True
        if self.max_daily_loss and not self.halted:
            total = self.day_pnl + self._leg_mtm(self.ce, "CE") + self._leg_mtm(self.pe, "PE")
            if total <= -self.max_daily_loss:
                for side, opt_side in ((self.ce, "CE"), (self.pe, "PE")):
                    if side.status == "open":
                        self._close_leg(side, f"RISK_EXIT_{opt_side}", events, opt_side)
                self.halted = True
                events.append(Event(type="DAILY_STOP", pnl=round(self.day_pnl, 2),
                                    extra={"limit": self.max_daily_loss}))

    def _itm_neighbor(self, strike, opt_side, strikes):
        """One step further ITM than `strike` -- lower for a call (a lower
        strike is more in-the-money), higher for a put -- clamped to the
        nearest strike actually available. Used as a second fill candidate
        for a target that structurally may never reach its price: a far-OTM
        strike (like the leader) can sit well below a rich target price all
        day, while a richer, nearer-the-money strike may plausibly drift
        DOWN through that same price instead of needing to rise up to it."""
        step = -100 if opt_side == "CE" else 100
        wanted = strike + step
        if opt_side == "CE":
            cands = [s for s in strikes if s <= wanted]
            return max(cands) if cands else None
        else:
            cands = [s for s in strikes if s >= wanted]
            return min(cands) if cands else None

    def _leader_by_smoothed(self, strikes, current, opt_side):
        cum_vol = self.cum_vol[opt_side]
        first_oi = self.first_oi[opt_side]
        last_oi = self.last_oi[opt_side]
        candidates = {}
        for s in strikes:
            if s in last_oi and s in first_oi and cum_vol.get(s, 0) > 0:
                if last_oi[s] - first_oi[s] > 0:
                    candidates[s] = cum_vol[s]
        if not candidates:
            return current
        best = max(candidates, key=candidates.get)
        if current is None:
            return best
        if best != current and candidates.get(best, 0) >= cum_vol.get(current, 0) * self.leader_margin:
            return best
        return current

    def _overshoot_fallback(self, strikes, leader, target_price, opt_side):
        if leader is None or target_price is None:
            return leader
        last_close = self.last_close[opt_side]
        price = last_close.get(leader)
        if price is None or price < target_price:
            return leader
        below = [s for s in strikes if s in last_close and last_close[s] < target_price]
        if not below:
            return leader
        below.sort(key=lambda s: abs(s - leader))
        return below[0]

    def _buy_px(self, p):
        return None if p is None else round(p + self.slippage, 2)

    def _sell_px(self, p):
        return None if p is None else round(max(0.05, p - self.slippage), 2)

    def _close_leg(self, side: _Side, tag_prefix, events, opt_side, exit_price=None):
        last_close = self.last_close[opt_side]
        if exit_price is None:
            exit_price = last_close.get(side.entry_strike)
        hedge_exit = self._sell_px(last_close.get(side.hedge_strike))  # hedge is SOLD to close
        if exit_price is None:
            events.append(Event(type="EXIT_PRICE_MISSING", strike=side.entry_strike,
                                extra={"wanted": tag_prefix}))
            return
        exit_price = self._buy_px(exit_price)  # short is BOUGHT back
        pnl = (side.entry_price - exit_price) * self.lot_size
        if side.hedge_strike is not None and hedge_exit is not None and side.hedge_entry_price is not None:
            pnl += (hedge_exit - side.hedge_entry_price) * self.lot_size
        self.day_pnl += pnl
        events.append(Event(type=tag_prefix, strike=side.entry_strike, price=exit_price,
                             hedge_strike=side.hedge_strike, hedge_price=hedge_exit, pnl=round(pnl, 2)))
        side.status = "pending"
        side.entry_price = None
        side.entry_strike = None
        side.hedge_strike = None
        side.hedge_entry_price = None

    def _open_leg(self, side: _Side, tag, events, opt_side, fill_price):
        """Returns False (leg stays pending) if no valid hedge exists --
        rule 9. The skip is reported once per target strike, not every
        candle the target keeps re-triggering."""
        hedge_strike = self._hedge_strike_for(side.target_strike, opt_side)
        hedge_price = self._buy_px(self.last_close[opt_side].get(hedge_strike)) if hedge_strike is not None else None
        fill_price = self._sell_px(fill_price)
        if hedge_strike is None or hedge_price is None:
            if side.skip_logged_for != side.target_strike:
                side.skip_logged_for = side.target_strike
                events.append(Event(type=f"ENTRY_SKIPPED_{opt_side}", strike=side.target_strike, price=fill_price,
                                    extra={"reason": "no hedge strike >= %dpt OTM with a price" % self.hedge_distance}))
            return False
        side.status = "open"
        side.entry_price = fill_price
        side.entry_strike = side.target_strike
        side.hedge_strike = hedge_strike
        side.hedge_entry_price = hedge_price
        events.append(Event(type=tag, strike=side.target_strike, price=fill_price,
                             hedge_strike=hedge_strike, hedge_price=hedge_price))
        return True

    @staticmethod
    def _fill_price(candle, target_price):
        """A resting sell-limit at target_price fills at target_price when
        price rises THROUGH it mid-candle (open was below target, high
        touched/passed it) -- standard limit-order behaviour. But if the
        candle GAPS straight past the limit (open already >= target), the
        realistic fill is at that open print, not the now-stale target --
        selling exactly at target when the market opened well above it
        understates the premium actually collectible there."""
        open_ = candle.get("open")
        if open_ is not None and open_ >= target_price:
            return open_
        return target_price

    @staticmethod
    def _fill_price_breakdown(candle, target_price):
        """Mirror of _fill_price for a price that was ABOVE target and is
        now crossing DOWN through it -- the ITM-neighbor breakdown trigger.
        A gap-down candle (open already <= target) fills at that lower
        open; otherwise it fills exactly at target_price as price crosses
        down through it mid-candle."""
        open_ = candle.get("open")
        if open_ is not None and open_ <= target_price:
            return open_
        return target_price

    def _check_itm_neighbor(self, side: _Side, candles, opt_side, strikes):
        """Returns a (fill_strike, fill_price) tuple if the neighbor should
        fill THIS candle, else (None, None). Never fires on the first
        candle a neighbor is observed just because it already happens to
        be past target -- only a genuine crossing (rising through target
        from below, or breaking down through it from above) triggers a
        fill. This is deliberately different from the primary target
        strike's plain 'high >= target' check: the neighbor is a fallback
        being newly watched mid-day, so its price already being wherever
        it is when we start watching carries no signal on its own."""
        itm = self._itm_neighbor(side.target_strike, opt_side, strikes)
        if itm is None or itm == side.target_strike or itm not in candles:
            return None, None
        if side.itm_neighbor_strike != itm:
            side.itm_neighbor_strike = itm
            side.itm_was_above = None  # freshly watching a (possibly new) neighbor

        c = candles[itm]
        target = side.target_price
        was_above = side.itm_was_above
        is_above_now = c["close"] >= target

        fill_strike, fill_price = None, None
        if was_above is None:
            # First observation of this neighbor: if it's already below
            # target, a rise-through in this SAME candle still counts
            # (nothing "already there" about rising further within the
            # first candle we happen to look). If it's already above,
            # just record that and wait for a real breakdown later.
            if not is_above_now and c["high"] >= target + self.trade_through:
                fill_strike, fill_price = itm, self._fill_price(c, target)
        elif was_above and c["low"] <= target - self.trade_through:
            fill_strike, fill_price = itm, self._fill_price_breakdown(c, target)
        elif not was_above and c["high"] >= target + self.trade_through:
            fill_strike, fill_price = itm, self._fill_price(c, target)

        side.itm_was_above = is_above_now
        return fill_strike, fill_price

    # ---- public API ---------------------------------------------------

    def on_candle_close(self, ce_candles: dict, pe_candles: dict) -> list:
        """ce_candles / pe_candles: {strike: {'open':, 'high':, 'close':, 'volume':, 'oi':}}
        for every strike with data in THIS candle. 'open' is used only to
        detect a gap-through-the-limit fill (see _fill_price) -- callers
        that genuinely can't supply it may omit the key, and every fill
        will fall back to filling exactly at the target price. volume is this candle's
        own volume (not cumulative) -- if your data source gives cumulative
        day volume, subtract the previous cumulative value yourself before
        calling this. Returns a list of Event.
        """
        events = []

        for strike, c in ce_candles.items():
            self.cum_vol["CE"][strike] = self.cum_vol["CE"].get(strike, 0) + c["volume"]
            self.last_oi["CE"][strike] = c["oi"]
            self.last_close["CE"][strike] = c["close"]
            self.first_oi["CE"].setdefault(strike, c["oi"])
            self.first_high["CE"].setdefault(strike, c["high"])
        for strike, c in pe_candles.items():
            self.cum_vol["PE"][strike] = self.cum_vol["PE"].get(strike, 0) + c["volume"]
            self.last_oi["PE"][strike] = c["oi"]
            self.last_close["PE"][strike] = c["close"]
            self.first_oi["PE"].setdefault(strike, c["oi"])
            self.first_high["PE"].setdefault(strike, c["high"])

        self._check_stops(ce_candles, pe_candles, events)

        new_ce_leader = self._leader_by_smoothed(self.ce_leader_strikes, self.ce.leader, "CE")
        new_pe_leader = self._leader_by_smoothed(self.pe_leader_strikes, self.pe.leader, "PE")

        if new_ce_leader != self.ce.leader:
            events.append(Event(type="LEADER_CE", strike=new_ce_leader, extra={"prev": self.ce.leader}))
            self.ce.leader = new_ce_leader
            if self.pe.leader:
                self.pe.target_price = self.first_high["CE"].get(self.ce.leader)
            # A genuine leadership shift (not the leader's own price merely
            # drifting past its target -- that's the overshoot-fallback
            # case below, which does NOT roll) away from an OPEN position
            # is the only thing that rolls a leg mid-day.
            if self.ce.status == "open" and self.ce.entry_strike != new_ce_leader:
                self._close_leg(self.ce, "ROLL_EXIT_CE", events, "CE")
                self.num_rolls += 1
        if new_pe_leader != self.pe.leader:
            events.append(Event(type="LEADER_PE", strike=new_pe_leader, extra={"prev": self.pe.leader}))
            self.pe.leader = new_pe_leader
            if self.ce.leader:
                self.ce.target_price = self.first_high["PE"].get(self.pe.leader)
            if self.pe.status == "open" and self.pe.entry_strike != new_pe_leader:
                self._close_leg(self.pe, "ROLL_EXIT_PE", events, "PE")
                self.num_rolls += 1

        if self.ce.leader and self.pe.leader and self.ce.target_price is None:
            self.ce.target_price = self.first_high["PE"].get(self.pe.leader)
        if self.ce.leader and self.pe.leader and self.pe.target_price is None:
            self.pe.target_price = self.first_high["CE"].get(self.ce.leader)

        # Target-strike re-selection (including the overshoot-fallback
        # search) only runs while a leg is still PENDING. Once a leg is
        # OPEN, it is held as-is until EOD square-off, no matter how
        # leadership or price moves afterward -- no more rolling mid-day.
        if self.ce.target_price is not None and self.ce.leader and self.ce.status != "open":
            new_strike = self._overshoot_fallback(self.ce_leader_strikes, self.ce.leader, self.ce.target_price, "CE")
            if new_strike != self.ce.target_strike:
                self.ce.target_strike = new_strike
                events.append(Event(type="TARGET_CE", strike=new_strike, price=self.ce.target_price))
        if self.pe.target_price is not None and self.pe.leader and self.pe.status != "open":
            new_strike = self._overshoot_fallback(self.pe_leader_strikes, self.pe.leader, self.pe.target_price, "PE")
            if new_strike != self.pe.target_strike:
                self.pe.target_strike = new_strike
                events.append(Event(type="TARGET_PE", strike=new_strike, price=self.pe.target_price))

        # Entry check: the target strike itself rising up to the target
        # price is the primary path. But a far-OTM target can structurally
        # sit below a rich target price all day and never fill -- so a
        # richer, one-step-more-ITM neighbor is checked too, on the chance
        # IT drifts down through that same price instead. Filling via the
        # neighbor switches the actually-traded strike there and, following
        # the same convention used everywhere else in this engine, updates
        # the OPPOSITE leg's target price to that neighbor's own
        # first-candle high (whichever CE/PE strike actually gets traded is
        # what the other side's price should be cross-referenced against).
        if self.ce.status == "pending" and self.ce.target_strike is not None and not (self.halted or self.ce.stopped):
            fill_strike, fill_price = None, None
            if self.ce.target_strike in ce_candles and ce_candles[self.ce.target_strike]["high"] >= self.ce.target_price + self.trade_through:
                fill_strike = self.ce.target_strike
                fill_price = self._fill_price(ce_candles[fill_strike], self.ce.target_price)
            else:
                fill_strike, fill_price = self._check_itm_neighbor(self.ce, ce_candles, "CE", self.ce_leader_strikes)
            if fill_strike is not None:
                prev_target = self.ce.target_strike
                self.ce.target_strike = fill_strike
                if self._open_leg(self.ce, "ENTRY_CE", events, "CE", fill_price):
                    if fill_strike != prev_target:
                        events.insert(len(events) - 1, Event(type="TARGET_CE", strike=fill_strike,
                                                             price=self.ce.target_price,
                                                             extra={"itm_fallback": True, "from": prev_target}))
                        new_price = self.first_high["CE"].get(fill_strike)
                        if new_price is not None:
                            self.pe.target_price = new_price
                else:
                    self.ce.target_strike = prev_target

        if self.pe.status == "pending" and self.pe.target_strike is not None and not (self.halted or self.pe.stopped):
            fill_strike, fill_price = None, None
            if self.pe.target_strike in pe_candles and pe_candles[self.pe.target_strike]["high"] >= self.pe.target_price + self.trade_through:
                fill_strike = self.pe.target_strike
                fill_price = self._fill_price(pe_candles[fill_strike], self.pe.target_price)
            else:
                fill_strike, fill_price = self._check_itm_neighbor(self.pe, pe_candles, "PE", self.pe_leader_strikes)
            if fill_strike is not None:
                prev_target = self.pe.target_strike
                self.pe.target_strike = fill_strike
                if self._open_leg(self.pe, "ENTRY_PE", events, "PE", fill_price):
                    if fill_strike != prev_target:
                        events.insert(len(events) - 1, Event(type="TARGET_PE", strike=fill_strike,
                                                             price=self.pe.target_price,
                                                             extra={"itm_fallback": True, "from": prev_target}))
                        new_price = self.first_high["PE"].get(fill_strike)
                        if new_price is not None:
                            self.ce.target_price = new_price
                else:
                    self.pe.target_strike = prev_target

        return events

    def force_eod_exit(self) -> list:
        events = []
        if self.ce.status == "open":
            self._close_leg(self.ce, "EOD_EXIT_CE", events, "CE")
        if self.pe.status == "open":
            self._close_leg(self.pe, "EOD_EXIT_PE", events, "PE")
        return events

    def status_snapshot(self):
        return {
            "ce_leader": self.ce.leader, "ce_target_strike": self.ce.target_strike,
            "ce_target_price": self.ce.target_price, "ce_status": self.ce.status,
            "ce_entry_strike": self.ce.entry_strike, "ce_entry_price": self.ce.entry_price,
            "ce_hedge_strike": self.ce.hedge_strike, "ce_hedge_price": self.ce.hedge_entry_price,
            "pe_leader": self.pe.leader, "pe_target_strike": self.pe.target_strike,
            "pe_target_price": self.pe.target_price, "pe_status": self.pe.status,
            "pe_entry_strike": self.pe.entry_strike, "pe_entry_price": self.pe.entry_price,
            "pe_hedge_strike": self.pe.hedge_strike, "pe_hedge_price": self.pe.hedge_entry_price,
            "day_pnl": round(self.day_pnl, 2), "num_rolls": self.num_rolls,
            "open_mtm": round(self._leg_mtm(self.ce, "CE") + self._leg_mtm(self.pe, "PE"), 2),
            "halted": self.halted, "ce_stopped": self.ce.stopped, "pe_stopped": self.pe.stopped,
        }
