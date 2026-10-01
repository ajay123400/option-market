"""paper_trade.py -- live (no real orders) forward-test of the highest-
volume-strike cross-strangle strategy (strategy_engine.StrategyEngine --
the exact engine backtest.py uses).

Candles come from Fyers' own 5-min history endpoint (true exchange OHLCV +
OI per candle), fetched right after each candle closes -- NOT built from
15-second LTP polling as before. That fixes three problems at once:
  - highs/lows are the real ones (polling missed intra-candle extremes, so
    entry/SL "touch" checks were unreliable),
  - volume is the real per-candle volume,
  - a late start or a mid-day restart simply REPLAYS every candle already
    finished today, so the leader/first-candle rules see the real 09:15
    candle instead of "whatever candle the app happened to start on".
    Events already logged by an earlier run today are not logged or sent
    again (the engine is deterministic, so the replay reaches the same
    state).

The strike universe follows spot: every cycle the chain window
(STRIKECOUNT each side of ATM) is re-read and any new strike is added to
the engine seeded with its OWN day-so-far history (see
StrategyEngine.update_strikes).

Runs every trading day (NSE calendar, IST) until the process exits; skips
weekends/holidays. Last candle processed is the 15:20 one (square-off at its
close, 15:25) -- identical to backtest.py's SQUARE_OFF_TIME cut.
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime, time as dtime, timedelta

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import fyers_auth
import fyers_option_chain as chain_mod
import historical_recorder
import market_calendar as mc
import paths
import telegram_notify as tg
from strategy_engine import StrategyEngine

BASE = paths.BASE_DIR
LOG_PATH = os.path.join(BASE, "results", "paper_trade_log.jsonl")
STATE_PATH = os.path.join(BASE, "results", "paper_state.json")  # live engine state for the Option Chain page
STRIKECOUNT = 20                # strikes each side of ATM tracked (±1000pt)
CANDLE_MINUTES = 5
SQUARE_OFF_TIME = dtime(15, 25)  # last candle processed = 15:20 (closes 15:25)
SETTLE_DELAY_SEC = 30            # wait after a candle closes before trusting its history row
REQUEST_SPACING_SEC = 0.13       # ~7.5 req/s -- under Fyers' 10/s data limit
LIVE_EVENT_MAX_AGE = timedelta(minutes=10)  # older replayed events aren't pushed to Telegram one by one


def _f(v):
    return "?" if v is None else f"{v:.2f}"


def fmt_event(e, candle_time):
    t = candle_time.strftime("%H:%M")
    side = "CALL" if e.type.endswith("CE") else "PUT"
    if e.type in ("LEADER_CE", "LEADER_PE"):
        prev = e.extra.get("prev")
        arrow = f"{prev} -> {e.strike}" if prev else f"{e.strike} (first)"
        return f"📊 [{t}] {side} leader: {arrow}"
    if e.type in ("TARGET_CE", "TARGET_PE"):
        return f"🎯 [{t}] {side} target set: SELL {e.strike} @ Rs{_f(e.price)} (waiting for high to touch)"
    if e.type in ("ENTRY_CE", "ENTRY_PE"):
        return (f"✅ [{t}] ENTRY: Sold {side} {e.strike} @ Rs{_f(e.price)}\n"
                f"   Hedge: Bought {side} {e.hedge_strike} @ Rs{_f(e.hedge_price)}")
    if e.type.startswith("ENTRY_SKIPPED_"):
        return f"⏭ [{t}] {side} entry at {e.strike} SKIPPED -- {e.extra.get('reason')}"
    labels = {"ROLL_EXIT": "🔄 ROLL EXIT", "EOD_EXIT": "🏁 EOD EXIT",
              "SL_EXIT": "🛑 STOP-LOSS EXIT", "RISK_EXIT": "🧯 DAILY-LOSS-CAP EXIT"}
    for prefix, label in labels.items():
        if e.type.startswith(prefix):
            return (f"{label} [{t}]: Closed {side} {e.strike} @ Rs{_f(e.price)} "
                    f"(+hedge {e.hedge_strike} @ Rs{_f(e.hedge_price)}) | P&L: Rs{_f(e.pnl)}")
    if e.type == "DAILY_STOP":
        return f"🧯 [{t}] Daily loss cap (Rs{e.extra.get('limit'):,.0f}) hit -- no more entries today. Day P&L: Rs{_f(e.pnl)}"
    if e.type == "EXIT_PRICE_MISSING":
        return f"⚠️ [{t}] Could not exit {e.strike} ({e.extra.get('wanted')}) -- no price for it yet"
    return f"[{t}] {e.type}: {e}"


def _event_key(row):
    return (row.get("time"), row.get("type"), row.get("strike"))


def _logged_keys_for(day):
    keys = set()
    if not os.path.exists(LOG_PATH):
        return keys
    prefix = day.isoformat()
    with open(LOG_PATH) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (row.get("time") or "").startswith(prefix):
                keys.add(_event_key(row))
    return keys


def log_event(e, candle_time):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    row = {"time": candle_time.strftime("%Y-%m-%dT%H:%M:%S"), "type": e.type, "strike": e.strike,
           "price": e.price, "hedge_strike": e.hedge_strike, "hedge_price": e.hedge_price, "pnl": e.pnl}
    if e.extra:
        row["extra"] = e.extra
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def fetch_day_candles(symbol, day):
    """{candle_start (naive IST datetime): {open,high,low,close,volume,oi}}"""
    rows = historical_recorder._fetch_history_candles(symbol, day.isoformat(), day.isoformat())
    out = {}
    for r in rows:
        ts = datetime.fromtimestamp(r[0], mc.IST).replace(tzinfo=None)
        out[ts] = {"open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5],
                   "oi": r[6] if len(r) > 6 else 0}
    return out


class DayRunner:
    def __init__(self, day):
        self.day = day
        self.expiry = chain_mod.list_expiries()[0]
        self.symbols = {}      # ("CE"|"PE", strike) -> fyers symbol
        self.data = {}         # ("CE"|"PE", strike) -> {ts: candle}
        self.registered = set()
        self.engine = None
        self.last_ts = None
        self.logged = _logged_keys_for(day)
        self.done = False

    def _refresh_universe(self):
        chain = chain_mod.get_chain(strikecount=STRIKECOUNT, expiry_timestamp=self.expiry["expiry"])
        for row in chain["strikes"]:
            for key, opt in (("ce", "CE"), ("pe", "PE")):
                if row[key] and row[key].get("symbol"):
                    self.symbols[(opt, float(row["strike"]))] = row[key]["symbol"]

    def _fetch_all(self):
        for key, sym in self.symbols.items():
            try:
                self.data[key] = fetch_day_candles(sym, self.day)
            except Exception:
                pass  # keep the previous fetch for this symbol
            time.sleep(REQUEST_SPACING_SEC)

    def _seed_for(self, key, before_ts):
        rows = [(ts, c) for ts, c in sorted(self.data.get(key, {}).items()) if ts < before_ts]
        if not rows:
            return None
        return {"first_high": rows[0][1]["high"], "first_oi": rows[0][1]["oi"],
                "cum_vol": sum(c["volume"] for _, c in rows),
                "last_close": rows[-1][1]["close"], "last_oi": rows[-1][1]["oi"]}

    def _register_new(self, before_ts):
        new = [k for k in self.symbols if k not in self.registered and self.data.get(k)]
        if not new:
            return
        if self.engine is None:
            self.engine = StrategyEngine([k[1] for k in new if k[0] == "CE"], [k[1] for k in new if k[0] == "PE"])
        else:
            seeds = {k: sd for k in new if (sd := self._seed_for(k, before_ts))}
            self.engine.update_strikes([k[1] for k in new if k[0] == "CE"],
                                       [k[1] for k in new if k[0] == "PE"], seeds)
        self.registered.update(new)

    def _emit(self, events, ts, now):
        live = (now - ts) <= LIVE_EVENT_MAX_AGE + timedelta(minutes=CANDLE_MINUTES)
        for e in events:
            row = {"time": ts.strftime("%Y-%m-%dT%H:%M:%S"), "type": e.type, "strike": e.strike}
            if _event_key(row) in self.logged:
                continue  # already logged (and sent) by an earlier run today
            log_event(e, ts)
            self.logged.add(_event_key(row))
            msg = fmt_event(e, ts)
            print(msg)
            if live:
                tg.send(msg)
        return live

    def publish_state(self, finished=False):
        """Writes the engine's CURRENT view (real leaders, targets, open
        shorts/hedges, day P&L) for the Option Chain page to highlight --
        the chain used to guess the leader itself from raw volume, which
        often wasn't the strike the algo was actually using."""
        if self.engine is None:
            return
        paths.atomic_write_json(STATE_PATH, {
            "date": self.day.isoformat(),
            "expiry_ts": str(self.expiry["expiry"]),
            "expiry_date": self.expiry["date"],
            "last_candle": self.last_ts.strftime("%H:%M") if self.last_ts else None,
            "updated_at": mc.now_ist().isoformat(timespec="seconds"),
            "finished": finished,
            **self.engine.status_snapshot(),
        })

    def cycle(self):
        """Processes every candle that has closed since the last cycle.
        Returns True once the day is finished (EOD square-off done)."""
        now = mc.now_ist().replace(tzinfo=None)
        self._refresh_universe()
        self._fetch_all()
        cutoff = now - timedelta(minutes=CANDLE_MINUTES, seconds=SETTLE_DELAY_SEC)
        pending = sorted({ts for d in self.data.values() for ts in d
                          if ts <= cutoff and ts.time() < SQUARE_OFF_TIME
                          and (self.last_ts is None or ts > self.last_ts)})
        replayed = 0
        for ts in pending:
            self._register_new(ts)
            if self.engine is None:
                continue
            ce = {k[1]: d[ts] for k, d in self.data.items() if k[0] == "CE" and k in self.registered and ts in d}
            pe = {k[1]: d[ts] for k, d in self.data.items() if k[0] == "PE" and k in self.registered and ts in d}
            if not self._emit(self.engine.on_candle_close(ce, pe), ts, now):
                replayed += 1
            self.last_ts = ts
        if replayed:
            snap = self.engine.status_snapshot()
            tg.send(f"⏩ Paper trading caught up {replayed} earlier candle(s) from exchange history. "
                    f"CE: {snap['ce_status']} {snap['ce_entry_strike'] or ''} | PE: {snap['pe_status']} "
                    f"{snap['pe_entry_strike'] or ''} | Day P&L Rs{snap['day_pnl']:.2f}")

        self.publish_state()
        last_candle_start = datetime.combine(self.day, SQUARE_OFF_TIME) - timedelta(minutes=CANDLE_MINUTES)
        if self.engine is not None and self.last_ts is not None and self.last_ts >= last_candle_start:
            self._emit(self.engine.force_eod_exit(), self.last_ts, now)
            self.publish_state(finished=True)
            snap = self.engine.status_snapshot()
            summary = (f"📅 Day summary {self.day} -- Total P&L: Rs{snap['day_pnl']:.2f} (gross, before charges), "
                       f"rolls: {snap['num_rolls']}" + (" | daily loss cap hit" if snap["halted"] else ""))
            print(summary)
            if ("SUMMARY", self.day.isoformat()) not in self.logged:
                tg.send(summary)
                self.logged.add(("SUMMARY", self.day.isoformat()))
            return True
        return False


def _sleep_until(dt_naive_ist):
    while True:
        remaining = (dt_naive_ist - mc.now_ist().replace(tzinfo=None)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30))


def run_day(day):
    runner = DayRunner(day)
    tg.send(f"🚀 Paper trading {day} -- NIFTY {runner.expiry['date']} expiry, ±{STRIKECOUNT} strikes, "
            f"SL {runner_param('sl_multiple')}x premium, daily loss cap Rs{runner_param('max_daily_loss'):,.0f}.")
    while True:
        try:
            if runner.cycle():
                return
        except Exception as e:
            traceback.print_exc()
            tg.send(f"⚠️ Paper trading cycle failed ({type(e).__name__}: {e}) -- retrying next candle.")
        now = mc.now_ist().replace(tzinfo=None)
        base = datetime.combine(day, mc.MARKET_OPEN)
        n = int((now - base).total_seconds() // (CANDLE_MINUTES * 60)) + 1
        _sleep_until(base + timedelta(minutes=n * CANDLE_MINUTES, seconds=SETTLE_DELAY_SEC))


def runner_param(name):
    import strategy_engine
    return {"sl_multiple": strategy_engine.SL_MULTIPLE, "max_daily_loss": strategy_engine.MAX_DAILY_LOSS}[name]


def main():
    """Runs every trading day for as long as the process lives."""
    print("Logging in to Fyers (automated TOTP flow)...")
    fyers_auth.login()
    print("Fyers login OK.")
    while True:
        now = mc.now_ist()
        today = now.date()
        first_candle_done = datetime.combine(today, mc.MARKET_OPEN) + timedelta(
            minutes=CANDLE_MINUTES, seconds=SETTLE_DELAY_SEC)
        if mc.is_trading_day(today) and now.time() < SQUARE_OFF_TIME:
            print(f"Paper trading {today}: waiting for the first candle to close..."
                  if now.replace(tzinfo=None) < first_candle_done else f"Paper trading {today}: catching up + live.")
            _sleep_until(first_candle_done)
            run_day(today)
        # sleep until 09:00 IST on the next trading day
        nxt = today + timedelta(days=1)
        while not mc.is_trading_day(nxt):
            nxt += timedelta(days=1)
        print(f"Next paper-trading session: {nxt}")
        _sleep_until(datetime.combine(nxt, dtime(9, 0)))


if __name__ == "__main__":
    main()
