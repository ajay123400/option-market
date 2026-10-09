"""manual_monitor.py -- server-side SL/Target/intraday-square-off
enforcement and expiry settlement for the manual Strategy Builder's open
legs, independent of whether any browser tab is open.

Runs as a daemon thread inside the app (start_background_monitor(), called
from app.py and launcher.py) and can still be run standalone
(`python manual_monitor.py`). It publishes a heartbeat (status()) that the
Strategy Builder shows, so "is my stop-loss actually being watched?" is
visible at a glance. The page's own client-side check is only a fallback
for when this heartbeat is stale.

Rules:
- SL / Target: triggered when LTP crosses the level; the leg is closed AT
  THE LTP that crossed it, not at the level -- a stop is a market exit,
  and booking the level itself understated every gap-through loss.
- INTRADAY legs square off between SQUARE_OFF_TIME and MARKET_CLOSE (IST)
  on weekdays -- 15 min before NSE's F&O close (15:40 since 2026-08-03).
- Expired legs (any trade type) are settled at intrinsic value against the
  index's expiry-day close -- manual_trades.settle_expired_legs().

All times are IST regardless of the machine's timezone.
"""
import sys
import threading
import time
import traceback

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import alerts
import fyers_option_chain as chain_mod
import manual_trades
import market_calendar as mc
import telegram_notify as tg

IST = mc.IST
POLL_INTERVAL_SEC = 10
IDLE_POLL_INTERVAL_SEC = 30  # no open legs / market closed -- no point hammering Fyers
HEARTBEAT_STALE_SEC = 60
ERROR_ALERT_AFTER = 3  # consecutive failed passes before a Telegram alert

_thread = None
_stop_event = None
_state = {"started_at": None, "last_beat": None, "last_error": None, "last_error_at": None,
          "consecutive_errors": 0, "open_legs": 0, "market_open": False, "unpriced_legs": []}
_state_lock = threading.Lock()


def _now():
    return mc.now_ist()


def is_market_hours(now=None):
    """NSE F&O session, IST, trading days only (weekends + NSE holidays excluded)."""
    return mc.is_market_open(now or _now())


def is_square_off_window(now=None):
    return mc.is_square_off_window(now or _now())


def check_and_close(leg, ltp, now=None):
    """Returns (record, message) if the leg was closed this pass, else
    (None, None). Exits always book at `ltp`."""
    if leg.get("trade_type") == "INTRADAY" and is_square_off_window(now):
        rec = manual_trades.close_leg(leg["id"], ltp, exit_reason="SQUARE_OFF")
        return rec, f"Intraday square-off (15:25): {leg['side']} {leg['strike']}{leg['type']} @ {ltp}"

    sl, target, short = leg.get("sl"), leg.get("target"), leg["side"] == "SELL"
    if sl is not None and (ltp >= sl if short else ltp <= sl):
        rec = manual_trades.close_leg(leg["id"], ltp, exit_reason="SL", extra={"trigger": sl})
        return rec, f"SL hit: {leg['side']} {leg['strike']}{leg['type']} (SL {sl}) exited @ {ltp}"
    if target is not None and (ltp <= target if short else ltp >= target):
        rec = manual_trades.close_leg(leg["id"], ltp, exit_reason="TARGET", extra={"trigger": target})
        return rec, f"Target hit: {leg['side']} {leg['strike']}{leg['type']} (target {target}) exited @ {ltp}"
    return None, None


def _notify(msg, record=None):
    print(msg)
    pnl = f" | P&L: Rs{record['pnl']:.2f}" if record and record.get("pnl") is not None else ""
    tg.send(f"🔔 [risk monitor] {msg}{pnl}")


def run_one_pass():
    """One enforcement pass. Returns the number of legs still open."""
    now = _now()
    for rec in manual_trades.square_off_stale_intraday():
        _notify(f"Late intraday square-off: {rec['side']} {rec['strike']}{rec['type']} booked @ {rec['exit_price']} "
                f"(its 15:25 price on {rec['exit_time'][:10]} -- the app wasn't running then)", rec)
    for rec in manual_trades.settle_expired_legs():
        _notify(f"Expiry settlement: {rec['side']} {rec['strike']}{rec['type']} "
                f"settled @ {rec['exit_price']} (NIFTY close {rec.get('settlement_price')})", rec)

    positions = [p for p in manual_trades.load_positions() if not manual_trades.is_leg_expired(p, now)]
    with _state_lock:
        _state["market_open"] = is_market_hours(now)
    if not is_market_hours(now):
        return len(positions)
    if not positions:
        _check_alerts([], {})
        return 0

    lookup, _, _, _ = manual_trades.price_lookup_for_legs(positions)
    _check_alerts(positions, lookup)
    unpriced = [f"{p['side']} {p['strike']}{p['type']}" for p in positions if lookup.get(p["symbol"]) is None]
    with _state_lock:
        _state["unpriced_legs"] = unpriced  # these legs are NOT protected this pass -- shown in the UI
    for leg in positions:
        ltp = lookup.get(leg["symbol"])
        if ltp is None:
            continue
        record, msg = check_and_close(leg, ltp, now)
        if record is not None:
            _notify(msg, record)
    return len(manual_trades.load_positions())


def _check_alerts(positions, lookup):
    """User alerts (alerts.py): NIFTY level, a contract's LTP, or total
    live P&L of the open legs -- one quotes call for whatever they need."""
    armed = [a for a in alerts.load() if not a.get("triggered_at")]
    if not armed:
        return
    syms = {a["symbol"] for a in armed if a.get("symbol")} | {chain_mod.INDEX_SYMBOL}
    quotes = chain_mod.get_quotes(sorted(syms))
    ltps = {s: q["ltp"] for s, q in quotes.items()}
    ltps.update({k: v for k, v in lookup.items() if v is not None})
    live = [manual_trades._pnl(p, ltps[p["symbol"]]) for p in positions if ltps.get(p["symbol"]) is not None]
    open_pnl = round(sum(live), 2) if positions and len(live) == len(positions) else None
    alerts.check(ltps.get(chain_mod.INDEX_SYMBOL), ltps, open_pnl, lambda msg: (print(msg), tg.send(msg)))


def _beat(n_open=None, error=None):
    with _state_lock:
        _state["last_beat"] = time.time()
        if n_open is not None:
            _state["open_legs"] = n_open
        if error is None:
            _state["consecutive_errors"] = 0
        else:
            _state["last_error"] = error
            _state["last_error_at"] = time.time()
            _state["consecutive_errors"] += 1


def run_loop(stop_event):
    with _state_lock:
        _state["started_at"] = time.time()
    try:
        n = manual_trades.migrate_leg_expiries()
        if n:
            print(f"[risk monitor] Filled in the real expiry on {n} open leg(s).")
    except Exception as e:
        print(f"[risk monitor] Expiry migration failed (will retry via the positions page): {e}")

    alerted = False
    while not stop_event.is_set():
        try:
            n_open = run_one_pass()
            _beat(n_open)
            alerted = False
        except Exception as e:
            n_open = 1  # stay on the fast interval if we couldn't tell
            _beat(error=f"{type(e).__name__}: {e}")
            print(f"[risk monitor] pass failed: {e}")
            with _state_lock:
                failing = _state["consecutive_errors"] >= ERROR_ALERT_AFTER
            if failing and not alerted:
                tg.send(f"⚠️ [risk monitor] {ERROR_ALERT_AFTER} consecutive checks failed -- "
                        f"SL/Target may NOT be enforced. Last error: {e}")
                alerted = True
        fast = n_open and is_market_hours()
        stop_event.wait(POLL_INTERVAL_SEC if fast else IDLE_POLL_INTERVAL_SEC)


def _guarded_loop(stop_event):
    try:
        run_loop(stop_event)
    except Exception:
        traceback.print_exc()
        tg.send("🛑 [risk monitor] crashed -- SL/Target/square-off are NOT being enforced. Restart the app.")


def start_background_monitor():
    """Idempotent -- safe to call once at app startup."""
    global _thread, _stop_event
    if _thread is not None and _thread.is_alive():
        return _thread
    _stop_event = threading.Event()
    _thread = threading.Thread(target=_guarded_loop, args=(_stop_event,), daemon=True, name="risk-monitor")
    _thread.start()
    return _thread


def status():
    """Heartbeat for the UI. alive = thread running AND a pass finished in
    the last HEARTBEAT_STALE_SEC (a hung Fyers call shows as not alive)."""
    with _state_lock:
        s = dict(_state)
    running = _thread is not None and _thread.is_alive()
    beat_age = (time.time() - s["last_beat"]) if s["last_beat"] else None
    s["running"] = running
    s["beat_age_sec"] = round(beat_age, 1) if beat_age is not None else None
    s["alive"] = bool(running and beat_age is not None and beat_age <= HEARTBEAT_STALE_SEC
                      and s["consecutive_errors"] < ERROR_ALERT_AFTER)
    return s


def main():
    import broker
    print(f"Logging in to {broker.name()} (automated TOTP flow)...")
    broker.login()
    print(f"{broker.name()} login OK. Risk monitor running -- SL/Target/15:25 square-off/expiry settlement. Ctrl+C to stop.")
    tg.send("👁 Risk monitor started (standalone).")
    stop = threading.Event()
    try:
        run_loop(stop)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
