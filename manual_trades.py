"""manual_trades.py -- backend for the manual Strategy Builder tab: legs
picked straight off the option chain (buy/sell, any strike/type/qty),
persisted open positions, an intrinsic-value payoff-at-expiry curve plus a
best-effort "today" (T+0) curve built from our own Black-Scholes Greeks
(see greeks.py), and a permanent closed-trade history separate from the
automated strategy's own log.
"""
import calendar
import contextlib
import json
import os
import re
import time
from datetime import date, datetime, time as dtime, timedelta, timezone

import charges as charges_mod
import fyers_option_chain as chain_mod
import greeks as greeks_mod
import paths

LOT_SIZE = 65  # CURRENT NSE lot size for NIFTY -- used for NEW legs only


def lot_of(leg):
    """The lot size the leg was traded at (stored on it at entry). A future
    NSE lot-size revision must not reprice legs already open/closed; legs
    saved before this field existed fall back to the current size."""
    return leg.get("lot_size") or LOT_SIZE
POSITIONS_FILE = os.path.join(paths.BASE_DIR, "manual_positions.json")
HISTORY_FILE = os.path.join(paths.BASE_DIR, "results", "manual_trade_log.jsonl")
JOURNAL_NOTES_FILE = os.path.join(paths.BASE_DIR, "results", "journal_notes.json")
LOCK_FILE = os.path.join(paths.BASE_DIR, ".manual_positions.lock")


@contextlib.contextmanager
def _positions_lock(timeout=5.0):
    """Cross-process lock around every read-modify-write of
    manual_positions.json -- needed once app.py (browser-driven closes) and
    manual_monitor.py (background closes) can run at the same time. Without
    this, both processes closing the same leg in the same instant (most
    likely right at the 15:25 intraday square-off, when both notice it on
    their next poll within moments of each other) would each read the leg
    before either writes: the second write clobbers the first, so the leg
    reappears "open" and its close gets logged twice in history. A stale
    lock (left behind by a process that crashed mid-write) is stolen after
    `timeout` rather than deadlocking every future close forever."""
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.monotonic() > deadline:
                try:
                    os.remove(LOCK_FILE)
                except OSError:
                    pass
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass


def load_positions():
    if not os.path.exists(POSITIONS_FILE):
        return []
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def save_positions(positions):
    paths.atomic_write_json(POSITIONS_FILE, positions)


# ---- expiry resolution -----------------------------------------------------
# Legs used to be saved with expiry "" whenever they were added against the
# NEAREST expiry (the frontend's dropdown value for it). "" means "whatever
# is nearest right now", so once that expiry passed the leg silently
# re-pointed at the NEXT week's chain: no LTP, SL/Target unwatched, Close
# All skipping it, and time-to-expiry taken from the wrong series. Every leg
# now stores its own concrete expiry: `expiry` (Fyers' epoch-second
# timestamp string, the key get_chain() takes) plus `expiry_date`
# (YYYY-MM-DD, IST) for display and settlement.

IST = timezone(timedelta(hours=5, minutes=30))
EXPIRY_CLOSE = dtime(15, 40)  # matches the Fyers expiry timestamps (15:40 IST)
SETTLEMENT_DELAY_MIN = 30     # wait this long after the close before settling
_MONTH_CODES = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
_WEEKLY_MONTH = {**{str(i): i for i in range(1, 10)}, "O": 10, "N": 11, "D": 12}
_MONTHLY_RE = re.compile(r"^(?:NSE:)?NIFTY(\d{2})([A-Z]{3})\d+(?:CE|PE)$")
_WEEKLY_RE = re.compile(r"^(?:NSE:)?NIFTY(\d{2})([1-9OND])(\d{2})\d+(?:CE|PE)$")


def _parse_symbol_expiry(symbol):
    """(date, is_monthly) from an NSE option symbol, or (None, None).
    Weekly symbols carry the exact date (NIFTY26908... = 2026-09-08).
    Monthly ones only carry the month (NIFTY26SEP...) -- the date returned
    is that month's last Tuesday, which the caller replaces with the real
    listed monthly expiry whenever Fyers still lists it (a holiday can move
    it a day earlier)."""
    m = _WEEKLY_RE.match(symbol or "")
    if m:
        try:
            return date(2000 + int(m.group(1)), _WEEKLY_MONTH[m.group(2)], int(m.group(3))), False
        except ValueError:
            return None, None
    m = _MONTHLY_RE.match(symbol or "")
    if m and m.group(2) in _MONTH_CODES:
        year, month = 2000 + int(m.group(1)), _MONTH_CODES[m.group(2)]
        last = date(year, month, calendar.monthrange(year, month)[1])
        return last - timedelta(days=(last.weekday() - 1) % 7), True  # 1 = Tuesday
    return None, None


def resolve_leg_expiry(symbol, expiry=""):
    """(expiry_ts_str, expiry_date_str) for a leg. `expiry` is whatever the
    frontend sent ("" = nearest). Matched against Fyers' live expiry list so
    the stored timestamp is exactly the key get_chain() expects; falls back
    to a timestamp built from the symbol itself (an already-expired
    contract Fyers no longer lists) so settlement still knows the date."""
    sym_date, is_monthly = _parse_symbol_expiry(symbol)
    try:
        listed = chain_mod.list_expiries()
    except Exception:
        listed = []

    def _listed_date(e):
        d, m, y = e["date"].split("-")
        return date(int(y), int(m), int(d))

    if expiry:
        for e in listed:
            if str(e["expiry"]) == str(expiry):
                return str(e["expiry"]), _listed_date(e).isoformat()
        exp_date = datetime.fromtimestamp(float(expiry), IST).date()
        return str(expiry), exp_date.isoformat()

    if sym_date is not None:
        for e in listed:
            ld = _listed_date(e)
            if ld == sym_date or (is_monthly and e.get("expiry_flag") == "M"
                                  and (ld.year, ld.month) == (sym_date.year, sym_date.month)):
                return str(e["expiry"]), ld.isoformat()
        ts = datetime.combine(sym_date, EXPIRY_CLOSE, IST).timestamp()
        return str(int(ts)), sym_date.isoformat()

    if listed:  # unparseable symbol -- last resort: the nearest expiry, as before
        return str(listed[0]["expiry"]), _listed_date(listed[0]).isoformat()
    raise ValueError(f"Could not determine the expiry for {symbol}.")


def migrate_leg_expiries():
    """One-time (idempotent) fix-up of open legs saved before expiries were
    stored explicitly: fills in `expiry` / `expiry_date` from the symbol.
    Returns the number of legs updated."""
    positions = load_positions()
    if all(p.get("expiry") and p.get("expiry_date") for p in positions):
        return 0
    resolved = {}
    for p in positions:
        if not (p.get("expiry") and p.get("expiry_date")):
            resolved[p["id"]] = resolve_leg_expiry(p["symbol"], p.get("expiry") or "")
    with _positions_lock():
        positions = load_positions()
        n = 0
        for p in positions:
            if p["id"] in resolved and not (p.get("expiry") and p.get("expiry_date")):
                p["expiry"], p["expiry_date"] = resolved[p["id"]]
                n += 1
        if n:
            save_positions(positions)
    return n


def is_leg_expired(leg, now=None):
    exp_date = leg.get("expiry_date")
    if not exp_date:
        return False
    now = now or datetime.now(IST)
    close_dt = datetime.combine(date.fromisoformat(exp_date), EXPIRY_CLOSE, IST)
    return now >= close_dt


_last_settle_attempt = 0.0
_SETTLE_RETRY_SEC = 300


def _index_close_on(exp_date):
    """NIFTY 50's official close on `exp_date` (or the last trading day
    before it, for a holiday-shifted expiry) from Fyers' daily candles --
    NSE settles index options at the underlying's closing price on expiry
    day. None if Fyers doesn't return it (caller retries later)."""
    import historical_recorder
    candles = historical_recorder._fetch_history_candles(
        chain_mod.INDEX_SYMBOL, (exp_date - timedelta(days=6)).isoformat(), exp_date.isoformat(), resolution="D")
    best = None
    for row in candles:
        d = datetime.fromtimestamp(row[0], IST).date()
        if d <= exp_date and (best is None or d > best[0]):
            best = (d, float(row[4]))
    return best[1] if best else None


def settle_expired_legs(force=False):
    """Closes every open leg whose expiry has passed at its INTRINSIC value
    against the index's expiry-day close -- what the exchange actually
    settles at. Without this a POSITIONAL leg outlived its expiry forever
    with no price. Throttled (network call) unless force=True. Returns the
    list of settlement records."""
    global _last_settle_attempt
    now = datetime.now(IST)
    expired = [p for p in load_positions() if is_leg_expired(p, now)
               and now >= datetime.combine(date.fromisoformat(p["expiry_date"]), EXPIRY_CLOSE, IST)
               + timedelta(minutes=SETTLEMENT_DELAY_MIN)]
    if not expired:
        return []
    if not force and time.time() - _last_settle_attempt < _SETTLE_RETRY_SEC:
        return []
    _last_settle_attempt = time.time()
    records = []
    closes = {}
    for leg in expired:
        exp_date = date.fromisoformat(leg["expiry_date"])
        if exp_date not in closes:
            try:
                closes[exp_date] = _index_close_on(exp_date)
            except Exception:
                closes[exp_date] = None
        settle = closes[exp_date]
        if settle is None:
            continue
        K = leg["strike"]
        intrinsic = round(max(0.0, settle - K) if leg["type"] == "CE" else max(0.0, K - settle), 2)
        rec = close_leg(leg["id"], intrinsic, exit_reason="EXPIRED",
                        exit_time=datetime.combine(exp_date, EXPIRY_CLOSE).isoformat(timespec="seconds"),
                        extra={"settlement_price": settle})
        if rec:
            records.append(rec)
    return records


_last_stale_attempt = 0.0
SQUARE_OFF_CANDLE = dtime(15, 20)  # the 5-min candle that closes at the 15:25 square-off


def _square_off_price(symbol, day):
    """Close of `symbol`'s 15:20 candle (= price at 15:25) on `day`, from
    Fyers 5-min history; falls back to the last candle before it."""
    import historical_recorder
    rows = historical_recorder._fetch_history_candles(symbol, day.isoformat(), day.isoformat())
    best = None
    for r in rows:
        t = datetime.fromtimestamp(r[0], IST)
        if t.date() == day and t.time() <= SQUARE_OFF_CANDLE and (best is None or t > best[0]):
            best = (t, float(r[4]))
    return best[1] if best else None


def square_off_stale_intraday(force=False):
    """An INTRADAY leg still open on a LATER day than it was entered (the
    app / monitor wasn't running at 15:25 that day) is closed at what the
    15:25 square-off would have got -- that day's 15:20-candle close from
    exchange history -- and booked at that time, reason SQUARE_OFF_LATE.
    Before this it silently became an overnight position, carried until
    15:25 the NEXT day. Throttled; returns the close records."""
    global _last_stale_attempt
    today = datetime.now(IST).date()
    stale = [p for p in load_positions() if p.get("trade_type") == "INTRADAY"
             and (p.get("entry_time") or "")[:10] and date.fromisoformat(p["entry_time"][:10]) < today]
    if not stale or (not force and time.time() - _last_stale_attempt < _SETTLE_RETRY_SEC):
        return []
    _last_stale_attempt = time.time()
    records = []
    for leg in stale:
        day = date.fromisoformat(leg["entry_time"][:10])
        try:
            price = _square_off_price(leg["symbol"], day)
        except Exception:
            price = None
        if price is None:
            continue
        rec = close_leg(leg["id"], price, exit_reason="SQUARE_OFF_LATE",
                        exit_time=datetime.combine(day, dtime(15, 25)).isoformat(timespec="seconds"))
        if rec:
            records.append(rec)
    return records


def add_leg(symbol, strike, opt_type, side, qty, entry_price, expiry, sl=None, target=None,
            trade_type="INTRADAY", batch_id=None):
    """batch_id=None leaves the leg unsaved (a draft -- doesn't appear in
    the Journal until "Save Strategy" groups it, so a mis-clicked order
    never pollutes the journal as if it were a real strategy). Pass an
    EXISTING open batch_id to add this leg straight into that already-
    journaled strategy instead -- e.g. adding a hedge to a condor you
    already saved -- with no separate re-save step needed; the Journal
    entry just grows to include it immediately."""
    expiry, expiry_date = resolve_leg_expiry(symbol, expiry)
    with _positions_lock():
        positions = load_positions()
        leg = {
            "id": str(int(time.time() * 1000)),
            "batch_id": batch_id,
            "symbol": symbol, "strike": strike, "type": opt_type, "side": side,
            "qty": qty, "entry_price": entry_price, "expiry": expiry, "expiry_date": expiry_date,
            "entry_time": datetime.now().isoformat(timespec="seconds"),
            "sl": sl, "target": target, "lot_size": LOT_SIZE,
            "trade_type": trade_type,  # INTRADAY -> auto-squares off at 15:25; POSITIONAL -> holds indefinitely
        }
        positions.append(leg)
        save_positions(positions)
        return leg


def _k(strike):
    """23100.0 -> '23100' for display (strikes are stored as floats)."""
    return str(int(strike)) if float(strike).is_integer() else str(strike)


def list_open_batches():
    """Every currently-open, already-saved strategy -- for the Add Leg
    form's "add to an existing strategy" choice, and the "Viewing" strategy
    switcher. Also flags whether any unsaved draft legs currently exist
    (batch_id None) -- they're excluded from `batches` since they aren't a
    strategy yet, but the switcher still needs to know a "draft" view is
    available to offer."""
    positions = load_positions()
    groups = {}
    has_draft = False
    for p in positions:
        bid = p.get("batch_id")
        if not bid:
            has_draft = True
            continue
        groups.setdefault(bid, []).append(p)
    batches = [
        {"batch_id": bid, "description": ", ".join(f"{l['side']} {_k(l['strike'])}{l['type']}" for l in legs),
         "name": next((l.get("batch_name") for l in legs if l.get("batch_name")), None),
         "legs": len(legs), "opened": min((l.get("entry_time") or "") for l in legs),
         "expiry_date": min((l.get("expiry_date") or "") for l in legs) or None}
        for bid, legs in groups.items()
    ]
    batches.sort(key=lambda b: b["opened"], reverse=True)  # newest strategy first
    return {"batches": batches, "has_draft": has_draft}


def rename_strategy(batch_id, name):
    """Sets (or clears) the display name of one saved strategy."""
    name = (name or "").strip()[:40] or None
    with _positions_lock():
        positions = load_positions()
        hit = False
        for p in positions:
            if p.get("batch_id") == batch_id:
                p["batch_name"] = name
                hit = True
        if not hit:
            raise ValueError("Strategy not found (already closed?)")
        save_positions(positions)


def save_strategy(leg_ids=None, name=None):
    """Groups legs into one Journal entry. With leg_ids given, groups
    exactly those (must all currently be open and unsaved); with none,
    groups every currently-open leg that isn't already saved to some
    strategy. Returns the new batch_id, or None if there was nothing
    eligible to save."""
    with _positions_lock():
        positions = load_positions()
        if leg_ids is not None:
            eligible = [p for p in positions if p["id"] in leg_ids and not p.get("batch_id")]
        else:
            eligible = [p for p in positions if not p.get("batch_id")]
        if not eligible:
            return None
        batch_id = f"batch_{int(time.time() * 1000)}"
        eligible_ids = {p["id"] for p in eligible}
        for p in positions:
            if p["id"] in eligible_ids:
                p["batch_id"] = batch_id
                p["batch_name"] = (name or "").strip()[:40] or None
        save_positions(positions)
        return batch_id


def set_trade_type(leg_id, trade_type):
    if trade_type not in ("INTRADAY", "POSITIONAL"):
        raise ValueError("trade_type must be INTRADAY or POSITIONAL")
    with _positions_lock():
        positions = load_positions()
        leg = None
        for p in positions:
            if p["id"] == leg_id:
                p["trade_type"] = trade_type
                leg = p
                break
        if leg is not None:
            save_positions(positions)
        return leg


def validate_risk_levels(leg, ref_price, sl=None, target=None):
    """Raises ValueError if an SL/target sits on the wrong side of the
    current price. The trigger check is "price has crossed the level", so a
    wrong-side level fires on the very next poll -- e.g. a SHORT at 100 with
    its SL typed as 90 instantly "stops out" -- and used to book the level
    itself as the exit price, i.e. a profit that never existed."""
    if ref_price is None:
        return
    short = leg["side"] == "SELL"
    if sl is not None:
        if short and sl <= ref_price:
            raise ValueError(f"SL for a SELL leg must be ABOVE the current price ({ref_price}); got {sl}.")
        if not short and sl >= ref_price:
            raise ValueError(f"SL for a BUY leg must be BELOW the current price ({ref_price}); got {sl}.")
    if target is not None:
        if short and target >= ref_price:
            raise ValueError(f"Target for a SELL leg must be BELOW the current price ({ref_price}); got {target}.")
        if not short and target <= ref_price:
            raise ValueError(f"Target for a BUY leg must be ABOVE the current price ({ref_price}); got {target}.")


def update_leg_risk(leg_id, sl=None, target=None, clear_sl=False, clear_target=False, ref_price=None):
    """Sets/updates (or clears) the SL/target on an already-open leg.
    The in-app risk monitor (manual_monitor.py) checks these against live
    LTP and closes the leg AT THAT LTP when crossed. ref_price: current
    LTP (falls back to entry price) for validate_risk_levels()."""
    with _positions_lock():
        positions = load_positions()
        leg = None
        for p in positions:
            if p["id"] == leg_id:
                validate_risk_levels(p, ref_price if ref_price is not None else p["entry_price"],
                                     sl=None if clear_sl else sl, target=None if clear_target else target)
                if clear_sl:
                    p["sl"] = None
                elif sl is not None:
                    p["sl"] = sl
                if clear_target:
                    p["target"] = None
                elif target is not None:
                    p["target"] = target
                leg = p
                break
        if leg is not None:
            save_positions(positions)
        return leg


def chain_price_map(expiry_ts, strikecount=20):
    """{fyers_symbol: ltp} across the whole fetched chain for one expiry, plus spot."""
    data = chain_mod.get_chain(strikecount=strikecount, expiry_timestamp=expiry_ts)
    lookup = {}
    for s in data["strikes"]:
        if s["ce"]:
            lookup[s["ce"]["symbol"]] = s["ce"]["ltp"]
        if s["pe"]:
            lookup[s["pe"]["symbol"]] = s["pe"]["ltp"]
    return lookup, data["spot"]


def fill_price(symbol, side):
    """Realistic paper fill: a BUY pays the ASK, a SELL receives the BID
    (entries used to fill at LTP, which flatters every trade by half the
    spread). Falls back to LTP when the book is empty. Returns
    (price, basis) with basis 'ask' / 'bid' / 'ltp', or (None, None)."""
    q = chain_mod.get_quotes([symbol]).get(symbol)
    if not q:
        return None, None
    book = q.get("ask") if side == "BUY" else q.get("bid")
    if book and book > 0:
        return float(book), ("ask" if side == "BUY" else "bid")
    return (float(q["ltp"]), "ltp") if q.get("ltp") else (None, None)


def price_lookup_for_legs(legs, strikecount=20):
    """Merges a price lookup across every DISTINCT expiry actually present
    among these legs, not just one global expiry -- a leg added against a
    non-default expiry (e.g. switched to 15-Sep on the Option Chain before
    quick-adding) otherwise never resolves a live LTP, because a single
    query-string expiry used for the whole positions list stays on
    whichever expiry happened to be selected elsewhere (usually the
    nearest/default one). Shared by app.py (browser-driven watching) and
    manual_monitor.py (background watching) so both resolve prices the
    same way.

    Also returns:
    - expiry_ts_map: {expiry_string_on_the_leg: resolved epoch-second
      timestamp}, resolving the "" (nearest) placeholder to an actual
      moment -- for compute_todays_curve()'s time-to-expiry math.
    - forward_map: {expiry_string_on_the_leg: put-call-parity-implied
      forward for that expiry} -- see fyers_option_chain.get_chain()'s use
      of greeks.synthetic_forward() for why raw spot alone isn't the right
      "S" to feed Black-Scholes (NIFTY options price off the futures, not
      the cash index). compute_todays_curve() needs this per-expiry, not
      just the option chain's own already-computed Greeks, because a
      manual leg's IV has to be solved fresh from ITS OWN current LTP.
    Both computed from data already fetched here -- no extra chain calls."""
    expiries = {leg.get("expiry") or "" for leg in legs}
    lookup = {}
    spot = None
    expiry_ts_map = {}
    forward_map = {}
    for exp in expiries:
        try:
            data = chain_mod.get_chain(strikecount=strikecount, expiry_timestamp=exp)
        except Exception:
            continue
        for s in data["strikes"]:
            if s["ce"]:
                lookup[s["ce"]["symbol"]] = s["ce"]["ltp"]
            if s["pe"]:
                lookup[s["pe"]["symbol"]] = s["pe"]["ltp"]
        if spot is None:
            spot = data["spot"]
        expiry_ts_map[exp] = data.get("resolved_expiry_ts")
        if data["spot"]:
            near_atm = sorted(data["strikes"], key=lambda r: abs(r["strike"] - data["spot"]))[:6]
            quotes = [(r["strike"], r["ce"]["ltp"] if r["ce"] else None, r["pe"]["ltp"] if r["pe"] else None)
                      for r in near_atm]
            forward_map[exp] = greeks_mod.synthetic_forward(quotes) or data["spot"]
    # Any leg outside the chain window (far hedge, or after a big move)
    # gets priced directly by symbol instead of silently having no LTP.
    missing = [leg["symbol"] for leg in legs if lookup.get(leg["symbol"]) is None]
    if missing:
        try:
            for sym, q in chain_mod.get_quotes(missing).items():
                lookup[sym] = q["ltp"]
        except Exception:
            pass
    return lookup, spot, expiry_ts_map, forward_map


def greeks_lookup_for_legs(legs, strikecount=20):
    """{symbol: {iv,delta,gamma,theta,vega}} across every distinct expiry
    present among these legs -- mirrors price_lookup_for_legs's per-expiry
    fan-out, but pulls the chain's own already-computed Greeks
    (include_greeks=True) instead of just LTP, so the Strategy Builder's
    Greeks tab doesn't need a second forward/IV-solve pass of its own."""
    expiries = {leg.get("expiry") or "" for leg in legs}
    lookup = {}
    for exp in expiries:
        try:
            data = chain_mod.get_chain(strikecount=strikecount, expiry_timestamp=exp, include_greeks=True)
        except Exception:
            continue
        for s in data["strikes"]:
            for side in ("ce", "pe"):
                leg = s[side]
                if leg:
                    lookup[leg["symbol"]] = {"iv": leg.get("iv"), "delta": leg.get("delta"),
                                              "gamma": leg.get("gamma"), "theta": leg.get("theta"),
                                              "vega": leg.get("vega")}
    return lookup


def add_basket(legs_spec, trade_type="INTRADAY", name=None):
    """Adds every leg in `legs_spec` (each a dict with symbol/strike/type/
    side/qty/entry_price/expiry) under ONE new batch_id in a single write
    -- a multi-leg strategy placed via the Basket Order tool appears as
    one Journal entry immediately, with no separate Save Strategy step.
    Millisecond-based ids (add_leg()'s scheme) would collide when several
    legs are created in the same loop iteration, so each gets an index
    suffix here to stay unique even when added in the same millisecond."""
    resolved = [resolve_leg_expiry(spec["symbol"], spec.get("expiry", "")) for spec in legs_spec]
    with _positions_lock():
        positions = load_positions()
        batch_id = f"batch_{int(time.time() * 1000)}"
        now_iso = datetime.now().isoformat(timespec="seconds")
        new_legs = []
        for i, spec in enumerate(legs_spec):
            leg = {
                "id": f"{int(time.time() * 1000)}_{i}",
                "batch_id": batch_id,
                "symbol": spec["symbol"], "strike": spec["strike"], "type": spec["type"], "side": spec["side"],
                "qty": int(spec["qty"]), "entry_price": float(spec["entry_price"]),
                "expiry": resolved[i][0], "expiry_date": resolved[i][1],
                "entry_time": now_iso, "sl": None, "target": None, "trade_type": trade_type, "lot_size": LOT_SIZE,
                "batch_name": (name or "").strip()[:40] or None,
            }
            positions.append(leg)
            new_legs.append(leg)
        save_positions(positions)
        return batch_id, new_legs


def _pnl(leg, price):
    diff = (leg["entry_price"] - price) if leg["side"] == "SELL" else (price - leg["entry_price"])
    return diff * lot_of(leg) * leg["qty"]


def _with_charges(record):
    """Adds estimated round-trip `charges` and `net_pnl` to a close record."""
    c = charges_mod.round_trip(record["side"], record["entry_price"], record["exit_price"],
                               lot_of(record) * record["qty"], record.get("exit_reason"))
    record["charges"] = c
    record["net_pnl"] = round(record["pnl"] - c, 2)
    return record


def est_open_charges(leg, ltp):
    """Charges a still-open leg would have paid in total if closed at ltp."""
    return charges_mod.round_trip(leg["side"], leg["entry_price"], ltp, lot_of(leg) * leg["qty"])


def _log_history(record):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def close_leg(leg_id, exit_price, exit_reason="MANUAL", exit_time=None, extra=None):
    """exit_reason: MANUAL / SL / TARGET / SQUARE_OFF / EXPIRED / CLOSE_ALL --
    recorded on the history row so the Journal can tell a stop-out from a
    discretionary exit. exit_time overrides "now" (used by expiry
    settlement, which books at the expiry close, not whenever it ran)."""
    with _positions_lock():
        positions = load_positions()
        remaining, closed = [], None
        for p in positions:
            if p["id"] == leg_id:
                closed = p
            else:
                remaining.append(p)
        if closed is None:
            return None
        record = {**closed, "exit_price": exit_price,
                  "exit_time": exit_time or datetime.now().isoformat(timespec="seconds"),
                  "exit_reason": exit_reason, **(extra or {}),
                  "pnl": round(_pnl(closed, exit_price), 2)}
        _with_charges(record)
        _log_history(record)
        save_positions(remaining)
        return record


def close_all(price_lookup, batch_filter=None):
    """price_lookup: {symbol: current_ltp}. Closes every open leg at its
    current LTP; legs whose symbol isn't in price_lookup are left open and
    returned in `skipped` so the UI can say so (it used to report success).
    Returns (records, skipped).

    batch_filter: None (default) closes every open leg, matching the old
    behaviour. Pass a batch_id to close only that strategy's legs, or the
    literal "__draft__" to close only unsaved-draft legs -- everything
    else stays open untouched. Lets "Close All" mean "close what I'm
    currently viewing" once multiple strategies can be open at once,
    rather than always nuking every open position on the account."""
    with _positions_lock():
        positions = load_positions()
        remaining, records, skipped = [], [], []
        for p in positions:
            if batch_filter is not None:
                leg_batch = p.get("batch_id")
                matches = (leg_batch is None and batch_filter == "__draft__") or (leg_batch == batch_filter)
                if not matches:
                    remaining.append(p)
                    continue
            price = price_lookup.get(p["symbol"])
            if price is None:
                remaining.append(p)
                skipped.append({"id": p["id"], "leg": f"{p['side']} {p['strike']}{p['type']}"})
                continue
            record = {**p, "exit_price": price,
                       "exit_time": datetime.now().isoformat(timespec="seconds"),
                       "exit_reason": "CLOSE_ALL",
                       "pnl": round(_pnl(p, price), 2)}
            _with_charges(record)
            _log_history(record)
            records.append(record)
        save_positions(remaining)
    return records, skipped


DELETED_FILE = os.path.join(paths.BASE_DIR, "results", "deleted_trades.jsonl")


def delete_strategy(batch_id):
    """Removes a strategy entered by mistake -- its open legs AND any legs
    already booked to history -- WITHOUT booking any P&L, so it disappears
    from positions and the Journal. batch_id "__draft__" = the unsaved draft
    legs. Every removed record is appended to results/deleted_trades.jsonl
    first (with the time of deletion), so a mistaken delete can be undone by
    hand. Returns (open_legs_removed, history_rows_removed)."""
    if not batch_id:
        raise ValueError("No strategy given.")
    match = (lambda r: not r.get("batch_id")) if batch_id == "__draft__" else (lambda r: r.get("batch_id") == batch_id)
    stamp = datetime.now().isoformat(timespec="seconds")
    with _positions_lock():
        positions = load_positions()
        gone = [p for p in positions if match(p)]
        keep = [p for p in positions if not match(p)]
        hist_rows, hist_gone = [], []
        if batch_id != "__draft__" and os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        hist_rows.append(line)  # keep unreadable lines exactly as they were
                        continue
                    (hist_gone if match(r) else hist_rows).append(r if match(r) else line)
        if not gone and not hist_gone:
            raise ValueError("Nothing to delete (strategy not found).")
        os.makedirs(os.path.dirname(DELETED_FILE), exist_ok=True)
        with open(DELETED_FILE, "a") as f:
            for r in gone:
                f.write(json.dumps({"deleted_at": stamp, "source": "open", **r}) + "\n")
            for r in hist_gone:
                f.write(json.dumps({"deleted_at": stamp, "source": "history", **r}) + "\n")
        if hist_gone:
            tmp = HISTORY_FILE + ".tmp"
            with open(tmp, "w") as f:
                for r in hist_rows:
                    f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
            os.replace(tmp, HISTORY_FILE)
        if gone:
            save_positions(keep)
    if batch_id != "__draft__":
        notes = _load_notes()
        if batch_id in notes:
            notes.pop(batch_id)
            paths.atomic_write_json(JOURNAL_NOTES_FILE, notes)
    return len(gone), len(hist_gone)


def preview_positions(legs_spec):
    """Hypothetical open legs for a payoff PREVIEW -- same shape as saved
    positions, but never written anywhere (no position file, no Journal)."""
    now_iso = datetime.now().isoformat(timespec="seconds")
    out = []
    for i, spec in enumerate(legs_spec):
        exp, exp_date = resolve_leg_expiry(spec["symbol"], spec.get("expiry", ""))
        out.append({"id": f"preview_{i}", "batch_id": "__preview__", "symbol": spec["symbol"],
                    "strike": float(spec["strike"]), "type": spec["type"], "side": spec["side"],
                    "qty": int(spec.get("qty") or 1), "entry_price": float(spec["price"]),
                    "expiry": exp, "expiry_date": exp_date, "entry_time": now_iso,
                    "sl": None, "target": None, "trade_type": "INTRADAY", "lot_size": LOT_SIZE})
    return out


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    rows = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # one half-written line (crash mid-append) must not break history/journal
    rows.sort(key=lambda r: r.get("exit_time", ""), reverse=True)
    return rows


def positions_with_live(positions, price_lookup):
    """positions annotated with live LTP + live per-leg P&L. Takes the
    positions list explicitly (rather than re-reading the file itself) so
    callers can run this over an already-fetched, possibly-filtered list
    without a second load_positions() disk read."""
    out = []
    for p in positions:
        ltp = price_lookup.get(p["symbol"])
        row = dict(p)
        row["ltp"] = ltp
        row["live_pnl"] = round(_pnl(p, ltp), 2) if ltp is not None else None
        row["est_charges"] = est_open_charges(p, ltp) if ltp is not None else None
        row["expired"] = is_leg_expired(p)
        out.append(row)
    return out


def payoff_curve(positions, spot, num_points=41, price_step=50):
    """Standard at-expiry intrinsic-value payoff line -- the guaranteed
    end-state regardless of any volatility assumption. See
    compute_todays_curve() below for the (approximate, Black-Scholes-based)
    "if the underlying moved there right now" line.
    Range is spot +/- ~1000 points (roughly the same ATM window the option
    chain itself shows) -- wide enough for typical breakevens without
    stretching the chart so far that the near-the-money shape (the part
    that actually matters) gets squeezed into a sliver against the tails."""
    if not positions:
        return []
    half = num_points // 2
    prices = [spot + (i - half) * price_step for i in range(num_points)]
    # The payoff only bends AT strikes -- a grid anchored on an arbitrary
    # spot (23512.4, say) never lands exactly on one, which clipped the
    # peak of e.g. a short straddle. Every strike inside the window is
    # added as its own point so the line's corners are drawn exactly.
    lo, hi = prices[0], prices[-1]
    prices = sorted(set(prices) | {float(l["strike"]) for l in positions if lo < l["strike"] < hi})
    return [{"price": S, "pnl": round(_expiry_pnl(positions, S), 2)} for S in prices]


def _expiry_pnl(positions, S):
    """At-expiry P&L (Rs) of `positions` if the index settles at S."""
    total = 0.0
    for leg in positions:
        K = leg["strike"]
        intrinsic = max(0.0, S - K) if leg["type"] == "CE" else max(0.0, K - S)
        if leg["side"] == "SELL":
            total += (leg["entry_price"] - intrinsic) * lot_of(leg) * leg["qty"]
        else:
            total += (intrinsic - leg["entry_price"]) * lot_of(leg) * leg["qty"]
    return total


def expiry_risk(positions, offset=0.0):
    """EXACT max profit / max loss / breakevens of the at-expiry payoff
    over the whole price range 0..infinity -- not just the charted window.

    The payoff is piecewise-linear with corners only at strikes, so its
    extremes can only sit at S=0, at a strike, or at S->infinity. Beyond
    the highest strike the slope is (long calls - short calls) * lot size:
    positive -> profit is unlimited, negative -> loss is unlimited (e.g. a
    naked short call, which the old +/-1000pt sampled window reported as a
    finite, badly understated "max loss"). `offset` is P&L already realized
    on booked legs of the same strategy, added to every point.

    Returns max_profit / max_loss as a number, or None with the matching
    *_unlimited flag set True."""
    if not positions:
        return None
    strikes = sorted({float(l["strike"]) for l in positions})
    points = [0.0] + strikes
    vals = [_expiry_pnl(positions, S) + offset for S in points]
    slope_up = sum((1 if l["side"] == "BUY" else -1) * l["qty"] * lot_of(l)
                   for l in positions if l["type"] == "CE")

    max_profit_unlimited = slope_up > 0
    max_loss_unlimited = slope_up < 0
    max_profit = None if max_profit_unlimited else max(vals)
    max_loss = None if max_loss_unlimited else min(vals)

    breakevens = []
    for i in range(1, len(points)):
        a, b = vals[i - 1], vals[i]
        if (a < 0) != (b < 0) and b != a:
            breakevens.append(round(points[i - 1] + (-a / (b - a)) * (points[i] - points[i - 1]), 2))
    # Beyond the last strike: the line keeps going with slope_up.
    last_val = vals[-1]
    if slope_up and (last_val < 0) != (slope_up < 0) and last_val != 0:
        breakevens.append(round(points[-1] - last_val / slope_up, 2))
    return {
        "max_profit": None if max_profit is None else round(max_profit, 2),
        "max_loss": None if max_loss is None else round(max_loss, 2),
        "max_profit_unlimited": max_profit_unlimited,
        "max_loss_unlimited": max_loss_unlimited,
        "breakevens": breakevens,
    }


def compute_todays_curve(positions, spot, price_lookup, expiry_ts_map, forward_map, prices):
    """"Today" (T+0) payoff curve: at each hypothetical underlying price in
    `prices`, reprices every leg via Black-Scholes using ITS OWN implied
    vol -- solved from that leg's current live LTP and held FIXED across
    the curve -- and its own real time-to-expiry from right now. This
    shows sensitivity to price movement only (not to IV changing further,
    nor to time passing beyond this instant) -- a snapshot, not a forecast.

    Returns None (caller falls back to payoff_curve()'s at-expiry-only
    line) if ANY leg's IV can't be solved right now (missing/stale LTP, or
    its expiry has effectively passed) -- silently mixing a real T+0 leg
    with an intrinsic-only one would misrepresent the combined curve's
    shape, so this only ever returns a curve where every leg is genuinely
    priced the same way.

    Uses each leg's own expiry's put-call-parity forward (forward_map),
    not raw spot, both to solve that leg's IV and to reprice across the
    curve -- NIFTY options price off the futures, which commonly sit
    40-80pt above the cash index (see greeks.synthetic_forward's
    docstring), so using raw spot systematically understates ITM puts /
    overstates ITM calls relative to what the market is actually pricing,
    and can make IV unsolvable outright for a leg that's gone ITM (a real
    failure hit live: a ~23900 strike PE's LTP sat below its RAW-spot
    intrinsic value, which made bisection's bounds check correctly refuse
    to invert it -- switching to the forward fixed this). `prices` stays
    anchored on raw spot (matching payoff_curve()'s x-axis, so both lines
    share one scale) -- each hypothetical price is shifted by that
    expiry's basis (forward - spot) before repricing, keeping the same
    spot-to-forward gap constant across the curve rather than mixing
    forward-based IV with spot-based repricing."""
    if not positions or not prices or not spot:
        return None
    now = time.time()
    leg_calcs = []
    for leg in positions:
        ltp = price_lookup.get(leg["symbol"])
        if ltp is None:
            return None
        expiry_key = leg.get("expiry") or ""
        expiry_ts = expiry_ts_map.get(expiry_key)
        forward = forward_map.get(expiry_key)
        if not expiry_ts or not forward:
            return None
        T = (expiry_ts - now) / (365 * 86400)
        if T <= 0:
            return None
        is_call = leg["type"] == "CE"
        iv = greeks_mod.implied_vol_fwd(ltp, forward, leg["strike"], T, greeks_mod.RISK_FREE_RATE, is_call)
        if iv is None:
            return None
        basis = forward - spot
        leg_calcs.append((leg, iv, T, is_call, basis))

    curve = []
    for S in prices:
        total = 0.0
        for leg, iv, T, is_call, basis in leg_calcs:
            price_now = greeks_mod.b76_price(S + basis, leg["strike"], T, greeks_mod.RISK_FREE_RATE, iv, is_call)
            if leg["side"] == "SELL":
                total += (leg["entry_price"] - price_now) * lot_of(leg) * leg["qty"]
            else:
                total += (price_now - leg["entry_price"]) * lot_of(leg) * leg["qty"]
        curve.append({"price": S, "pnl": round(total, 2)})
    return curve


def scenario_grid(positions, spot, price_lookup, expiry_ts_map, forward_map,
                  days_forward=0, moves_pct=(-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3), iv_shifts=(-3, 0, 3),
                  realized_pnl=0.0):
    """What-if P&L table: rows = NIFTY move (% from spot), columns = IV
    change (vol points, applied to every leg), all `days_forward` calendar
    days from now. Each leg is repriced with Black-76 from its OWN implied
    vol (solved from its live LTP) and its own expiry's forward; a leg
    whose expiry falls within `days_forward` is valued at intrinsic.
    Returns None if any leg can't be priced (same all-or-nothing rule as
    the T+0 curve)."""
    if not positions or not spot:
        return None
    now = time.time()
    legs = []
    for leg in positions:
        ltp = price_lookup.get(leg["symbol"])
        key = leg.get("expiry") or ""
        exp_ts, fwd = expiry_ts_map.get(key), forward_map.get(key)
        if ltp is None or not exp_ts or not fwd:
            return None
        T = (exp_ts - now) / (365 * 86400)
        if T <= 0:
            return None
        iv = greeks_mod.implied_vol_fwd(ltp, fwd, leg["strike"], T, greeks_mod.RISK_FREE_RATE, leg["type"] == "CE")
        if iv is None:
            return None
        legs.append((leg, iv, T - days_forward / 365, fwd - spot))
    rows = []
    for m in moves_pct:
        S = spot * (1 + m / 100)
        cells = []
        for shift in iv_shifts:
            total = realized_pnl or 0.0
            for leg, iv, T2, basis in legs:
                is_call = leg["type"] == "CE"
                if T2 <= 0:
                    px = max(0.0, S - leg["strike"]) if is_call else max(0.0, leg["strike"] - S)
                else:
                    px = greeks_mod.b76_price(S + basis, leg["strike"], T2, greeks_mod.RISK_FREE_RATE,
                                              max(0.01, iv + shift / 100), is_call)
                sign = -1 if leg["side"] == "SELL" else 1
                total += sign * (px - leg["entry_price"]) * lot_of(leg) * leg["qty"]
            cells.append(round(total, 2))
        rows.append({"move_pct": m, "spot": round(S, 1), "pnl": cells})
    return {"iv_shifts": list(iv_shifts), "days_forward": days_forward, "rows": rows}


def _load_notes():
    if not os.path.exists(JOURNAL_NOTES_FILE):
        return {}
    with open(JOURNAL_NOTES_FILE) as f:
        return json.load(f)


def save_journal_note(batch_id, note):
    os.makedirs(os.path.dirname(JOURNAL_NOTES_FILE), exist_ok=True)
    notes = _load_notes()
    notes[batch_id] = note
    paths.atomic_write_json(JOURNAL_NOTES_FILE, notes)


def get_journal(price_lookup=None):
    """One entry per SAVED STRATEGY (batch_id), not per leg -- groups every
    closed leg (from history) and every still-open leg (from current
    positions) that share a batch_id into a single journal row: combined
    legs, total P&L (realized for closed legs, live for any still open,
    using price_lookup if given), first entry time, last exit time (None
    while any leg in the batch is still open), and status OPEN/CLOSED.

    A leg with no batch_id was never explicitly saved via "Save Strategy"
    (or predates that feature) -- it's excluded here entirely rather than
    shown as its own one-leg entry, so a stray/mis-clicked order that gets
    closed out doesn't clutter the journal as if it were a real strategy."""
    notes = _load_notes()
    batches = {}

    for rec in load_history():
        batch_id = rec.get("batch_id")
        if not batch_id:
            continue
        b = batches.setdefault(batch_id, {"legs": [], "open_count": 0})
        b["legs"].append({**rec, "status": "CLOSED"})

    for p in load_positions():
        batch_id = p.get("batch_id")
        if not batch_id:
            continue
        b = batches.setdefault(batch_id, {"legs": [], "open_count": 0})
        live_pnl = None
        if price_lookup is not None:
            ltp = price_lookup.get(p["symbol"])
            if ltp is not None:
                live_pnl = round(_pnl(p, ltp), 2)
        b["legs"].append({**p, "status": "OPEN", "pnl": live_pnl, "ltp": ltp if price_lookup is not None else None,
                          "charges": est_open_charges(p, ltp) if price_lookup is not None and ltp is not None else None})
        b["open_count"] += 1

    journal = []
    for batch_id, b in batches.items():
        legs = sorted(b["legs"], key=lambda x: x.get("entry_time", ""))
        total_pnl = sum(l["pnl"] for l in legs if l.get("pnl") is not None)
        any_pnl_missing = any(l.get("pnl") is None for l in legs)
        for l in legs:  # older history rows predate stored charges -- estimate them the same way
            if l.get("charges") is None and l["status"] == "CLOSED":
                l["charges"] = charges_mod.round_trip(l["side"], l["entry_price"], l["exit_price"],
                                                      lot_of(l) * l["qty"], l.get("exit_reason"))
        total_charges = round(sum(l.get("charges") or 0 for l in legs), 2)
        entry_time = min((l["entry_time"] for l in legs), default=None)
        exit_times = [l["exit_time"] for l in legs if l.get("exit_time")]
        exit_time = max(exit_times) if (b["open_count"] == 0 and exit_times) else None
        description = ", ".join(f"{l['side']} {_k(l['strike'])}{l['type']}" for l in legs)
        name = next((l.get("batch_name") for l in legs if l.get("batch_name")), None)
        if name:
            description = f"{name} — {description}"
        journal.append({
            "batch_id": batch_id,
            "status": "OPEN" if b["open_count"] > 0 else "CLOSED",
            "description": description,
            "legs": legs,
            "total_pnl": None if any_pnl_missing else round(total_pnl, 2),
            "total_charges": total_charges,
            "net_pnl": None if any_pnl_missing else round(total_pnl - total_charges, 2),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "note": notes.get(batch_id, ""),
        })
    journal.sort(key=lambda j: j["entry_time"] or "", reverse=True)
    return journal


def summary(positions, curve, realized_pnl=0.0):
    """Stats for the Strategy Builder / Simulator. Max profit/loss and
    breakevens come from expiry_risk() over the WHOLE price range, not from
    `curve` (which only spans the charted +/-1000pt window) -- `curve` is
    kept as a parameter only to preserve the "no chart -> no stats" rule.
    realized_pnl: P&L already booked on closed legs of this same strategy,
    folded into max profit/loss/breakevens the same way the chart folds it
    into the curve."""
    net_credit = sum(
        (leg["entry_price"] if leg["side"] == "SELL" else -leg["entry_price"]) * lot_of(leg) * leg["qty"]
        for leg in positions
    )
    risk = expiry_risk(positions, offset=realized_pnl or 0.0) if curve else None
    if not risk:
        return {"net_credit": round(net_credit, 2), "max_profit": None, "max_loss": None,
                "max_profit_unlimited": False, "max_loss_unlimited": False, "breakevens": []}
    return {"net_credit": round(net_credit, 2), **risk}
