"""fyers_option_ws.py -- live tick-by-tick feed for a NIFTY option chain,
adapted from StockDashboard's fyers_ws.py (same FyersDataSocket wrapper,
same on_open/on_error/on_close handling), but subscribing to one expiry's
full option chain (+ the NIFTY 50 index for spot) instead of the whole
equity universe.

Maintains an in-memory {fyers_symbol: tick_dict} cache plus a per-symbol
"first OI seen today" snapshot (for computing intraday OI change), which
Flask routes / the strategy engine read directly -- no per-request round
trip to Fyers.
"""
import logging
import threading
import time

from fyers_apiv3.FyersWebsocket import data_ws

import fyers_auth
import fyers_option_symbols

logger = logging.getLogger(__name__)

_price_cache = {}
_day_open_oi = {}          # {symbol: oi at first tick seen today}
_day_open_oi_date = None   # 'YYYY-MM-DD' the above snapshot belongs to
_cache_lock = threading.Lock()
_socket = None
_subscribed = set()
_connected = False
_last_tick_at = None

NIFTY_INDEX_SYMBOL = "NSE:NIFTY50-INDEX"


class FyersOptionWsError(RuntimeError):
    pass


def _on_message(msg):
    global _last_tick_at, _day_open_oi_date
    sym = msg.get("symbol")
    if not sym:
        return
    today = time.strftime("%Y-%m-%d")
    with _cache_lock:
        if _day_open_oi_date != today:
            _day_open_oi_date = today
            _day_open_oi.clear()
        _price_cache[sym] = msg
        oi = msg.get("oi")
        if oi is not None and sym not in _day_open_oi:
            _day_open_oi[sym] = oi
        _last_tick_at = time.monotonic()


def _on_error(msg):
    global _connected
    invalid = msg.get("invalid_symbols") if isinstance(msg, dict) else None
    if invalid:
        logger.warning(f"subscribe rejected for invalid symbols: {invalid}")
        _subscribed.difference_update(invalid)
        if _socket is not None and _subscribed:
            _socket.subscribe(symbols=list(_subscribed), data_type="SymbolUpdate")
        return
    _connected = False
    logger.error(f"option websocket error: {msg}")


def _on_close(msg):
    global _connected
    _connected = False
    logger.warning(f"option websocket closed: {msg}")


def _on_open():
    global _connected
    _connected = True
    logger.info(f"fyers_option_ws connected, subscribing {len(_subscribed)} symbols")
    if _subscribed:
        symbols = list(_subscribed)
        CHUNK = 500
        for i in range(0, len(symbols), CHUNK):
            _socket.subscribe(symbols=symbols[i:i + CHUNK], data_type="SymbolUpdate")


def resolve_chain_symbols(underlying="NIFTY", expiry=None, strike_range=None):
    """Returns (expiry_used, sorted_list_of_fyers_option_symbols)."""
    expiries = fyers_option_symbols.list_expiries(underlying)
    if not expiries:
        raise FyersOptionWsError("No expiries found from Fyers symbol master.")
    expiry = expiry or expiries[0]
    strikes = fyers_option_symbols.available_strikes(underlying, expiry)
    if strike_range is not None:
        # caller can narrow later once spot is known; default = full chain
        pass
    chain = fyers_option_symbols.chain_symbols(underlying, expiry, strikes)
    symbols = []
    for s, d in chain.items():
        if d["CE"]:
            symbols.append(d["CE"])
        if d["PE"]:
            symbols.append(d["PE"])
    return expiry, sorted(symbols)


TOKEN_RETRY_INTERVAL_SEC = 30
_start_thread = None


def start(underlying="NIFTY", expiry=None):
    """Opens the websocket connection in a background thread, subscribing
    to the full option chain for the given (or nearest) expiry plus the
    NIFTY index. Requires a same-day cached Fyers token (run
    `python fyers_auth.py` once) -- does not trigger an interactive login
    itself, so it never hangs the caller."""
    global _socket, _start_thread
    if _socket is not None or _start_thread is not None:
        return

    def _connect_when_ready():
        global _socket
        token = fyers_auth._load_cached_token()
        while not token:
            time.sleep(TOKEN_RETRY_INTERVAL_SEC)
            token = fyers_auth._load_cached_token()

        expiry_used, symbols = resolve_chain_symbols(underlying, expiry)
        _subscribed.update(symbols)
        _subscribed.add(NIFTY_INDEX_SYMBOL)
        logger.info(f"option chain expiry resolved to {expiry_used}, {len(symbols)} option symbols")

        _socket = data_ws.FyersDataSocket(
            access_token=fyers_auth.get_auth_header(),
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=_on_open,
            on_close=_on_close,
            on_error=_on_error,
            on_message=_on_message,
        )
        _socket.connect()

    if not fyers_auth._load_cached_token():
        logger.warning(
            f"No cached Fyers access_token yet -- will keep checking every "
            f"{TOKEN_RETRY_INTERVAL_SEC}s. Run `python fyers_auth.py` "
            f"whenever convenient; the feed connects automatically once a "
            f"token appears."
        )
    _start_thread = threading.Thread(target=_connect_when_ready, daemon=True)
    _start_thread.start()


def get_tick(fyers_symbol):
    with _cache_lock:
        return _price_cache.get(fyers_symbol)


def get_all_ticks():
    with _cache_lock:
        return dict(_price_cache)


def get_day_open_oi(fyers_symbol):
    with _cache_lock:
        return _day_open_oi.get(fyers_symbol)


def is_connected():
    return _connected


def seconds_since_last_tick():
    if _last_tick_at is None:
        return None
    return time.monotonic() - _last_tick_at


def cached_symbol_count():
    with _cache_lock:
        return len(_price_cache)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting option chain websocket (Ctrl+C to stop)...")
    start()
    try:
        while True:
            time.sleep(5)
            print(f"connected={is_connected()} symbols_cached={cached_symbol_count()} "
                  f"last_tick_age={seconds_since_last_tick()}")
    except KeyboardInterrupt:
        pass
