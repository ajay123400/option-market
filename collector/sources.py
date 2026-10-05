"""Fyers-facing adapters: GET-only REST client, read-only token provider (NEVER logs in), websocket feed. Everything injectable for offline tests."""
import logging
import re
import threading
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Callable, Dict, List, Optional, Tuple

from .config import WS_SYMBOL_LIMIT

log = logging.getLogger("collector")


class AuthImportError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ token (read-only)
class TokenProvider:
    """Reads today's token from the app's cache through fyers_auth._load_cached_token(), exactly as fyers_option_ws.py does. It never logs in and never touches the app's token-refresh helpers."""

    def __init__(self, module=None):
        self._m = module

    def _mod(self):
        if self._m is None:
            try:
                import fyers_auth                      # imports the app's .env; run from the repo root with the app's Python environment
            except (KeyError, ImportError) as e:
                raise AuthImportError(f"cannot import fyers_auth ({type(e).__name__}: {e}); run from the repo root with the app's .env present") from e
            self._m = fyers_auth
        return self._m

    def current(self) -> Optional[Tuple[str, str]]:
        m = self._mod()
        tok = m._load_cached_token()
        return (m.APP_ID, tok) if tok else None

    def wait_for_token(self, now_fn: Callable[[], datetime], sleep: Callable[[float], None], deadline: datetime, retry_s: float) -> Optional[Tuple[str, str]]:
        """Retry every retry_s until `deadline`; at least one attempt even if the deadline has already passed. None = no token (caller exits with a clear log line)."""
        attempt = 0
        while True:
            attempt += 1
            t = self.current()
            if t:
                if attempt > 1:
                    log.info("cached Fyers token found after %d attempts", attempt)
                return t
            remaining = (deadline - now_fn()).total_seconds()
            if remaining <= 0:
                return None
            log.warning("no valid cached Fyers token yet (the app's daily login has not happened); retrying in %.0f s (until %s)", min(retry_s, remaining), deadline.strftime("%H:%M"))
            sleep(min(retry_s, remaining))


# ------------------------------------------------------------------------------------------------ REST (GET only)
class RestResult:
    def __init__(self, name, status=None, body=None, latency_ms=None, error=None, server_date=None, rx_wall=None):
        self.name, self.status, self.body, self.latency_ms, self.error, self.server_date, self.rx_wall = name, status, body, latency_ms, error, server_date, rx_wall
        msg = str((body or {}).get("message", "")) if isinstance(body, dict) else ""
        self.message = msg
        self.rate_limited = status == 429 or (isinstance(body, dict) and (body.get("code") == -429 or "rate limit" in msg.lower()))
        self.auth_failed = status in (401, 403) or (isinstance(body, dict) and body.get("s") == "error" and bool(re.search(r"token|authenticate|expired|unauthor", msg, re.I)))
        self.ok = status == 200 and isinstance(body, dict) and body.get("s") == "ok" and not self.rate_limited

    def skew_s(self) -> Optional[float]:
        """local receive time minus the server Date header (header has 1 s resolution, so +/- 0.5 s)."""
        if not self.server_date or self.rx_wall is None:
            return None
        try:
            return self.rx_wall - parsedate_to_datetime(self.server_date).timestamp()
        except Exception:
            return None


class RestClient:
    """The ONLY REST entry point. HTTP GET only; spacing >= min_gap_s between calls; the Authorization header is built per call and never stored or logged."""

    def __init__(self, auth_fn: Callable[[], Optional[str]], session=None, sleep=time.sleep, clock=time.monotonic, wall=time.time, min_gap_s: float = 2.0, timeout: float = 15.0):
        self._auth, self._s, self._sleep, self._clock, self._wall, self._gap, self._timeout, self._last = auth_fn, session, sleep, clock, wall, min_gap_s, timeout, None
        self.calls = 0

    def get(self, name: str, url: str, params: Optional[dict] = None) -> RestResult:
        if self._s is None:
            import requests
            self._s = requests
        if self._last is not None:
            wait = self._gap - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        auth = self._auth()
        t0 = self._clock()
        self.calls += 1
        try:
            r = self._s.get(url, headers={"Authorization": auth} if auth else {}, params=params, timeout=self._timeout)
        except Exception as e:
            self._last = self._clock()
            return RestResult(name, error=f"{type(e).__name__}: {str(e)[:200]}", latency_ms=round((self._last - t0) * 1000, 1))
        self._last = self._clock()
        try:
            body = r.json()
        except Exception:
            body = None
        return RestResult(name, r.status_code, body, round((self._last - t0) * 1000, 1), None, r.headers.get("Date"), self._wall())


# ------------------------------------------------------------------------------------------------ websocket
class WsFeed:
    """Full-mode (litemode=False) Fyers data websocket; keeps the latest tick of every symbol with OUR receive time. One connection; the library enforces 5000 symbols per connection."""

    def __init__(self, auth_fn: Callable[[], Optional[str]], factory=None, wall=time.time, spawn=None):
        self._auth, self._factory, self._wall, self._spawn = auth_fn, factory, wall, spawn
        self._lock = threading.Lock()
        self._ticks: Dict[str, Tuple[float, dict]] = {}
        self.subscribed: set = set()
        self.connected = False
        self.ticks_total = 0
        self.events: List[dict] = []
        self.invalid_symbols: set = set()
        self._sock = None

    def _ev(self, kind, detail=""):
        with self._lock:
            self.events.append(dict(t=self._wall(), kind=kind, detail=str(detail)[:300]))
            del self.events[:-200]

    # callbacks (library threads)
    def _on_message(self, msg):
        rx = self._wall()
        if isinstance(msg, dict) and msg.get("symbol"):
            with self._lock:
                self._ticks[msg["symbol"]] = (rx, msg)
                self.ticks_total += 1

    def _on_open(self):
        self.connected = True
        self._ev("connected")
        syms = sorted(self.subscribed)
        for i in range(0, len(syms), 500):
            self._sock.subscribe(symbols=syms[i:i + 500], data_type="SymbolUpdate")
        self._ev("subscribed", len(syms))

    def _on_close(self, msg):
        self.connected = False
        self._ev("closed", msg)

    def _on_error(self, msg):
        inv = msg.get("invalid_symbols") if isinstance(msg, dict) else None
        if inv:
            self.invalid_symbols.update(inv)
            self.subscribed.difference_update(inv)
            self._ev("invalid_symbols", inv)
            return
        self._ev("error", msg)

    def start(self, symbols):
        self.subscribed = set(symbols)
        if len(self.subscribed) >= WS_SYMBOL_LIMIT:
            raise ValueError(f"{len(self.subscribed)} symbols exceed the websocket limit of {WS_SYMBOL_LIMIT}")
        factory = self._factory
        if factory is None:
            from fyers_apiv3.FyersWebsocket import data_ws
            factory = data_ws.FyersDataSocket
        self._sock = factory(access_token=self._auth(), log_path="", litemode=False, write_to_file=False, reconnect=True,
                             on_connect=self._on_open, on_close=self._on_close, on_error=self._on_error, on_message=self._on_message)
        if self._spawn is not None:
            self._spawn(self._sock.connect)                      # tests: run synchronously
        else:
            threading.Thread(target=self._sock.connect, daemon=True, name="fyers-ws").start()

    def add_symbols(self, symbols) -> List[str]:
        """Subscribe symbols not yet subscribed (also while connected). Returns the symbols newly requested. Errors are logged and recorded; the caller falls back to REST for those rows."""
        new = sorted(set(symbols) - self.subscribed - self.invalid_symbols)
        if not new:
            return []
        if len(self.subscribed) + len(new) >= WS_SYMBOL_LIMIT:
            self._ev("limit", f"{len(new)} new symbols would exceed {WS_SYMBOL_LIMIT}")
            return []
        self.subscribed.update(new)
        if self.connected:
            try:
                for i in range(0, len(new), 500):
                    self._sock.subscribe(symbols=new[i:i + 500], data_type="SymbolUpdate")
                self._ev("subscribed_more", len(new))
            except Exception as e:
                self._ev("subscribe_failed", e)
                log.warning("websocket subscribe of %d new symbols failed: %s", len(new), e)
        return new

    def snapshot(self) -> Dict[str, Tuple[float, dict]]:
        with self._lock:
            return dict(self._ticks)

    def stop(self):
        try:
            if self._sock is not None:
                self._sock.close_connection()
        except Exception as e:
            self._ev("close_failed", e)
        self.connected = False

    def restart(self):
        syms = sorted(self.subscribed)
        self.stop()
        self.start(syms)
