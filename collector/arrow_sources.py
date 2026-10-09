"""Arrow (iRage) adapters with the SAME interfaces as sources.py (TokenProvider / RestClient / WsFeed), so the recorder, store, quality flags
and analysis run unchanged on either broker. Selected when the app's .env has DATA_SOURCE=arrow (the app is moving off Fyers: from 9 Oct
2026 Fyers' Standard plan allows 5,000 data calls a day and 50 websocket symbols, below this recorder's needs).

  * symbols stay Fyers-style ("NSE:NIFTY26O1322650CE") in the database; arrow_chain.instruments() maps them to Arrow tokens.
  * chain "call" = /info/option-chain (strikes + openingOI = previous-day OI) + /info/quotes/full for those contracts and the future
    (<= 100 per call), returned in options-chain-v3 shape so universe.parse_chain() reads it. Arrow has no REST quote for the index,
    so spot / India VIX come from this recorder's own websocket (spot_fn / vix_fn), falling back to the last 1-min candle.
  * websocket = wss://ds.arrow.trade "full" mode, big-endian binary (arrow_chain.parse_tick), mapped to the Fyers field names the
    recorder reads (bid_price, ask_price, bid_size, ask_size, vol_traded_today, last_traded_qty, avg_trade_price, tot_buy_qty,
    tot_sell_qty, exch_feed_time, last_traded_time). Arrow allows 1,024 symbols per account; two connections (app + recorder) were
    verified to run side by side on 2026-10-09.
  * login goes through arrow_auth (shared .arrow_session.json cache with the app, so whichever starts first logs in once a day).
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from .sources import AuthImportError, RestResult, TokenProvider

log = logging.getLogger("collector")
IST = timezone(timedelta(hours=5, minutes=30))
INDEX_SYMBOL = "NSE:NIFTY50-INDEX"
VIX_SYMBOL = "NSE:INDIAVIX-INDEX"
INDEX_TOKEN, VIX_TOKEN = 26000, 26017
WS_LIMIT = 1024
WS_LABEL = "arrow:ws-full"
CHAIN_LABEL = "arrow:rest-chain"


def _ac():
    try:
        import arrow_chain
        return arrow_chain
    except (KeyError, ImportError) as e:
        raise AuthImportError(f"cannot import arrow_chain ({type(e).__name__}: {e}); run from the repo root with the app's .env present") from e


class ArrowTokenProvider(TokenProvider):
    """Today's Arrow token via arrow_auth (cache first; logs in once a day if neither the app nor an earlier run has)."""

    def current(self) -> Optional[Tuple[str, str]]:
        try:
            import arrow_auth
        except (KeyError, ImportError) as e:
            raise AuthImportError(f"cannot import arrow_auth ({type(e).__name__}: {e})") from e
        try:
            return (arrow_auth.APP_ID, arrow_auth.get_token())
        except Exception as e:
            log.warning("Arrow token not available: %s", e)
            return None


class ArrowRestClient:
    """get(name, url, params) with options-chain-v3 params {symbol, strikecount, timestamp} -> RestResult whose body is options-chain-v3
    shaped. Spacing >= min_gap_s between HTTP calls (Arrow: 10 req/s per endpoint)."""

    def __init__(self, auth_fn=None, session=None, sleep=time.sleep, clock=time.monotonic, wall=time.time, min_gap_s: float = 0.25, timeout: float = 15.0):
        self._s, self._sleep, self._clock, self._wall, self._gap, self._timeout, self._last = session, sleep, clock, wall, min_gap_s, timeout, None
        self.calls = 0
        self.spot_fn = None          # set by the recorder: () -> latest index LTP from its websocket (or None)
        self.vix_fn = None

    def _http(self, method, url, body=None):
        if self._s is None:
            import requests
            self._s = requests
        if self._last is not None:
            wait = self._gap - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        import arrow_auth
        self.calls += 1
        try:
            r = self._s.request(method, url, json=body, headers=arrow_auth.headers(), timeout=self._timeout)
        finally:
            self._last = self._clock()
        try:
            j = r.json()
        except Exception:
            j = None
        return r.status_code, j, r.headers.get("Date")

    def get(self, name: str, url: str, params: Optional[dict] = None) -> RestResult:
        ac = _ac()
        t0 = self._clock()
        status, date_hdr = None, None
        try:
            inst = ac.instruments()
            exps = inst["expiries"]
            ts = (params or {}).get("timestamp") or ""
            if ts:
                e = min(exps, key=lambda x: abs(x["ts"] - float(ts)))
            else:
                e = next((x for x in exps if x["ts"] > time.time()), exps[-1])
            n = int((params or {}).get("strikecount") or 16)
            status, oc, date_hdr = self._http("POST", f"{ac.EDGE}/info/option-chain",
                                              {"underlying": "NIFTY", "exchange": "INDEX", "expiry": e["date"].strftime("%d-%b-%Y").upper(), "count": str(n)})
            if status == 429 or not isinstance(oc, dict) or oc.get("status") != "success":
                return RestResult(name, status, {"s": "error", "code": -429 if status == 429 else status, "message": (oc or {}).get("message", "") if isinstance(oc, dict) else ""},
                                  round((self._clock() - t0) * 1000, 1), None, date_hdr, self._wall())
            opening = {int(x["token"]): int(float(x["openingOI"])) for x in oc.get("data") or [] if x.get("token") and x.get("openingOI") not in (None, "")}
            recs = [inst["by_token"][t] for t in opening if t in inst["by_token"]]
            fut = inst["futs"][0] if inst["futs"] else None
            want = recs + ([fut] if fut else [])
            quotes = {}
            for i in range(0, len(want), 100):
                batch = want[i:i + 100]
                status, qj, date_hdr = self._http("POST", f"{ac.EDGE}/info/quotes/full", [{"exchange": "NFO", "symbol": r["tsym"]} for r in batch])
                if status == 429 or not isinstance(qj, dict) or qj.get("status") != "success":
                    return RestResult(name, status, {"s": "error", "code": -429 if status == 429 else status, "message": (qj or {}).get("message", "") if isinstance(qj, dict) else ""},
                                      round((self._clock() - t0) * 1000, 1), None, date_hdr, self._wall())
                tok_of = {r["tsym"]: r["token"] for r in batch}
                for q in qj.get("data") or []:
                    tok = q.get("token") or tok_of.get(q.get("symbol"))
                    if tok is not None:
                        quotes[int(tok)] = q
            spot = self.spot_fn() if self.spot_fn else None
            if not spot:                                  # no websocket yet (bootstrap) or no tick: the last 1-min close, up to a week back
                today = datetime.now(IST).date()
                c = ac.history_candles(INDEX_SYMBOL, (today - timedelta(days=7)).isoformat(), today.isoformat(), 1)
                spot = c[-1][4] if c else None
            vix = self.vix_fn() if self.vix_fn else None
            px = lambda v: None if v is None else v / 100
            lvl = lambda q, side: ((q.get(side) or [{}])[0] or {})
            rows = [{"strike_price": -1, "ltp": spot, "fp": px((quotes.get(fut["token"]) or {}).get("ltp")) if fut else None}]
            for r in recs:
                q = quotes.get(r["token"])
                if not q:
                    continue
                oi, prev = q.get("oi"), opening.get(r["token"])
                b, a = lvl(q, "bids"), lvl(q, "asks")
                rows.append({"symbol": r["fyers"], "strike_price": r["strike"], "option_type": r["type"], "ltp": px(q.get("ltp")),
                             "bid": px(b.get("price")) if b.get("quantity") else 0.0, "ask": px(a.get("price")) if a.get("quantity") else 0.0,
                             "volume": q.get("volume"), "oi": oi, "prev_oi": prev, "oich": (oi - prev) if oi is not None and prev is not None else None})
            body = {"s": "ok", "code": 200, "data": {"optionsChain": rows, "expiryData": ac.list_expiries(), "indiavixData": {"ltp": vix}}}
            return RestResult(name, 200, body, round((self._clock() - t0) * 1000, 1), None, date_hdr, self._wall())
        except AuthImportError:
            raise
        except Exception as e:
            return RestResult(name, status, None, round((self._clock() - t0) * 1000, 1), f"{type(e).__name__}: {str(e)[:200]}", date_hdr, self._wall())


class ArrowWsFeed:
    """Same public surface as sources.WsFeed; snapshot() is keyed by Fyers-style symbol with Fyers field names."""

    def __init__(self, auth_fn=None, factory=None, wall=time.time, spawn=None):
        self._wall, self._factory, self._spawn = wall, factory, spawn
        self._lock = threading.Lock()
        self._ticks: Dict[str, Tuple[float, dict]] = {}
        self.subscribed: set = set()
        self.connected = False
        self.ticks_total = 0
        self.events: List[dict] = []
        self.invalid_symbols: set = set()
        self._ws = None
        self._stop = threading.Event()
        self._tok2sym: Dict[int, str] = {}
        self._sym2tok: Dict[str, int] = {}

    def _ev(self, kind, detail=""):
        with self._lock:
            self.events.append(dict(t=self._wall(), kind=kind, detail=str(detail)[:300]))
            del self.events[:-200]

    def _resolve(self, symbols):
        inst = _ac().instruments()
        out = []
        for s in symbols:
            if s in self._sym2tok:
                out.append(s); continue
            if s == INDEX_SYMBOL:
                tok = INDEX_TOKEN
            elif s == VIX_SYMBOL:
                tok = VIX_TOKEN
            else:
                rec = inst["by_fyers"].get(s)
                tok = rec["token"] if rec else None
            if tok is None:
                self.invalid_symbols.add(s)
                continue
            self._sym2tok[s], self._tok2sym[tok] = tok, s
            out.append(s)
        return out

    def _send_sub(self, symbols):
        toks = [self._sym2tok[s] for s in symbols]
        for i in range(0, len(toks), 200):
            self._ws.send(json.dumps({"code": "sub", "mode": "full", "full": toks[i:i + 200]}))

    def _on_open(self, ws):
        self.connected = True
        self._ev("connected")
        try:
            self._send_sub(sorted(self.subscribed))
            self._ev("subscribed", len(self.subscribed))
        except Exception as e:
            self._ev("subscribe_failed", e)

    def _on_msg(self, ws, m):
        if not isinstance(m, (bytes, bytearray)):
            return
        t = _ac().parse_tick(m)
        if t is None:
            return
        sym = self._tok2sym.get(t["token"])
        if sym is None:
            return
        rx = self._wall()
        msg = {"symbol": sym, "ltp": t.get("ltp"), "bid_price": t.get("bid"), "ask_price": t.get("ask"), "bid_size": t.get("bid_qty"),
               "ask_size": t.get("ask_qty"), "vol_traded_today": t.get("volume"), "last_traded_qty": t.get("ltq"), "avg_trade_price": t.get("avg"),
               "tot_buy_qty": t.get("tbq"), "tot_sell_qty": t.get("tsq"), "exch_feed_time": t.get("ft"), "last_traded_time": t.get("ltt"), "oi": t.get("oi")}
        with self._lock:
            self._ticks[sym] = (rx, msg)
            self.ticks_total += 1

    def _on_close(self, ws, *a):
        self.connected = False
        self._ev("closed", a)

    def _loop(self, stop):
        import websocket
        import arrow_auth
        backoff = 2
        while not stop.is_set():
            try:
                url = f"{_ac().WS_URL}?appID={arrow_auth.APP_ID}&token={arrow_auth.get_token()}"
                self._ws = (self._factory or websocket.WebSocketApp)(url, on_open=self._on_open, on_message=self._on_msg, on_close=self._on_close,
                                                                     on_error=lambda ws, e: self._ev("error", e))
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
                backoff = 2
            except Exception as e:
                self._ev("error", e)
            self.connected = False
            if stop.wait(backoff):
                break
            backoff = min(backoff * 2, 60)

    def start(self, symbols):
        syms = self._resolve(symbols)
        if len(syms) >= WS_LIMIT:
            raise ValueError(f"{len(syms)} symbols exceed the Arrow websocket limit of {WS_LIMIT}")
        self.subscribed = set(syms)
        self._stop = stop = threading.Event()          # one event per connection loop: a restart never revives the old loop
        if self._spawn is not None:
            self._spawn(lambda: self._loop(stop))
        else:
            threading.Thread(target=self._loop, args=(stop,), daemon=True, name="arrow-ws").start()

    def add_symbols(self, symbols) -> List[str]:
        new = sorted(set(symbols) - self.subscribed - self.invalid_symbols)
        new = self._resolve(new)
        if not new:
            return []
        if len(self.subscribed) + len(new) >= WS_LIMIT:
            self._ev("limit", f"{len(new)} new symbols would exceed {WS_LIMIT}")
            return []
        self.subscribed.update(new)
        if self.connected and self._ws is not None:
            try:
                self._send_sub(new)
                self._ev("subscribed_more", len(new))
            except Exception as e:
                self._ev("subscribe_failed", e)
                log.warning("websocket subscribe of %d new symbols failed: %s", len(new), e)
        return new

    def snapshot(self) -> Dict[str, Tuple[float, dict]]:
        with self._lock:
            return dict(self._ticks)

    def ltp(self, symbol):
        with self._lock:
            t = self._ticks.get(symbol)
        return t[1].get("ltp") if t else None

    def stop(self):
        self._stop.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception as e:
            self._ev("close_failed", e)
        self.connected = False

    def restart(self):
        syms = sorted(self.subscribed)
        self.stop()
        time.sleep(1)
        self.start(syms)
