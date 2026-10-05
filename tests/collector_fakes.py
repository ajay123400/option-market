"""Shared fakes for the collector tests: a fake clock, a fake Fyers world (REST chain + websocket) and a fake fyers_auth module. No network."""
import json
import threading
from datetime import datetime, timedelta
from email.utils import formatdate

from collector.config import IST

EXPIRY_DATA = [dict(date="06-10-2026", expiry="1791281400", expiry_flag="W"), dict(date="13-10-2026", expiry="1791886200", expiry_flag="W"), dict(date="19-10-2026", expiry="1792404600", expiry_flag="W"),
               dict(date="27-10-2026", expiry="1793095800", expiry_flag="M"), dict(date="03-11-2026", expiry="1793700600", expiry_flag="W")]
PREFIX = {"1791281400": "26O06", "1791886200": "26O13", "1792404600": "26O19"}
FUT = "NSE:NIFTY26OCTFUT"
INDEX = "NSE:NIFTY50-INDEX"


class FakeClock:
    def __init__(self, start: datetime):
        self.t = start.timestamp()
        self.stop = threading.Event()
        self.hooks = []

    def now(self):
        return datetime.fromtimestamp(self.t, IST)

    def wall(self):
        return self.t

    def monotonic(self):
        return self.t

    def sleep(self, s):
        if s > 0:
            self.t += s
            for h in list(self.hooks):
                h()


class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}

    def json(self):
        if self._b is None:
            raise ValueError("no json")
        return self._b


class World:
    """Spot, chain responses (strikecount honoured), OI that changes per call, websocket ticks for whatever is subscribed, scripted failures."""

    def __init__(self, clock, spot=22501.45, skew=1.5):
        self.clock, self.spot, self.skew = clock, spot, skew
        self.chain_calls = []                     # (timestamp param, strikecount, fake time)
        self.rest_calls = 0
        self.rate_limit_on = set()                # 1-based REST call numbers that return 429
        self.fail_status = None                   # every REST call returns this status
        self.sock = None
        self.vol = {}
        self.silent_symbols = set()               # no websocket ticks for these
        self.drop_index = False

    # ---- REST
    def session(self):
        w = self

        class S:
            def get(self, url, headers=None, params=None, timeout=None):
                assert url.endswith("/data/options-chain-v3"), url
                assert "Authorization" in headers
                return w.chain(params)
        return S()

    def chain(self, params):
        self.rest_calls += 1
        self.chain_calls.append((params.get("timestamp"), params.get("strikecount"), self.clock.t))
        hdr = {"Date": formatdate(int(self.clock.t - self.skew), usegmt=True)}
        if self.rest_calls in self.rate_limit_on:
            return FakeResp(429, {"s": "error", "code": -429, "message": "rate limit"}, hdr)
        if self.fail_status:
            return FakeResp(self.fail_status, None, hdr)
        ts = params.get("timestamp") or EXPIRY_DATA[0]["expiry"]
        n = int(params["strikecount"])
        atm = int(round(self.spot / 50) * 50)
        rows = [dict(strike_price=-1, ltp=self.spot, symbol=INDEX, fp=22611.0, ask=0, bid=0)]
        for k in range(atm - 50 * n, atm + 50 * n + 1, 50):
            for t in ("CE", "PE"):
                sym = f"NSE:NIFTY{PREFIX[ts]}{k}{t}"
                rows.append(dict(strike_price=k, option_type=t, symbol=sym, ltp=100.0, bid=99.9, ask=100.1, oi=1000000 + self.rest_calls * 10, oich=5, prev_oi=999000, volume=5000, fyToken="1"))
        return FakeResp(200, dict(code=200, s="ok", message="", data=dict(optionsChain=rows, expiryData=EXPIRY_DATA, indiavixData=dict(ltp=14.5), callOi=1, putOi=2)), hdr)

    # ---- websocket
    def factory(self):
        w = self

        class Sock:
            def __init__(self, access_token, log_path, litemode, write_to_file, reconnect, on_connect, on_close, on_error, on_message):
                assert litemode is False and write_to_file is False and ":" in access_token
                self.on_connect, self.on_message, self.subscribed, self.closed, self.sub_calls = on_connect, on_message, [], False, []
                w.sock = self

            def connect(self):
                self.on_connect()
                w.emit()

            def subscribe(self, symbols, data_type):
                assert data_type == "SymbolUpdate" and len(symbols) <= 500
                self.subscribed += list(symbols)
                self.sub_calls.append(list(symbols))
                w.emit()

            def close_connection(self):
                self.closed = True
        return Sock

    def emit(self):
        if self.sock is None or self.sock.closed:
            return
        t = int(self.clock.t)
        for s in sorted(set(self.sock.subscribed)):
            if s in self.silent_symbols or (s == INDEX and self.drop_index):
                continue
            if s == INDEX:
                self.sock.on_message(dict(symbol=s, type="if", ltp=self.spot, exch_feed_time=t - 1))
                continue
            self.vol[s] = self.vol.get(s, 1000) + 65
            self.sock.on_message(dict(symbol=s, type="sf", ltp=100.0, bid_price=99.9, ask_price=100.1, bid_size=130, ask_size=195, vol_traded_today=self.vol[s], exch_feed_time=t - 1, last_traded_time=t - 3,
                                      last_traded_qty=65, tot_buy_qty=10, tot_sell_qty=12, avg_trade_price=99.5))


class FakeAuth:
    """Stands in for fyers_auth: only the cache reader exists; any login path raises."""
    APP_ID = "APP-100"

    def __init__(self, token="TOKEN.SECRET", available_after_calls=0):
        self.token, self.after, self.calls = token, available_after_calls, 0

    def _load_cached_token(self):
        self.calls += 1
        return self.token if self.calls > self.after else None

    def __getattr__(self, name):
        if name in ("login", "get_access_token", "get_auth_header", "_save_cached_token", "exchange_auth_code"):
            raise AssertionError(f"the recorder must never call fyers_auth.{name}")
        raise AttributeError(name)


def start(hh=11, mm=0, ss=0):
    return datetime(2026, 10, 5, hh, mm, ss, tzinfo=IST)
