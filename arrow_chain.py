"""arrow_chain.py -- the option chain, quotes and history candles from Arrow
(iRage), returned in EXACTLY the shapes fyers_option_chain / historical_
recorder already return, with Fyers-style symbols ("NSE:NIFTY26O1322650CE"),
so every caller (Option Chain page, plan, range engine, journal, paper book)
works unchanged. fyers_option_chain switches to these functions when .env
has DATA_SOURCE=arrow.

Why: from 9 Oct 2026 the Fyers Standard plan allows 5,000 data calls a day
and 50 websocket symbols -- the Option Chain page alone used more. Arrow:
10 req/s per endpoint, no daily cap, 1,024 websocket symbols.

Live data = one websocket (wss://ds.arrow.trade, "full" mode: LTP, close,
volume, OI, exchange times, 5-level depth; big-endian binary, prices in
paise). Tokens are subscribed on demand the first time a chain/quote needs
them; until a token's first tick arrives its row is seeded from REST
/info/quotes/full (<= 100 symbols a call). Previous-day OI (for OI change)
comes from /info/option-chain's openingOI, once per expiry per day.
Packet layout and endpoints follow the broker's Go SDK (arrow/streams.go).
"""
import csv
import io
import json
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

import arrow_auth
import paths

IST = timezone(timedelta(hours=5, minutes=30))
EDGE = "https://edge.arrow.trade"
HIST = "https://historical-api.arrow.trade"
WS_URL = "wss://ds.arrow.trade"
INDEX_TOKEN = 26000                       # NSEIDX "Nifty 50"
INDEX_SYMBOL = "NSE:NIFTY50-INDEX"        # the app's (Fyers) name for it
STEP = 50
_MASTER_CACHE = Path(paths.BASE_DIR) / "data" / "arrow_nifty_instruments.json"
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_WEEKLY_M = "123456789OND"


class ArrowDataError(RuntimeError):
    pass


def _session_check(body):
    msg = str((body or {}).get("message") or "").lower() if isinstance(body, dict) else ""
    if "session" in msg or "unauthor" in msg:
        arrow_auth.invalidate()


# ---------------------------------------------------------------- instruments
_inst = None                   # built once per day
_inst_lock = threading.Lock()


def _expiry_ts(d):
    # 15:40 IST -- the exact convention of Fyers' expiry timestamps, which the
    # paper book, manual trades and Telegram history already store as keys
    return datetime(d.year, d.month, d.day, 15, 40, tzinfo=IST).timestamp()


def fyers_symbol(exp_date, strike, typ, monthly):
    """Fyers' name for a NIFTY option: weekly NIFTY{yy}{m}{dd}{strike}{CE|PE}
    (m = 1-9, O, N, D), the month's last expiry NIFTY{yy}{MON}{strike}{CE|PE}."""
    k = int(strike) if float(strike).is_integer() else strike
    if monthly:
        return f"NSE:NIFTY{exp_date:%y}{_MONTHS[exp_date.month - 1]}{k}{typ}"
    return f"NSE:NIFTY{exp_date:%y}{_WEEKLY_M[exp_date.month - 1]}{exp_date:%d}{k}{typ}"


def _download_master():
    r = requests.get(f"{EDGE}/nse", headers=arrow_auth.headers(), timeout=90)
    if r.status_code != 200 or not r.text.startswith("Exchange"):
        r = requests.get(f"{EDGE}/all", headers=arrow_auth.headers(), timeout=180)
    rows = csv.reader(io.StringIO(r.text))
    head = next(rows)
    I = {k: i for i, k in enumerate(head)}
    opts, futs = [], []
    for x in rows:
        if len(x) < len(head) or x[I["Underlying"]] != "NIFTY" or x[I["ExchSeg"]] != "NSEFO":
            continue
        rec = {"token": int(x[I["Token"]]), "tsym": x[I["TradingSymbol"]], "expiry": x[I["Expiry"]],
               "lot": int(float(x[I["LotSize"]] or 0))}
        if x[I["OptionType"]] in ("CE", "PE"):
            rec.update(strike=float(x[I["StrikePrice"]]), type=x[I["OptionType"]])
            opts.append(rec)
        elif x[I["OptionType"]] == "XX":
            futs.append(rec)
    if not opts:
        raise ArrowDataError("Arrow symbol master has no NIFTY options")
    return {"date": time.strftime("%Y-%m-%d"), "opts": opts, "futs": futs}


def instruments():
    """{'expiries': [{date, ts, monthly}], 'by_key': {(date, strike, type): rec},
    'by_fyers': {fyers_symbol: rec}, 'by_token': {token: rec}, 'futs': [...]}"""
    global _inst
    today = time.strftime("%Y-%m-%d")
    if _inst and _inst["date"] == today:
        return _inst
    with _inst_lock:
        if _inst and _inst["date"] == today:
            return _inst
        raw = None
        try:
            raw = json.loads(_MASTER_CACHE.read_text())
            if raw.get("date") != today:
                raw = None
        except (OSError, ValueError):
            raw = None
        if raw is None:
            raw = _download_master()
            _MASTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _MASTER_CACHE.write_text(json.dumps(raw))
        now = time.time()
        dates = sorted({datetime.strptime(o["expiry"], "%d-%b-%Y").date() for o in raw["opts"]})
        dates = [d for d in dates if _expiry_ts(d) > now - 6 * 3600]
        last_in_month = {}
        for d in dates:
            last_in_month[(d.year, d.month)] = d
        exps = [{"date": d, "ts": _expiry_ts(d), "monthly": last_in_month[(d.year, d.month)] == d} for d in dates]
        monthly = {e["date"]: e["monthly"] for e in exps}
        by_key, by_fyers, by_token = {}, {}, {}
        for o in raw["opts"]:
            d = datetime.strptime(o["expiry"], "%d-%b-%Y").date()
            if d not in monthly:
                continue
            rec = dict(o, date=d, fyers=fyers_symbol(d, o["strike"], o["type"], monthly[d]))
            by_key[(d, o["strike"], o["type"])] = rec
            by_fyers[rec["fyers"]] = rec
            by_token[rec["token"]] = rec
        futs = []
        for f in raw["futs"]:
            d = datetime.strptime(f["expiry"], "%d-%b-%Y").date()
            if _expiry_ts(d) > now - 6 * 3600:
                rec = dict(f, date=d, fyers=f"NSE:NIFTY{d:%y}{_MONTHS[d.month - 1]}FUT")
                futs.append(rec)
                by_fyers[rec["fyers"]] = rec
                by_token[rec["token"]] = rec
        futs.sort(key=lambda r: r["date"])
        _inst = {"date": today, "expiries": exps, "by_key": by_key, "by_fyers": by_fyers, "by_token": by_token,
                 "futs": futs}
        return _inst


# ---------------------------------------------------------------- live feed
_i32 = struct.Struct(">i").unpack_from
_i64 = struct.Struct(">q").unpack_from
_i16 = struct.Struct(">h").unpack_from


def parse_tick(b):
    """One binary market packet -> dict (prices in rupees). None for control frames."""
    n = len(b)
    if n < 13:
        return None
    t = {"token": _i32(b, 0)[0], "ltp": _i32(b, 4)[0] / 100}
    if n in (17, 33):
        t["close"] = _i32(b, 13)[0] / 100
    if n in (93, 109, 241, 249, 265):
        t.update(close=_i32(b, 45)[0] / 100, volume=_i64(b, 53)[0], ltt=_i32(b, 61)[0], ft=_i32(b, 65)[0],
                 oi=_i64(b, 69)[0], tbq=_i64(b, 21)[0], tsq=_i64(b, 29)[0], ltq=_i32(b, 13)[0], avg=_i32(b, 17)[0] / 100)
    if n in (241, 249, 265):
        off = 109 if n >= 249 else 101
        bq, bp = _i64(b, off)[0], _i32(b, off + 8)[0]
        aq, ap = _i64(b, off + 70)[0], _i32(b, off + 78)[0]
        t.update(bid=bp / 100 if bq else 0.0, ask=ap / 100 if aq else 0.0, bid_qty=bq, ask_qty=aq)
    return t


class _Feed:
    def __init__(self):
        self.ticks = {}               # token -> latest tick dict (+ 'rx')
        self.subscribed = set()
        self.lock = threading.Lock()
        self.ws = None
        self.connected = False
        self.thread = None
        self.last_rx = 0.0

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True, name="arrow-feed")
        self.thread.start()

    def _run(self):
        import websocket
        backoff = 2
        while True:
            try:
                url = f"{WS_URL}?appID={arrow_auth.APP_ID}&token={arrow_auth.get_token()}"
                self.ws = websocket.WebSocketApp(url, on_open=self._on_open, on_message=self._on_msg,
                                                 on_close=self._on_close, on_error=lambda ws, e: None)
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
                backoff = 2
            except Exception:
                pass
            self.connected = False
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def _on_open(self, ws):
        self.connected = True
        with self.lock:
            toks = list(self.subscribed)
        for i in range(0, len(toks), 200):
            ws.send(json.dumps({"code": "sub", "mode": "full", "full": toks[i:i + 200]}))

    def _on_close(self, ws, *a):
        self.connected = False

    def _on_msg(self, ws, m):
        if not isinstance(m, (bytes, bytearray)):
            return
        t = parse_tick(m)
        if t is None:
            return
        now = time.time()
        t["rx"] = now
        self.last_rx = now
        with self.lock:
            old = self.ticks.get(t["token"])
            if old:
                old.update(t)
            else:
                self.ticks[t["token"]] = t

    def subscribe(self, tokens):
        new = [int(t) for t in tokens if int(t) not in self.subscribed]
        new = new[:max(0, _WS_MAX - len(self.subscribed))]   # beyond the cap: served over REST instead
        if not new:
            return []
        with self.lock:
            self.subscribed.update(new)
        self.start()
        if self.connected and self.ws:
            try:
                for i in range(0, len(new), 200):
                    self.ws.send(json.dumps({"code": "sub", "mode": "full", "full": new[i:i + 200]}))
            except Exception:
                pass
        return new

    def get(self, token):
        with self.lock:
            t = self.ticks.get(int(token))
            return dict(t) if t else None

    def put(self, token, data, force=False):
        with self.lock:
            if force or int(token) not in self.ticks:
                self.ticks[int(token)] = data


_WS_MAX = 950        # Arrow allows 1,024 websocket symbols per account
FEED = _Feed()


def _rest_quotes(recs, force=False):
    """Seed / refresh rows over REST for instruments without a fresh tick."""
    for i in range(0, len(recs), 100):
        batch = recs[i:i + 100]
        body = [{"exchange": "NFO", "symbol": r["tsym"]} for r in batch]
        r = requests.post(f"{EDGE}/info/quotes/full", json=body, headers=arrow_auth.headers(), timeout=15)
        try:
            j = r.json()
        except ValueError:
            raise ArrowDataError(f"quotes: HTTP {r.status_code}")
        if j.get("status") != "success":
            _session_check(j)
            raise ArrowDataError(f"quotes failed: {j.get('message')}")
        tok_of = {x["tsym"]: x["token"] for x in batch}
        now = time.time()
        for q in j.get("data") or []:
            tok = q.get("token") or tok_of.get(q.get("symbol"))
            if tok is None:
                continue
            bids, asks = q.get("bids") or [{}], q.get("asks") or [{}]
            FEED.put(tok, {"token": int(tok), "ltp": (q.get("ltp") or 0) / 100, "close": (q.get("close") or 0) / 100,
                           "volume": q.get("volume"), "oi": q.get("oi"), "ltt": q.get("ltt"), "ft": q.get("time") or q.get("ltt"),
                           "bid": (bids[0].get("price") or 0) / 100, "ask": (asks[0].get("price") or 0) / 100,
                           "rx": now, "rest": True}, force=force)


def _rows_for(recs, wait=1.5):
    """Latest tick for each instrument: subscribe, wait briefly for first
    ticks, REST-seed whatever is still missing."""
    FEED.subscribe([r["token"] for r in recs])
    deadline = time.time() + wait
    missing = [r for r in recs if FEED.get(r["token"]) is None]
    while missing and FEED.connected and time.time() < deadline:
        time.sleep(0.1)
        missing = [r for r in missing if FEED.get(r["token"]) is None]
    unsubscribed = [r for r in recs if r["token"] not in FEED.subscribed]
    if not FEED.connected:
        _rest_quotes(recs, force=True)      # websocket down: every row from REST, never a frozen tick
    elif missing or unsubscribed:
        if missing:
            _rest_quotes(missing)
        if unsubscribed:                    # over the websocket cap: always fresh from REST
            _rest_quotes(unsubscribed, force=True)
    return {r["token"]: FEED.get(r["token"]) for r in recs}


def spot():
    FEED.subscribe([INDEX_TOKEN])
    deadline = time.time() + 3
    t = FEED.get(INDEX_TOKEN)
    while (t is None or not t.get("ltp")) and time.time() < deadline:
        time.sleep(0.1)
        t = FEED.get(INDEX_TOKEN)
    if not t or not t.get("ltp"):
        today = datetime.now(IST).date().isoformat()        # feed down: last 1-min close
        c = history_candles(INDEX_SYMBOL, today, today, 1)
        if c:
            return c[-1][4]
        raise ArrowDataError("no NIFTY 50 tick from the Arrow feed yet")
    return t["ltp"]


# ---------------------------------------------------------------- opening OI
_open_oi = {}      # (date, expiry) -> {token: openingOI}


def _opening_oi(exp_date):
    key = (time.strftime("%Y-%m-%d"), exp_date)
    if key in _open_oi:
        return _open_oi[key]
    out = {}
    try:
        r = requests.post(f"{EDGE}/info/option-chain", headers=arrow_auth.headers(), timeout=15,
                          json={"underlying": "NIFTY", "exchange": "INDEX", "expiry": exp_date.strftime("%d-%b-%Y").upper(),
                                "count": "60"})
        j = r.json()
        for x in j.get("data") or []:
            if x.get("token") and x.get("openingOI") not in (None, ""):
                out[int(x["token"])] = int(float(x["openingOI"]))
    except Exception:
        return {}
    if out:
        _open_oi[key] = out
    return out


# ---------------------------------------------------------------- public API (Fyers shapes)
def _expiry_rows(inst):
    return [{"date": e["date"].strftime("%d-%m-%Y"), "expiry": str(int(e["ts"])),
             "expiry_flag": "M" if e["monthly"] else "W"} for e in inst["expiries"]]


def list_expiries(symbol=INDEX_SYMBOL):
    return _expiry_rows(instruments())


def get_chain(symbol=INDEX_SYMBOL, strikecount=10, expiry_timestamp="", include_greeks=False):
    inst = instruments()
    exps = inst["expiries"]
    if not exps:
        raise ArrowDataError("no live NIFTY expiries in the Arrow master")
    if expiry_timestamp:
        want = float(expiry_timestamp)
        e = min(exps, key=lambda x: abs(x["ts"] - want))
        if abs(e["ts"] - want) > 86400:
            raise ArrowDataError(f"expiry {expiry_timestamp} is not listed")
    else:
        now = time.time()
        e = next((x for x in exps if x["ts"] > now), exps[-1])
    s = spot()
    atm = round(s / STEP) * STEP
    lo, hi = atm - strikecount * STEP, atm + strikecount * STEP
    recs = [r for (d, k, typ), r in inst["by_key"].items() if d == e["date"] and lo <= k <= hi]
    ticks = _rows_for(recs)
    oi0 = _opening_oi(e["date"])
    by_strike = {}
    for r in recs:
        t = ticks.get(r["token"]) or {}
        ltp, close, oi = t.get("ltp"), t.get("close"), t.get("oi")
        prev = oi0.get(r["token"])
        k = int(r["strike"]) if float(r["strike"]).is_integer() else r["strike"]
        by_strike.setdefault(k, {"strike": k, "pe": None, "ce": None})
        by_strike[k]["ce" if r["type"] == "CE" else "pe"] = {
            "symbol": r["fyers"],
            "ltp": ltp,
            "ltpch": round(ltp - close, 2) if ltp is not None and close else None,
            "ltpchp": round((ltp - close) / close * 100, 2) if ltp is not None and close else None,
            "oi": oi,
            "oich": (oi - prev) if oi is not None and prev is not None else None,
            "oichp": round((oi - prev) / prev * 100, 2) if oi is not None and prev else None,
            "prev_oi": prev,
            "volume": t.get("volume"),
            "bid": t.get("bid"),
            "ask": t.get("ask"),
        }
    strikes = [by_strike[k] for k in sorted(by_strike)]
    out = {"spot": s, "expiries": _expiry_rows(inst), "strikes": strikes, "resolved_expiry_ts": e["ts"]}
    if include_greeks and s:
        import fyers_option_chain
        fyers_option_chain.add_greeks(out)
    return out


def get_quotes(symbols):
    """{symbol: {'ltp','bid','ask','tt','ch','chp'}} for Fyers-style symbols
    (options, NIFTY futures, the index)."""
    inst = instruments()
    out, recs = {}, []
    for sym in dict.fromkeys(symbols):
        if sym == INDEX_SYMBOL:
            try:
                out[sym] = {"ltp": spot(), "bid": None, "ask": None, "tt": int(time.time()), "ch": None, "chp": None}
            except ArrowDataError:
                pass
            continue
        rec = inst["by_fyers"].get(sym)
        if rec:
            recs.append(rec)
    ticks = _rows_for(recs) if recs else {}
    for r in recs:
        t = ticks.get(r["token"])
        if not t or t.get("ltp") is None:
            continue
        close = t.get("close")
        out[r["fyers"]] = {"ltp": t["ltp"], "bid": t.get("bid"), "ask": t.get("ask"), "tt": t.get("ft") or t.get("ltt"),
                           "ch": round(t["ltp"] - close, 2) if close else None,
                           "chp": round((t["ltp"] - close) / close * 100, 2) if close else None}
    return out


_RES = {1: "min", 3: "3min", 5: "5min", 10: "10min", 15: "15min", 30: "30min", 60: "hour", "D": "day"}


def history_candles(fyers_symbol, from_date, to_date, resolution=5):
    """[(epoch, open, high, low, close, volume, oi), ...] like Fyers /data/history
    (dates 'YYYY-MM-DD', inclusive). [] on any failure."""
    try:
        if fyers_symbol == INDEX_SYMBOL:
            ex, tok = "nse", INDEX_TOKEN
        else:
            rec = instruments()["by_fyers"].get(fyers_symbol)
            if not rec:
                return []
            ex, tok = "nfo", rec["token"]
        res = _RES.get(resolution if resolution == "D" else int(resolution), "5min")
        h = arrow_auth.headers()
        h["appID"] = h["appId"]
        day = res == "day"
        r = requests.get(f"{HIST}/candle/{ex}/{tok}/{res}", headers=h, timeout=20,
                         params={"from": f"{from_date}T{'00:00:00' if day else '09:00:00'}",
                                 "to": f"{to_date}T{'23:59:59' if day else '15:35:00'}", "oi": "1"})
        rows = r.json()
        if not isinstance(rows, list):
            _session_check(rows)
            return []
        out = []
        for x in rows:
            ts = datetime.strptime(x[0], "%Y-%m-%dT%H:%M:%S%z").timestamp()
            out.append((int(ts), x[1] / 100, x[2] / 100, x[3] / 100, x[4] / 100, x[5], x[6] if len(x) > 6 else 0))
        return out
    except Exception:
        return []


def status():
    return {"connected": FEED.connected, "subscribed": len(FEED.subscribed),
            "last_tick_age_s": round(time.time() - FEED.last_rx, 1) if FEED.last_rx else None}


def get_margin(legs, lot_size):
    """Exchange margin (SPAN + exposure, with hedge benefit) for a basket,
    from Arrow's /margin/basket -- same shape as the old Fyers get_margin.
    legs: [{'symbol' (Fyers-style), 'side': BUY/SELL, 'qty': lots}]. BUY
    legs first, as with Fyers (the response's final_margin is the hedged
    figure either way; initial_margin depends on order)."""
    inst = instruments()
    orders = []
    for l in sorted(legs, key=lambda l: 0 if l["side"] == "BUY" else 1):
        rec = inst["by_fyers"].get(l["symbol"])
        if not rec:
            raise ArrowDataError(f"margin: unknown symbol {l['symbol']}")
        orders.append({"exchange": "NFO", "symbol": rec["tsym"], "quantity": str(int(l["qty"]) * lot_size),
                       "price": "0", "product": "M", "transactionType": "B" if l["side"] == "BUY" else "S",
                       "order": "MKT", "includePositions": False})
    r = requests.post(f"{EDGE}/margin/basket", json={"orders": orders, "includePositions": False},
                      headers=arrow_auth.headers(), timeout=15)
    j = r.json()
    if j.get("status") != "success":
        _session_check(j)
        raise ArrowDataError(f"margin failed: {j.get('message')}")
    d = j.get("data") or {}
    return {"margin_total": d.get("final_margin"), "margin_avail": None}
