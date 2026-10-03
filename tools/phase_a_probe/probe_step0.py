"""Phase A, Step 0: live Fyers field and connectivity probe. READ-ONLY.

Answers only: which fields do /data/quotes, /data/options-chain-v3 and /data/depth return; can a second full-mode websocket coexist with the app; what do exch_feed_time and
last_traded_time do in observed data; does the existing cached token work read-only; what are status/latency/rate-limit behaviours.

Safety: HTTP GET only (no POST, no orders, no trading calls); the cached token is read through fyers_auth._load_cached_token() exactly as fyers_option_ws.py does -- this probe NEVER logs in,
never writes the token, never prints it, and refuses to write any file that contains it; no database, no strategy maths. A few REST calls (<= ~10) with >= 1 s spacing; a stop on any 429.
Nothing in the repository is modified: outputs go to --out (default ~/phase_a_probe_output/<timestamp>).

Run during NSE market hours (09:15-15:40 IST on a trading day), ideally WHILE the existing app is running, so that websocket coexistence is actually tested:
    python tools/phase_a_probe/probe_step0.py --existing-app-running yes --ws-seconds 120
"""
import argparse
import json
import os
import platform
import re
import statistics
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
BASE = "https://api-t1.fyers.in"
CHAIN_URL, QUOTES_URL, DEPTH_URL = BASE + "/data/options-chain-v3", BASE + "/data/quotes", BASE + "/data/depth"
MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
INDEX_SYMBOL = "NSE:NIFTY50-INDEX"
REQUIRED_WS_FIELDS = ["bid_price", "ask_price", "bid_size", "ask_size", "ltp", "exch_feed_time", "last_traded_time", "vol_traded_today", "last_traded_qty", "tot_buy_qty", "tot_sell_qty", "avg_trade_price"]
QUOTE_FIELDS_WANTED = ["lp", "bid", "ask", "tt", "volume", "ch", "chp", "spread", "atp", "open_price", "high_price", "low_price", "prev_close_price"]
_SECRETS = []


# ------------------------------------------------------------------------------------------------ safety helpers
def safe_write(path, text):
    for s in _SECRETS:
        if s and s in text:
            raise RuntimeError(f"refusing to write {path}: it contains the access token")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def jdump(obj):
    return json.dumps(obj, indent=1, default=str, sort_keys=False)


def ist(ts):
    try:
        return datetime.fromtimestamp(float(ts), IST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ HTTP (GET only)
class Http:
    """The ONLY network entry point for REST: a GET wrapper that records status, latency, size and rate-limit headers, never the Authorization header."""

    def __init__(self, app_id, token, session=None, sleep=time.sleep, min_gap=1.0):
        import requests
        self._s = session or requests
        self._auth = f"{app_id}:{token}"
        self.log, self._sleep, self._min_gap, self._last = [], sleep, min_gap, 0.0
        self.stop_on_429 = False

    def get(self, name, url, params=None, timeout=15):
        if self.stop_on_429:
            self.log.append(dict(name=name, skipped="a previous call returned HTTP 429; no further REST calls"))
            return None, None
        gap = self._min_gap - (time.monotonic() - self._last)
        if gap > 0:
            self._sleep(gap)
        rec = dict(name=name, url=url, params=params, local_time_ist=datetime.now(IST).isoformat(timespec="milliseconds"))
        t0 = time.monotonic()
        try:
            r = self._s.get(url, headers={"Authorization": self._auth}, params=params, timeout=timeout)
        except Exception as e:
            rec.update(error=type(e).__name__ + ": " + str(e)[:200], latency_ms=round((time.monotonic() - t0) * 1000, 1))
            self.log.append(rec)
            self._last = time.monotonic()
            return rec, None
        self._last = time.monotonic()
        rec["latency_ms"] = round((self._last - t0) * 1000, 1)
        rec["status"] = r.status_code
        rec["bytes"] = len(r.content or b"")
        rec["rate_limit_headers"] = {k: v for k, v in r.headers.items() if re.search(r"rate|limit|retry|remaining", k, re.I)}
        rec["server_date_header"] = r.headers.get("Date")
        try:
            body = r.json()
        except Exception:
            body = None
            rec["non_json_body_head"] = (r.text or "")[:300]
        if isinstance(body, dict):
            rec["body_s"], rec["body_code"], rec["body_message"] = body.get("s"), body.get("code"), body.get("message")
        if r.status_code == 429 or (isinstance(body, dict) and (body.get("code") == -429 or "rate limit" in str(body.get("message", "")).lower())):
            rec["rate_limited"] = True
            self.stop_on_429 = True
        self.log.append(rec)
        return rec, body


# ------------------------------------------------------------------------------------------------ schema inventories (pure)
def type_name(v):
    return type(v).__name__ if v is not None else "null"


def schema_of_rows(rows):
    """{key: {'present_in': n, 'of': N, 'types': [...], 'example': value}} over a list of dict rows."""
    out = {}
    for r in rows:
        for k, v in r.items():
            e = out.setdefault(k, dict(present_in=0, types=set(), example=None))
            e["present_in"] += 1
            e["types"].add(type_name(v))
            if e["example"] is None and v is not None:
                e["example"] = v
    n = len(rows)
    return {k: dict(present_in=e["present_in"], of=n, types=sorted(e["types"]), example=e["example"]) for k, e in sorted(out.items())}


def classify_keys(keys, values=None):
    """Name-based classes only (a hint, never proof); `values` (optional dict) additionally lists keys whose value is a list/dict (possible depth levels)."""
    keys = list(keys)
    return dict(timestamp_like=[k for k in keys if re.search(r"time|(^|_)ts($|_)|^tt$|date", k, re.I)],
                size_or_depth_like=[k for k in keys if re.search(r"size|qty|depth", k, re.I)],
                nested_value_keys=[k for k in keys if values is not None and isinstance(values.get(k), (list, dict))],
                bid_ask_like=[k for k in keys if re.search(r"bid|ask", k, re.I)],
                oi_like=[k for k in keys if re.search(r"(^|_)oi|open_?int", k, re.I)],
                volume_like=[k for k in keys if re.search(r"vol", k, re.I)])


def analyze_chain(body):
    data = (body or {}).get("data") or {}
    rows = [r for r in (data.get("optionsChain") or []) if isinstance(r, dict)]
    opt = [r for r in rows if r.get("strike_price", -1) != -1]
    und = [r for r in rows if r.get("strike_price", -1) == -1]
    sch = schema_of_rows(opt)
    return dict(top_level_keys=sorted((body or {}).keys()), data_keys=sorted(data.keys()), n_option_rows=len(opt), n_underlying_rows=len(und),
                option_row_schema=sch, underlying_row_schema=schema_of_rows(und), option_row_key_classes=classify_keys(sch.keys()),
                expiry_data_sample=(data.get("expiryData") or [])[:3], expiry_data_keys=sorted({k for e in (data.get("expiryData") or []) for k in e}))


def analyze_quotes(body):
    ds = [d for d in ((body or {}).get("d") or []) if isinstance(d, dict)]
    per = {}
    for d in ds:
        v = d.get("v") or {}
        per[d.get("n")] = dict(entry_keys=sorted(d.keys()), status=d.get("s"), v_keys=sorted(v.keys()), v_classes=classify_keys(v.keys(), v), values={k: v.get(k) for k in QUOTE_FIELDS_WANTED if k in v},
                               wanted_missing=[k for k in QUOTE_FIELDS_WANTED if k not in v])
    return dict(top_level_keys=sorted((body or {}).keys()), n_entries=len(ds), per_symbol=per,
                union_v_keys=sorted({k for p in per.values() for k in p["v_keys"]}))


def analyze_depth(rec, body):
    exists = bool(rec and rec.get("status") == 200 and isinstance(body, dict) and body.get("s") == "ok")
    out = dict(http_status=(rec or {}).get("status"), body_s=(rec or {}).get("body_s"), body_message=(rec or {}).get("body_message"), endpoint_returns_data=exists)
    if isinstance(body, dict):
        d = body.get("d")
        inner = list(d.values())[0] if isinstance(d, dict) and d else None
        out["top_level_keys"] = sorted(body.keys())
        if isinstance(inner, dict):
            out["inner_keys"] = sorted(inner.keys())
            out["classes"] = classify_keys(inner.keys(), inner)
            for lvl in ("bids", "ask", "asks", "bid"):
                if isinstance(inner.get(lvl), list) and inner[lvl]:
                    out[f"{lvl}_level_keys"] = sorted(inner[lvl][0].keys())
                    out[f"{lvl}_levels"] = len(inner[lvl])
    return out


# ------------------------------------------------------------------------------------------------ websocket analysis (pure)
def _chg(a, b, k):
    return a.get(k) != b.get(k)


def analyze_symbol_ticks(ticks):
    """ticks: list of (capture_wall_epoch, msg_dict) for ONE symbol, in arrival order. Reports only what the observed data shows."""
    n = len(ticks)
    present = {f: sum(1 for _, m in ticks if m.get(f) is not None) for f in REQUIRED_WS_FIELDS}
    out = dict(n_ticks=n, required_field_present_counts=present, all_required_present=all(c == n and n > 0 for c in present.values()), message_types=sorted({str(m.get("type")) for _, m in ticks}),
               extra_fields=sorted({k for _, m in ticks for k in m} - set(REQUIRED_WS_FIELDS)))
    if n == 0:
        return out
    feed = [(c, m.get("exch_feed_time")) for c, m in ticks if m.get("exch_feed_time") is not None]
    ltt = [(c, m.get("last_traded_time")) for c, m in ticks if m.get("last_traded_time") is not None]
    out["capture_minus_exch_feed_time_s"] = _stats([c - f for c, f in feed])
    out["capture_minus_last_traded_time_s"] = _stats([c - t for c, t in ltt])
    out["exch_feed_time_units_look_like_epoch_seconds"] = bool(feed) and all(1e9 < f < 4e9 for _, f in feed)
    out["exch_feed_time_never_decreases"] = all(feed[i][1] <= feed[i + 1][1] for i in range(len(feed) - 1)) if len(feed) > 1 else None
    out["last_traded_time_never_decreases"] = all(ltt[i][1] <= ltt[i + 1][1] for i in range(len(ltt) - 1)) if len(ltt) > 1 else None
    pairs = {}
    ltt_vs_vol = dict(both_changed=0, ltt_only=0, vol_only=0, neither=0)
    for (_, a), (_, b) in zip(ticks, ticks[1:]):
        trade = _chg(a, b, "ltp") or _chg(a, b, "vol_traded_today") or _chg(a, b, "last_traded_time")
        quote = any(_chg(a, b, k) for k in ("bid_price", "ask_price", "bid_size", "ask_size"))
        feed_chg = _chg(a, b, "exch_feed_time")
        key = f"trade_changed={trade}|quote_changed={quote}|exch_feed_time_changed={feed_chg}"
        pairs[key] = pairs.get(key, 0) + 1
        lc, vc = _chg(a, b, "last_traded_time"), _chg(a, b, "vol_traded_today")
        ltt_vs_vol["both_changed" if lc and vc else "ltt_only" if lc else "vol_only" if vc else "neither"] += 1
    out["consecutive_tick_pairs"] = pairs
    out["last_traded_time_vs_volume_changes"] = ltt_vs_vol
    qo = pairs.get("trade_changed=False|quote_changed=True|exch_feed_time_changed=True", 0)
    qn = pairs.get("trade_changed=False|quote_changed=True|exch_feed_time_changed=False", 0)
    out["quote_only_updates"] = dict(feed_time_advanced=qo, feed_time_unchanged=qn, enough_evidence=(qo + qn) >= 10)
    out["first_tick"] = {k: ticks[0][1].get(k) for k in REQUIRED_WS_FIELDS}
    out["last_tick"] = {k: ticks[-1][1].get(k) for k in REQUIRED_WS_FIELDS}
    out["exch_feed_time_ist_first_last"] = [ist(feed[0][1]), ist(feed[-1][1])] if feed else None
    out["last_traded_time_ist_first_last"] = [ist(ltt[0][1]), ist(ltt[-1][1])] if ltt else None
    return out


def _stats(xs):
    if not xs:
        return None
    return dict(n=len(xs), min=round(min(xs), 3), median=round(statistics.median(xs), 3), max=round(max(xs), 3))


def compare_rest_with_ws(rest_samples, ticks_by_symbol, window=5.0):
    """rest_samples: [(capture_wall, symbol, {'tt','bid','ask','lp',...})]. For each, the nearest-in-time ws tick of the same symbol (within `window` s) is compared field by field."""
    rows = []
    for cap, sym, q in rest_samples:
        ts = ticks_by_symbol.get(sym) or []
        if not ts:
            rows.append(dict(symbol=sym, note="no websocket ticks for this symbol")); continue
        c, m = min(ts, key=lambda t: abs(t[0] - cap))
        if abs(c - cap) > window:
            rows.append(dict(symbol=sym, note=f"nearest websocket tick is {abs(c - cap):.1f}s away (> {window}s)")); continue
        tt = q.get("tt")
        rows.append(dict(symbol=sym, rest_capture_ist=ist(cap), ws_capture_gap_s=round(c - cap, 3), rest_tt=tt, rest_tt_ist=ist(tt) if tt else None,
                         ws_exch_feed_time=m.get("exch_feed_time"), ws_last_traded_time=m.get("last_traded_time"),
                         tt_equals_exch_feed_time=(tt == m.get("exch_feed_time")) if tt is not None else None, tt_equals_last_traded_time=(tt == m.get("last_traded_time")) if tt is not None else None,
                         tt_minus_exch_feed_time_s=(tt - m["exch_feed_time"]) if tt is not None and m.get("exch_feed_time") is not None else None,
                         rest_bid=q.get("bid"), ws_bid=m.get("bid_price"), rest_ask=q.get("ask"), ws_ask=m.get("ask_price"), rest_lp=q.get("lp"), ws_ltp=m.get("ltp"),
                         bid_ask_equal=(q.get("bid") == m.get("bid_price") and q.get("ask") == m.get("ask_price"))))
    return rows


# ------------------------------------------------------------------------------------------------ websocket collector
class WsCollector:
    def __init__(self, app_id, token, symbols, factory=None):
        self.symbols, self.ticks, self.events, self._lock = list(symbols), [], [], threading.Lock()
        self._auth, self._sock, self._factory = f"{app_id}:{token}", None, factory

    def _ev(self, kind, detail=""):
        with self._lock:
            self.events.append(dict(t=time.time(), t_ist=datetime.now(IST).isoformat(timespec="milliseconds"), kind=kind, detail=str(detail)[:300]))

    def on_message(self, msg):
        cap = time.time()
        if isinstance(msg, dict) and msg.get("symbol"):
            with self._lock:
                self.ticks.append((cap, msg))

    def start(self):
        factory = self._factory
        if factory is None:
            from fyers_apiv3.FyersWebsocket import data_ws
            factory = data_ws.FyersDataSocket
        def on_open():
            self._ev("connected")
            self._sock.subscribe(symbols=self.symbols, data_type="SymbolUpdate")
            self._ev("subscribed", len(self.symbols))
        self._sock = factory(access_token=self._auth, log_path="", litemode=False, write_to_file=False, reconnect=False,
                             on_connect=on_open, on_close=lambda m: self._ev("closed", m), on_error=lambda m: self._ev("error", m), on_message=self.on_message)
        threading.Thread(target=self._sock.connect, daemon=True).start()

    def stop(self):
        try:
            self._sock.close_connection()
            self._ev("close_requested")
        except Exception as e:
            self._ev("close_failed", e)

    def by_symbol(self):
        out = {}
        with self._lock:
            for c, m in self.ticks:
                out.setdefault(m["symbol"], []).append((c, m))
        return out


# ------------------------------------------------------------------------------------------------ orchestration
def pick_symbols(chain_body, step_out=4):
    data = (chain_body or {}).get("data") or {}
    rows = data.get("optionsChain") or []
    spot = next((r.get("ltp") for r in rows if r.get("strike_price", -1) == -1), None)
    opts = [r for r in rows if r.get("strike_price", -1) != -1 and r.get("symbol")]
    if not spot or not opts:
        return None
    strikes = sorted({r["strike_price"] for r in opts})
    atm = min(strikes, key=lambda s: abs(s - spot))
    i = strikes.index(atm)
    otm = strikes[min(len(strikes) - 1, i + step_out)]
    find = lambda k, t: next((r["symbol"] for r in opts if r["strike_price"] == k and r.get("option_type") == t), None)
    return dict(spot=spot, atm=atm, otm_call_strike=otm, atm_ce=find(atm, "CE"), atm_pe=find(atm, "PE"), otm_ce=find(otm, "CE"))


def find_future(master_text, now_epoch):
    best = None
    for line in master_text.splitlines():
        p = line.split(",")
        if len(p) < 17 or p[16] != "XX" or p[13] != "NIFTY" or not re.fullmatch(r"NSE:NIFTY\d{2}[A-Z]{3}FUT", p[9]):
            continue
        try:
            exp = int(p[8])
        except ValueError:
            continue
        if exp >= now_epoch and (best is None or exp < best[0]):
            best = (exp, p[9])
    return best[1] if best else None


def market_window_warning(now=None):
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return "weekend"
    t = now.time()
    if not (datetime(2000, 1, 1, 9, 15).time() <= t <= datetime(2000, 1, 1, 15, 40).time()):
        return "outside 09:15-15:40 IST"
    return None


def run(out_dir, http, ws_factory, ws_seconds, existing_app, sleep=time.sleep, master_text=None, app_id="", token=""):
    os.makedirs(out_dir, exist_ok=True)
    summary = dict(probe="phase_a_step0", started_ist=datetime.now(IST).isoformat(timespec="seconds"), python=platform.python_version(), platform=platform.platform(), existing_app_running_asserted_by_operator=existing_app)
    try:
        import importlib.metadata as md
        summary["fyers_apiv3_version"] = md.version("fyers-apiv3")
    except Exception:
        summary["fyers_apiv3_version"] = None
    # 1. option chain (raw)
    rec, chain = http.get("options_chain_v3", CHAIN_URL, {"symbol": INDEX_SYMBOL, "strikecount": 5, "timestamp": ""})
    if chain is not None:
        safe_write(os.path.join(out_dir, "raw_options_chain_v3.json"), jdump(chain))
        summary["options_chain_v3"] = analyze_chain(chain)
    picks = pick_symbols(chain)
    summary["picked_symbols"] = picks
    if not picks:
        summary["fatal"] = "no usable option chain response; later steps skipped"
        _finish(out_dir, summary, http); return summary
    fut = None
    if master_text is not None:
        fut = find_future(master_text, time.time())
    symbols = [s for s in (picks["atm_ce"], picks["atm_pe"], picks["otm_ce"], INDEX_SYMBOL, fut) if s]
    summary["quote_and_ws_symbols"] = symbols
    # 2. quotes (raw), 3. depth
    rec, q = http.get("quotes_initial", QUOTES_URL, {"symbols": ",".join(symbols)})
    if q is not None:
        safe_write(os.path.join(out_dir, "raw_quotes_initial.json"), jdump(q))
        summary["quotes"] = analyze_quotes(q)
    drec, dbody = http.get("depth", DEPTH_URL, {"symbol": picks["atm_ce"], "ohlcv_flag": 1})
    safe_write(os.path.join(out_dir, "raw_depth.json"), jdump(dict(record=drec, body=dbody)))
    summary["depth"] = analyze_depth(drec, dbody)
    # 4. websocket, with a few REST quote samples taken while it runs
    ws = WsCollector(app_id, token, symbols, ws_factory)
    t0 = time.time()
    ws.start()
    rest_samples = []
    sample_at = [ws_seconds * f for f in (0.3, 0.55, 0.8)]
    for at in sample_at:
        wait = t0 + at - time.time()
        if wait > 0:
            sleep(wait)
        cap = time.time()
        r, qb = http.get("quotes_during_ws", QUOTES_URL, {"symbols": ",".join(symbols)})
        for d in ((qb or {}).get("d") or []):
            if isinstance(d, dict) and d.get("v"):
                rest_samples.append((cap, d.get("n"), d["v"]))
    wait = t0 + ws_seconds - time.time()
    if wait > 0:
        sleep(wait)
    ws.stop()
    by = ws.by_symbol()
    with open(os.path.join(out_dir, "ws_ticks.jsonl"), "w", encoding="utf-8") as f:
        for c, m in sorted(ws.ticks, key=lambda t: t[0]):
            line = json.dumps(dict(capture_epoch=c, capture_ist=ist(c), msg=m), default=str)
            for s in _SECRETS:
                if s and s in line:
                    raise RuntimeError("token in tick output")
            f.write(line + "\n")
    summary["websocket"] = dict(seconds=ws_seconds, events=ws.events, n_ticks_total=len(ws.ticks), per_symbol={s: analyze_symbol_ticks(t) for s, t in by.items()},
                                connected=any(e["kind"] == "connected" for e in ws.events), errors=[e for e in ws.events if e["kind"] in ("error", "close_failed")])
    summary["rest_vs_ws"] = compare_rest_with_ws(rest_samples, by)
    _finish(out_dir, summary, http)
    return summary


def clock_skew(http_log):
    out = []
    for r in http_log:
        d = r.get("server_date_header")
        if d and r.get("local_time_ist"):
            try:
                from email.utils import parsedate_to_datetime
                out.append((datetime.fromisoformat(r["local_time_ist"]) - parsedate_to_datetime(d)).total_seconds())
            except Exception:
                pass
    return _stats(out)


def _finish(out_dir, summary, http):
    summary["http_calls"] = http.log
    summary["local_minus_server_date_header_s"] = clock_skew(http.log)
    summary["finished_ist"] = datetime.now(IST).isoformat(timespec="seconds")
    safe_write(os.path.join(out_dir, "probe_summary.json"), jdump(summary))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "phase_a_probe_output", datetime.now(IST).strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--ws-seconds", type=float, default=120)
    ap.add_argument("--existing-app-running", choices=["yes", "no", "unknown"], default="unknown")
    ap.add_argument("--allow-closed", action="store_true", help="run outside market hours (timestamp findings will not be meaningful)")
    a = ap.parse_args(argv)
    warn = market_window_warning()
    if warn and not a.allow_closed:
        print(f"Market not open ({warn}); the probe's timestamp findings need live ticks. Re-run in market hours, or pass --allow-closed.")
        return 2
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        import fyers_auth
    except (KeyError, ImportError) as e:
        print(f"Cannot import the app's fyers_auth ({type(e).__name__}: {e}); run from the app's Python environment with its .env present. Nothing was sent.")
        return 3
    token = fyers_auth._load_cached_token()          # read-only: never logs in
    if not token:
        print("No valid cached token for today (run the app / `python fyers_auth.py` first). The probe does not log in.")
        return 4
    _SECRETS.extend([token, f"{fyers_auth.APP_ID}:{token}"])
    exp = None
    try:
        exp = fyers_auth._jwt_exp(token)
    except Exception:
        pass
    print(f"Cached token found (not printed); expires {ist(exp) if exp else 'unknown'} IST. Output dir: {a.out}")
    master = None
    try:
        import requests
        master = requests.get(MASTER_URL, timeout=30).text
    except Exception as e:
        print(f"symbol master not downloaded ({type(e).__name__}); futures symbol will be skipped")
    http = Http(fyers_auth.APP_ID, token)
    s = run(a.out, http, None, a.ws_seconds, a.existing_app_running, master_text=master, app_id=fyers_auth.APP_ID, token=token)
    print(f"done. ticks={s.get('websocket', {}).get('n_ticks_total')}  files in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
