"""Phase A Step 0 probe: offline tests of its pure analysis, its read-only/GET-only/no-token guarantees and its orchestration against fakes (no network, no Fyers)."""
import importlib.util
import json
import os
import time

import pytest

_p = os.path.join(os.path.dirname(__file__), "..", "tools", "phase_a_probe", "probe_step0.py")
spec = importlib.util.spec_from_file_location("probe_step0", _p)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

TOKEN = "SECRET.TOKEN.VALUE"


def chain_body():
    rows = [dict(strike_price=-1, ltp=24417.6, symbol="NSE:NIFTY50-INDEX")]
    for k in (24350, 24400, 24450, 24500, 24550):
        for t in ("CE", "PE"):
            rows.append(dict(strike_price=k, option_type=t, symbol=f"NSE:NIFTY26O06{k}{t}", ltp=100.0, bid=99.5, ask=100.5, oi=1000, oich=5, prev_oi=995, volume=1234, fyToken="1011"))
    return dict(code=200, s="ok", data=dict(optionsChain=rows, expiryData=[dict(date="06-10-2026", expiry="1791000000", expiry_flag="W")]))


def tick(sym="NSE:X", ltp=10.0, bid=9.9, ask=10.1, bs=100, as_=200, vol=1000, feed=1788603960, ltt=1788603900, qty=65):
    return dict(symbol=sym, type="sf", ltp=ltp, bid_price=bid, ask_price=ask, bid_size=bs, ask_size=as_, vol_traded_today=vol, exch_feed_time=feed, last_traded_time=ltt,
                last_traded_qty=qty, tot_buy_qty=5, tot_sell_qty=6, avg_trade_price=9.5)


# ------------------------------------------------------------------------------ pure analysis
def test_schema_and_classification():
    s = P.schema_of_rows([dict(a=1, b=None), dict(a=2.0, c="x")])
    assert s["a"]["types"] == ["float", "int"] and s["a"]["present_in"] == 2 and s["b"]["types"] == ["null"] and s["c"]["present_in"] == 1 and s["c"]["of"] == 2
    c = P.classify_keys(["bid", "ask", "bid_size", "oi", "oich", "prev_oi", "tt", "exch_feed_time", "volume", "ltp", "bids"])
    assert "tt" in c["timestamp_like"] and "exch_feed_time" in c["timestamp_like"] and "bid_size" in c["size_or_depth_like"] and c["size_or_depth_like"] == ["bid_size"]
    assert P.classify_keys(["bids", "ask", "x"], {"bids": [1], "ask": [2], "x": 1})["nested_value_keys"] == ["bids", "ask"]
    assert set(c["oi_like"]) == {"oi", "oich", "prev_oi"} and set(c["bid_ask_like"]) >= {"bid", "ask", "bid_size", "bids"}


def test_analyze_chain_reports_oi_and_absence_of_timestamp_and_size():
    a = P.analyze_chain(chain_body())
    assert a["n_option_rows"] == 10 and a["n_underlying_rows"] == 1
    assert {"bid", "ask", "oi", "oich", "prev_oi", "volume"} <= set(a["option_row_schema"])
    assert a["option_row_key_classes"]["timestamp_like"] == [] and a["option_row_key_classes"]["size_or_depth_like"] == []
    assert a["expiry_data_keys"] == ["date", "expiry", "expiry_flag"]


def test_analyze_quotes_wanted_missing():
    body = dict(s="ok", d=[dict(n="NSE:A", s="ok", v=dict(lp=1, bid=0.9, ask=1.1, tt=1788603960, ch=0.1))])
    a = P.analyze_quotes(body)["per_symbol"]["NSE:A"]
    assert a["v_keys"] == ["ask", "bid", "ch", "lp", "tt"] and "volume" in a["wanted_missing"] and a["values"]["tt"] == 1788603960


@pytest.mark.parametrize("rec,body,exists", [
    (dict(status=200, body_s="ok"), dict(s="ok", d={"NSE:A": dict(bids=[dict(price=1, volume=2, ord=1)], ask=[dict(price=2, volume=3, ord=1)], oi=5)}), True),
    (dict(status=404, body_s=None), None, False),
    (dict(status=200, body_s="error", body_message="invalid"), dict(s="error", message="invalid"), False),
])
def test_analyze_depth(rec, body, exists):
    a = P.analyze_depth(rec, body)
    assert a["endpoint_returns_data"] is exists
    if exists:
        assert a["bids_level_keys"] == ["ord", "price", "volume"] and a["classes"]["oi_like"] == ["oi"] and a["classes"]["nested_value_keys"] == ["bids", "ask"]


def test_tick_analysis_required_fields_and_lags():
    t0 = 1788603990.0
    ticks = [(t0 + i, tick(feed=1788603960 + i, ltt=1788603900)) for i in range(3)]
    a = P.analyze_symbol_ticks(ticks)
    assert a["all_required_present"] and a["capture_minus_exch_feed_time_s"] == dict(n=3, min=30.0, median=30.0, max=30.0)
    assert a["capture_minus_last_traded_time_s"]["median"] == 90.0 + 1 and a["exch_feed_time_never_decreases"] is True and a["exch_feed_time_units_look_like_epoch_seconds"] is True
    missing = [(t0, {k: v for k, v in tick().items() if k != "bid_size"})]
    assert P.analyze_symbol_ticks(missing)["all_required_present"] is False


def test_tick_pair_classification_distinguishes_quote_only_updates():
    base = tick()
    seq = [base,
           dict(base, bid_size=150, exch_feed_time=base["exch_feed_time"] + 1),                   # quote-only, feed advanced
           dict(base, bid_size=151, exch_feed_time=base["exch_feed_time"] + 1),                   # quote-only, feed unchanged
           dict(base, bid_size=151, ltp=10.1, vol_traded_today=1065, last_traded_time=base["last_traded_time"] + 5, exch_feed_time=base["exch_feed_time"] + 6)]  # trade
    a = P.analyze_symbol_ticks([(100.0 + i, m) for i, m in enumerate(seq)])
    pr = a["consecutive_tick_pairs"]
    assert pr["trade_changed=False|quote_changed=True|exch_feed_time_changed=True"] == 1
    assert pr["trade_changed=False|quote_changed=True|exch_feed_time_changed=False"] == 1
    assert pr["trade_changed=True|quote_changed=False|exch_feed_time_changed=True"] == 1
    assert a["quote_only_updates"] == dict(feed_time_advanced=1, feed_time_unchanged=1, enough_evidence=False)
    assert a["last_traded_time_vs_volume_changes"] == dict(both_changed=1, ltt_only=0, vol_only=0, neither=2)


def test_feed_time_decrease_is_reported():
    a = P.analyze_symbol_ticks([(1.0, tick(feed=1788603970)), (2.0, tick(feed=1788603960))])
    assert a["exch_feed_time_never_decreases"] is False


def test_rest_vs_ws_comparison():
    t = 1788603990.0
    ws = {"NSE:X": [(t - 2, tick(feed=1788603960)), (t + 20, tick(feed=1788604000))]}
    rows = P.compare_rest_with_ws([(t, "NSE:X", dict(tt=1788603960, bid=9.9, ask=10.1, lp=10.0)), (t, "NSE:Y", dict(tt=1)), (t + 300, "NSE:X", dict(tt=5))], ws)
    assert rows[0]["tt_equals_exch_feed_time"] is True and rows[0]["tt_equals_last_traded_time"] is False and rows[0]["bid_ask_equal"] is True and rows[0]["tt_minus_exch_feed_time_s"] == 0
    assert "no websocket ticks" in rows[1]["note"] and "away" in rows[2]["note"]


def test_pick_symbols_and_future_and_market_window():
    p = P.pick_symbols(chain_body())
    assert p["atm"] == 24400 and p["atm_ce"].endswith("24400CE") and p["atm_pe"].endswith("24400PE") and p["otm_ce"].endswith("24550CE")
    assert P.pick_symbols(dict(data=dict(optionsChain=[]))) is None
    master = "\n".join([",".join(["0"] * 8 + ["2000000000", "NSE:NIFTY26OCTFUT"] + ["0"] * 3 + ["NIFTY", "0", "0", "XX"]), ",".join(["0"] * 8 + ["1900000000", "NSE:NIFTY26SEPFUT"] + ["0"] * 3 + ["NIFTY", "0", "0", "XX"]),
                        ",".join(["0"] * 8 + ["1800000000", "NSE:NIFTY26AUGFUT"] + ["0"] * 3 + ["NIFTY", "0", "0", "XX"])])
    assert P.find_future(master, 1850000000) == "NSE:NIFTY26SEPFUT"
    from datetime import datetime
    assert P.market_window_warning(datetime(2026, 10, 3, 11, 0, tzinfo=P.IST)) == "weekend"
    assert P.market_window_warning(datetime(2026, 10, 5, 11, 0, tzinfo=P.IST)) is None
    assert "outside" in P.market_window_warning(datetime(2026, 10, 5, 16, 0, tzinfo=P.IST))


# ------------------------------------------------------------------------------ safety
class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}
        self.content = json.dumps(body).encode() if body is not None else b""
        self.text = json.dumps(body) if body is not None else "<html>no</html>"

    def json(self):
        if self._b is None:
            raise ValueError("not json")
        return self._b


class GetOnlySession:
    """Any attribute other than .get raises: proves the probe can only issue GET requests."""

    def __init__(self, router):
        self.router, self.calls = router, []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, dict(params or {}), headers))
        return self.router(url, params or {})

    def __getattr__(self, name):
        raise AssertionError(f"probe tried to use session.{name}")


def router(url, params):
    if url == P.CHAIN_URL:
        return FakeResp(200, chain_body(), {"Date": "Fri, 02 Oct 2026 10:00:00 GMT", "X-RateLimit-Remaining": "9"})
    if url == P.QUOTES_URL:
        syms = params["symbols"].split(",")
        return FakeResp(200, dict(s="ok", d=[dict(n=s, s="ok", v=dict(lp=10.0, bid=9.9, ask=10.1, tt=1788603960)) for s in syms]))
    if url == P.DEPTH_URL:
        return FakeResp(404, None)
    raise AssertionError(url)


def test_http_is_get_only_never_logs_authorization_and_records_headers():
    s = GetOnlySession(router)
    h = P.Http("APPID-100", TOKEN, session=s, sleep=lambda x: None)
    rec, body = h.get("c", P.CHAIN_URL, {"symbol": "x"})
    assert rec["status"] == 200 and rec["rate_limit_headers"] == {"X-RateLimit-Remaining": "9"} and rec["latency_ms"] >= 0 and "Authorization" not in json.dumps(rec) and TOKEN not in json.dumps(h.log)
    assert s.calls[0][2] == {"Authorization": f"APPID-100:{TOKEN}"}
    rec, body = h.get("d", P.DEPTH_URL, {"symbol": "x"})
    assert rec["status"] == 404 and body is None and "non_json_body_head" in rec


def test_http_stops_after_429():
    h = P.Http("A", TOKEN, session=GetOnlySession(lambda u, p: FakeResp(429, {"s": "error", "code": -429, "message": "rate limit"})), sleep=lambda x: None)
    h.get("a", P.QUOTES_URL)
    r, b = h.get("b", P.QUOTES_URL)
    assert r is None and h.log[0]["rate_limited"] is True and "skipped" in h.log[1]


def test_http_records_exceptions():
    def boom(u, p):
        raise ConnectionError("down")
    h = P.Http("A", TOKEN, session=GetOnlySession(boom), sleep=lambda x: None)
    rec, body = h.get("a", P.QUOTES_URL)
    assert "ConnectionError" in rec["error"] and body is None


def test_safe_write_refuses_token(tmp_path):
    P._SECRETS[:] = [TOKEN]
    try:
        with pytest.raises(RuntimeError):
            P.safe_write(str(tmp_path / "x"), "abc " + TOKEN)
        P.safe_write(str(tmp_path / "y"), "clean")
    finally:
        P._SECRETS.clear()


class FakeSocket:
    def __init__(self, access_token, log_path, litemode, write_to_file, reconnect, on_connect, on_close, on_error, on_message):
        assert litemode is False and reconnect is False and write_to_file is False
        self.cb, self.subscribed, self.closed = (on_connect, on_message), None, False

    def connect(self):
        self.cb[0]()
        for i, s in enumerate(self.subscribed):
            for j in range(4):
                self.cb[1](tick(sym=s, feed=1788603960 + j, bs=100 + j))

    def subscribe(self, symbols, data_type):
        assert data_type == "SymbolUpdate"
        self.subscribed = symbols

    def close_connection(self):
        self.closed = True


def test_run_end_to_end_with_fakes_writes_no_token(tmp_path):
    P._SECRETS[:] = [TOKEN, f"APPID-100:{TOKEN}"]
    try:
        h = P.Http("APPID-100", TOKEN, session=GetOnlySession(router), sleep=lambda x: None)
        out = str(tmp_path / "o")
        s = P.run(out, h, FakeSocket, 0.4, "yes", sleep=time.sleep, master_text=None, app_id="APPID-100", token=TOKEN)
        for f in ("raw_options_chain_v3.json", "raw_quotes_initial.json", "raw_depth.json", "ws_ticks.jsonl", "probe_summary.json"):
            txt = open(os.path.join(out, f), encoding="utf-8").read()
            assert TOKEN not in txt and txt
        assert s["depth"]["endpoint_returns_data"] is False and s["depth"]["http_status"] == 404
        assert s["websocket"]["connected"] and s["websocket"]["n_ticks_total"] == 4 * len(s["quote_and_ws_symbols"])
        assert all(v["all_required_present"] for v in s["websocket"]["per_symbol"].values())
        assert len(s["rest_vs_ws"]) >= 3 and s["existing_app_running_asserted_by_operator"] == "yes"
        assert [c["name"] for c in h.log].count("quotes_during_ws") == 3 and len(h.log) == 6
        assert s["local_minus_server_date_header_s"] is not None
    finally:
        P._SECRETS.clear()


def test_run_stops_cleanly_without_chain(tmp_path):
    h = P.Http("A", TOKEN, session=GetOnlySession(lambda u, p: FakeResp(500, None)), sleep=lambda x: None)
    s = P.run(str(tmp_path), h, FakeSocket, 1, "no", sleep=lambda x: None)
    assert "fatal" in s and os.path.exists(tmp_path / "probe_summary.json")


def test_main_refuses_outside_market_hours(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "market_window_warning", lambda now=None: "weekend")
    assert P.main(["--out", str(tmp_path)]) == 2


def test_probe_source_has_no_write_verbs_or_order_endpoints():
    src = open(_p, encoding="utf-8").read()
    assert not any(w in src for w in ("requests.post", ".post(", ".put(", ".delete(", "place_order", "/orders", "multiorder", "login(", "get_access_token", "get_auth_header"))


def test_rest_tt_string_regression():
    """Live finding 2026-10-05: REST `tt` is a STRING holding a whole-UTC-day date stamp; comparing it with websocket ints used to raise TypeError."""
    assert P.as_epoch("1791158400") == 1791158400 and P.as_epoch(1791158400) == 1791158400 and P.as_epoch("1791158400.0") == 1791158400
    assert P.as_epoch(None) is None and P.as_epoch("") is None and P.as_epoch("abc") is None and P.as_epoch(True) is None
    t = 1791172410.0
    ws = {"NSE:X": [(t, tick(feed=1791172410, ltt=1791172409))]}
    rows = P.compare_rest_with_ws([(t, "NSE:X", dict(tt="1791158400", bid=9.9, ask=10.1, lp=10.0))], ws)
    r = rows[0]
    assert r["rest_tt"] == 1791158400 and r["rest_tt_raw_type"] == "str" and r["rest_tt_is_whole_utc_day"] is True
    assert r["tt_minus_exch_feed_time_s"] == 1791158400 - 1791172410 and r["tt_equals_exch_feed_time"] is False and r["rest_tt_ist"] == "2026-10-05 05:30:00"
    ok = P.compare_rest_with_ws([(t, "NSE:X", dict(tt="1791172410"))], ws)[0]
    assert ok["tt_equals_exch_feed_time"] is True and ok["rest_tt_is_whole_utc_day"] is False
    q = P.analyze_quotes(dict(s="ok", d=[dict(n="NSE:A", s="ok", v=dict(lp=1, tt="1791158400"))]))
    assert q["per_symbol"]["NSE:A"]["tt_type"] == "str"
