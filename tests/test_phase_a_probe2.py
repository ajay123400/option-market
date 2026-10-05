"""Phase A Step 0b (second probe): offline tests of its universe selection, analyses, safety guarantees and orchestration against fakes (no network, no Fyers)."""
import csv
import gzip
import importlib.util
import json
import os
import time
from datetime import datetime

import pytest

_d = os.path.join(os.path.dirname(__file__), "..", "tools", "phase_a_probe")
_s = importlib.util.spec_from_file_location("probe_step0b", os.path.join(_d, "probe_step0b.py"))
B = importlib.util.module_from_spec(_s)
_s.loader.exec_module(B)
P = B.P
TOKEN = "SECRET.TOKEN.VALUE"
EXPS = [dict(date="06-10-2026", expiry="1791281400", expiry_flag="W"), dict(date="13-10-2026", expiry="1791886200", expiry_flag="W"), dict(date="19-10-2026", expiry="1792404600", expiry_flag="W")]
PREFIX = {"2026-10-06": "26O06", "2026-10-13": "26O13", "2026-10-19": "26O19"}


def master_text(strikes=range(20000, 25050, 50)):
    lines = []
    for ts, iso in ((1791281400, "2026-10-06"), (1791886200, "2026-10-13"), (1792404600, "2026-10-19")):
        for k in strikes:
            for t in ("CE", "PE"):
                p = [""] * 17
                p[8], p[9], p[13], p[15], p[16] = str(ts), f"NSE:NIFTY{PREFIX[iso]}{k}{t}", "NIFTY", str(float(k)), t
                lines.append(",".join(p))
    p = [""] * 17
    p[8], p[9], p[13], p[16] = "2000000000", "NSE:NIFTY26OCTFUT", "NIFTY", "XX"
    lines.append(",".join(p))
    return "\n".join(lines)


def chain_body(iso="2026-10-06", spot=22521.05, oi=1000, strikes=range(21900, 23150, 50)):
    rows = [dict(strike_price=-1, ltp=spot, symbol="NSE:NIFTY50-INDEX", fp=22643.5)]
    for k in strikes:
        for t in ("CE", "PE"):
            rows.append(dict(strike_price=k, option_type=t, symbol=f"NSE:NIFTY{PREFIX[iso]}{k}{t}", ltp=50.0, bid=49.9, ask=50.1, oi=oi, oich=1, prev_oi=oi - 1, volume=100, fyToken="1"))
    return dict(code=200, s="ok", data=dict(optionsChain=rows, expiryData=EXPS, indiavixData=dict(ltp=14.45), callOi=1, putOi=2))


def tick(sym="NSE:X", ltp=10.0, bid=9.9, ask=10.1, bs=100, as_=200, vol=1000, feed=1788603960, ltt=1788603900, qty=65, buy=5, sell=6, atp=9.5):
    return dict(symbol=sym, type="sf", ltp=ltp, bid_price=bid, ask_price=ask, bid_size=bs, ask_size=as_, vol_traded_today=vol, exch_feed_time=feed, last_traded_time=ltt, last_traded_qty=qty,
                tot_buy_qty=buy, tot_sell_qty=sell, avg_trade_price=atp)


# ------------------------------------------------------------------------------ pure helpers
def test_parse_master_and_date_conversion():
    m = B.parse_master(master_text(range(22450, 22600, 50)))
    assert set(m) == {"2026-10-06", "2026-10-13", "2026-10-19"}
    assert m["2026-10-06"][(22500.0, "CE")] == "NSE:NIFTY26O0622500CE" and len(m["2026-10-06"]) == 6     # futures row ignored
    assert B.iso_from_chain_date("06-10-2026") == "2026-10-06"


def test_build_universe_full_and_roles():
    uni, skipped = B.build_universe(22521.05, EXPS, B.parse_master(master_text()))
    assert skipped == [] and len(uni) == len(B.UNIVERSE_SPEC) == 22
    atm0 = [u for u in uni if u["expiry_index"] == 0 and u["role"] == "atm"]
    assert {u["strike"] for u in atm0} == {22500.0} and {u["type"] for u in atm0} == {"CE", "PE"}
    far = {(u["type"], u["offset_points"]) for u in uni if u["role"] == "far_wing"}
    assert far == {("CE", 1000), ("PE", -1000), ("CE", 1500), ("PE", -1500)}
    assert {u["expiry_date"] for u in uni} == {"2026-10-06", "2026-10-13", "2026-10-19"}
    assert len({u["symbol"] for u in uni}) == 22


def test_build_universe_substitutes_nearest_listed_and_skips_far():
    m = B.parse_master(master_text(range(22000, 23000, 100)))            # only hundreds listed
    uni, skipped = B.build_universe(22521.05, EXPS, m)
    atm = next(u for u in uni if u["expiry_index"] == 0 and u["role"] == "atm" and u["type"] == "CE")
    assert atm["strike"] in (22500.0,) and atm["target_strike"] == 22500
    m2 = B.parse_master(master_text(range(22450, 22550, 50)))             # no far wings listed
    uni2, skipped2 = B.build_universe(22521.05, EXPS, m2)
    assert any("> 200 pts" in s["reason"] for s in skipped2) and all(abs(u["offset_points"]) <= 300 for u in uni2)
    uni3, skipped3 = B.build_universe(22521.05, EXPS[:1], B.parse_master(master_text()))
    assert all(u["expiry_index"] == 0 for u in uni3) and any(s["reason"] == "expiry not listed" for s in skipped3)


def test_build_universe_uses_chain_fallback_when_master_missing():
    fb = {"2026-10-06": {(22500.0, "CE"): "NSE:NIFTY26O0622500CE", (22500.0, "PE"): "NSE:NIFTY26O0622500PE"}}
    uni, skipped = B.build_universe(22521.05, EXPS[:1], {}, fallback_syms=fb)
    assert [u["role"] for u in uni] == ["atm", "atm"] and len({u["symbol"] for u in uni}) == 2
    assert any("already in the universe" in s["reason"] for s in skipped)


def test_pair_change_signatures_by_hand():
    a = tick()
    seq = [a,
           dict(a, bid_size=150, exch_feed_time=a["exch_feed_time"] + 1),                                  # quote_size + feed_time
           dict(a, bid_size=150, exch_feed_time=a["exch_feed_time"] + 1),                                  # NOTHING_CHANGED
           dict(a, bid_size=150, exch_feed_time=a["exch_feed_time"] + 2),                                  # feed_time only
           dict(a, bid_size=150, exch_feed_time=a["exch_feed_time"] + 2, ltp=10.1, vol_traded_today=1065, last_traded_qty=65, last_traded_time=a["last_traded_time"] + 3, tot_buy_qty=9, avg_trade_price=9.6)]
    sig = B.pair_change_signatures([(float(i), m) for i, m in enumerate(seq)])
    assert sig == {"quote_size+feed_time": 1, "NOTHING_CHANGED": 1, "feed_time": 1, "ltp+volume+ltt+tot_qty+atp": 1}


def test_symbol_liquidity_row_by_hand():
    t = 1788603990.0
    seq = [tick(bid=9.9, ask=10.1, ltp=10.1, vol=1000, feed=1788603960 + i, ltt=1788603900) for i in range(3)]
    seq.append(tick(bid=0.0, ask=10.1, ltp=9.0, vol=1065, feed=1788603970, ltt=1788603965))
    seq.append(tick(bid=10.2, ask=10.1, ltp=10.0, vol=1065, feed=1788603971, ltt=1788603965))
    r = B.symbol_liquidity_row("NSE:X", dict(expiry_date="2026-10-06", strike=22500.0, type="CE", role="atm", offset_points=0), [(t + i, m) for i, m in enumerate(seq)], t_end=1788604000.0)
    assert r["n_ticks"] == 5 and r["volume_increments"] == 1 and r["contracts_traded"] == 65 and r["zero_bid_ticks"] == 1 and r["crossed_ticks"] == 1 and r["locked_ticks"] == 0
    assert r["ltp_ge_ask"] == 3 and r["ltp_le_bid"] == 1
    assert r["feed_minus_last_trade_max_s"] == 1788603962 - 1788603900 and r["last_trade_age_at_end_s"] == 35.0 and r["never_traded_today"] is False
    assert r["spread_rs_median"] == pytest.approx(0.2) and r["spread_rs_max"] == pytest.approx(0.2) and r["distinct_quote_states"] == 3
    assert r["change_signatures"] == {"feed_time": 2, "ltp+volume+ltt+quote_px+feed_time": 1, "ltp+quote_px+feed_time": 1}
    idx = B.symbol_liquidity_row("NSE:NIFTY50-INDEX", {}, [(t, dict(symbol="NSE:NIFTY50-INDEX", ltp=1.0, exch_feed_time=1))], 0)
    assert idx == dict(symbol="NSE:NIFTY50-INDEX", expiry_date=None, strike=None, type=None, role=None, offset_points=None, n_ticks=1)
    nt = B.symbol_liquidity_row("NSE:Y", {}, [], 0)
    assert nt["n_ticks"] == 0


def test_never_traded_symbol_is_flagged():
    seq = [tick(vol=0, ltt=0, feed=1788603960 + i) for i in range(3)]
    r = B.symbol_liquidity_row("NSE:Z", {}, [(float(i), m) for i, m in enumerate(seq)], 1788604000.0)
    assert r["never_traded_today"] is True and r["feed_minus_last_trade_max_s"] is None and r["last_trade_age_at_end_s"] is None


def test_oi_cadence():
    cad = B.oi_cadence([(1000.0, {"a": 1, "b": 2}), (1060.0, {"a": 1, "b": 3, "c": 9}), (1300.0, {"a": 5, "b": 3})])
    assert cad[0]["n_symbols"] == 2 and cad[0]["n_oi_changed"] == 1 and cad[0]["example_changes"] == {"b": [2, 3]} and cad[0]["seconds"] == 60.0
    assert cad[1]["n_symbols"] == 2 and cad[1]["n_oi_changed"] == 1


def test_chain_vs_ws():
    rows = [dict(symbol="NSE:A", bid=9.9, ask=10.1, ltp=10.0, volume=1010), dict(symbol="NSE:B", bid=1, ask=2, ltp=1, volume=1), dict(symbol="NSE:C", bid=1, ask=2, ltp=1, volume=1)]
    ws = {"NSE:A": [(100.0, tick(sym="NSE:A", vol=1000)), (130.0, tick(sym="NSE:A"))], "NSE:B": [(100.0, tick(sym="NSE:B", bid=1.0, ask=2.5, ltp=1.0, vol=1))]}
    r = B.chain_vs_ws(100.5, rows, ws)
    assert r["n_compared"] == 2 and r["bid_equal"] == 2 and r["ask_equal"] == 1 and r["ltp_equal"] == 2 and r["volume_equal"] == 1
    assert r["rows"][0]["chain_volume_minus_ws"] == 10


def test_schedule_is_sparse_and_ordered():
    s = B.schedule(600)
    assert [x[0] for x in s] == sorted(x[0] for x in s) and all(0 < x[0] < 600 for x in s)
    kinds = [x[1] for x in s]
    assert kinds.count("chain_e0") == 3 and kinds.count("quotes") == 3 and kinds.count("depth") == 1
    assert len(B.DEPTH_ROLES) == 5


def test_market_window_requires_mid_session():
    f = lambda h, m, wd=5: B.market_window_warning_b(600, datetime(2026, 10, wd, h, m, tzinfo=P.IST))
    assert f(11, 0) is None and "mid-session" in f(9, 20) and "mid-session" in f(15, 10) and f(11, 0, wd=3) == "weekend"


# ------------------------------------------------------------------------------ orchestration with fakes
class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}
        self.content = json.dumps(body).encode() if body is not None else b""
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._b is None:
            raise ValueError("no json")
        return self._b


class GetOnlySession:
    def __init__(self, router):
        self.router, self.calls = router, []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, dict(params or {}), headers))
        return self.router(url, params or {})

    def __getattr__(self, name):
        raise AssertionError(f"probe tried to use session.{name}")


def router(url, params):
    if url == P.CHAIN_URL:
        if params.get("timestamp") == "":
            return FakeResp(200, chain_body("2026-10-06"), {"Date": "Mon, 05 Oct 2026 03:52:52 GMT"})
        iso = {"1791886200": "2026-10-13", "1792404600": "2026-10-19"}[params["timestamp"]]
        return FakeResp(200, chain_body(iso))
    if url == P.QUOTES_URL:
        return FakeResp(200, dict(s="ok", d=[dict(n=s, s="ok", v=dict(lp=10.0, bid=9.9, ask=10.1, tt="1791158400")) for s in params["symbols"].split(",")]))
    if url == P.DEPTH_URL:
        return FakeResp(200, dict(s="ok", message="Success", d={params["symbol"]: dict(bids=[dict(price=9.9, volume=100, ord=3)] * 5, ask=[dict(price=10.1, volume=200, ord=2)] * 5, ltt=1788603900, oi=5, v=7)}))
    raise AssertionError(url)


class FakeSocket:
    def __init__(self, access_token, log_path, litemode, write_to_file, reconnect, on_connect, on_close, on_error, on_message):
        assert litemode is False and reconnect is False and write_to_file is False
        self.cb, self.subscribed = (on_connect, on_message), None

    def connect(self):
        self.cb[0]()
        for s in self.subscribed:
            for j in range(4):
                if s == P.INDEX_SYMBOL:
                    self.cb[1](dict(symbol=s, type="if", ltp=22500.0 + j, exch_feed_time=1788603960 + j))
                else:
                    self.cb[1](tick(sym=s, feed=1788603960 + j, bs=100 + j, vol=1000 + 65 * (j % 2)))

    def subscribe(self, symbols, data_type):
        assert data_type == "SymbolUpdate"
        self.subscribed = symbols

    def close_connection(self):
        pass


def test_run_end_to_end_with_fakes(tmp_path):
    P._SECRETS[:] = [TOKEN, f"APPID-100:{TOKEN}"]
    try:
        sess = GetOnlySession(router)
        h = P.Http("APPID-100", TOKEN, session=sess, sleep=lambda x: None, min_gap=0)
        out = str(tmp_path / "o")
        S = B.run(out, h, FakeSocket, 1.5, sleep=time.sleep, master_text=master_text(), app_id="APPID-100", token=TOKEN)
        for f in ("probe2_summary.json", "probe2_symbol_table.csv", "ws_ticks.jsonl.gz", "raw_chain_e0_sc12_pre.json", "raw_chain_e1_sc12_pre.json"):
            assert os.path.getsize(os.path.join(out, f)) > 0
        blob = open(os.path.join(out, "probe2_summary.json"), encoding="utf-8").read() + gzip.open(os.path.join(out, "ws_ticks.jsonl.gz"), "rt").read() + open(os.path.join(out, "probe2_symbol_table.csv")).read()
        assert TOKEN not in blob
        assert S["strikecount_12_check"]["honoured"] is True and S["strikecount_12_check"]["n_option_rows"] == 50 and S["strikecount_12_check"]["india_vix"] == 14.45 and S["strikecount_12_check"]["underlying_fp"] == 22643.5
        assert S["second_expiry_chain_check"]["differs_from_first_expiry_symbols"] is True and S["second_expiry_chain_check"]["n_option_rows"] == 50
        assert S["n_ws_symbols"] == 22 + 2 and S["universe_skipped"] == [] and S["websocket"]["connected"] and S["websocket"]["symbols_with_no_ticks"] == []
        assert S["websocket"]["n_ticks_total"] == 4 * 24
        names = [c["name"] for c in h.log]
        assert S["n_rest_calls"] == len(names) == 13 and names.count("depth") == 5 and sum(n.startswith("chain_e0_sc12_during") for n in names) == 3 and sum(n.startswith("quotes_during") for n in names) == 3
        assert all(c[0] in (P.CHAIN_URL, P.QUOTES_URL, P.DEPTH_URL) for c in sess.calls) and S["rate_limited"] is False
        assert len(S["chain_oi_cadence_e0"]) == 3 and all(c["n_oi_changed"] == 0 for c in S["chain_oi_cadence_e0"])
        assert len(S["chain_vs_ws_bid_ask"]) == 3 and S["quotes_tt_values"] == ["1791158400"] and all(r.get("rest_tt_is_whole_utc_day") for r in S["rest_vs_ws"] if "rest_tt" in r)
        assert len(S["depth"]["calls"]) == 5 and S["depth"]["calls"][0]["levels_bid"] == 5 and S["tick_trigger_signatures_all_option_and_future_symbols"]
        rows = list(csv.DictReader(open(os.path.join(out, "probe2_symbol_table.csv"), encoding="utf-8")))
        assert len(rows) == 24 and {"spread_rs_median", "never_traded_today", "feed_minus_last_trade_max_s"} <= set(rows[0])
    finally:
        P._SECRETS.clear()


def test_http_min_gap_is_enforced():
    slept = []
    h = P.Http("A", TOKEN, session=GetOnlySession(lambda u, p: FakeResp(200, dict(s="ok"))), sleep=slept.append, min_gap=2.0)
    h.get("a", P.QUOTES_URL)
    h.get("b", P.QUOTES_URL)
    assert len(slept) == 1 and 1.5 < slept[0] <= 2.0


def test_run_stops_without_chain(tmp_path):
    h = P.Http("A", TOKEN, session=GetOnlySession(lambda u, p: FakeResp(500, None)), sleep=lambda x: None, min_gap=0)
    S = B.run(str(tmp_path), h, FakeSocket, 1, sleep=lambda x: None)
    assert "fatal" in S and os.path.exists(tmp_path / "probe2_summary.json")


def test_main_refuses_outside_window(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "market_window_warning_b", lambda ws, now=None: "weekend")
    assert B.main(["--out", str(tmp_path)]) == 2


def test_source_is_read_only():
    src = open(os.path.join(_d, "probe_step0b.py"), encoding="utf-8").read()
    assert not any(w in src for w in ("requests.post", ".post(", ".put(", ".delete(", "place_order", "/orders", "multiorder", "login(", "get_access_token", "get_auth_header", "sqlite", "to_parquet"))
