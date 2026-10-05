"""Phase A Step 1 recorder (collector package): universe, quality flags, store, sources, recorder behaviour, export and report. Fully offline (fake clock, fake Fyers world)."""
import copy
import hashlib
import json
import logging
import os
import sqlite3
import sys
import threading
from datetime import date, datetime, timedelta

import pytest

from collector import config as C, export_parquet as EP, quality as Q, recorder as R, report as REP, sources as SRC, store as ST, universe as U
from tests.collector_fakes import EXPIRY_DATA, FUT, INDEX, FakeAuth, FakeClock, FakeResp, World, start

CFG = C.Config()


# ================================================================================================ config / universe
def test_data_dir_defaults_and_override():
    assert C.data_dir({"NIFTY_MICRO_DIR": r"D:\x"}, "nt") == r"D:\x"
    assert C.data_dir({}, "nt") == r"E:\nifty_microstructure"
    assert C.data_dir({}, "posix").endswith("nifty_microstructure") and "E:" not in C.data_dir({}, "posix")


def test_grid_instants():
    g = U.grid_instants(date(2026, 10, 5))
    assert len(g) == 77 and g[0].strftime("%H:%M") == "09:16" and g[-1].strftime("%H:%M") == "15:36"
    hm = [x.strftime("%H:%M") for x in g]
    assert all(t in hm for t in ("10:01", "13:01", "15:01")) and all((b - a).total_seconds() == 300 for a, b in zip(g, g[1:])) and all(x.tzinfo is not None and x.minute % 5 == 1 for x in g)


def test_select_expiries_rules():
    e = U.select_expiries(EXPIRY_DATA, date(2026, 10, 5))
    assert [x["date"] for x in e] == ["2026-10-06", "2026-10-13", "2026-10-19"] and [x["dte_days"] for x in e] == [1, 8, 14] and e[0]["epoch"] == 1791281400 and e[0]["timestamp_param"] == "1791281400"
    assert [x["date"] for x in U.select_expiries(EXPIRY_DATA, date(2026, 10, 6))][:2] == ["2026-10-06", "2026-10-13"]            # expiry day itself is d=0
    assert [x["date"] for x in U.select_expiries(EXPIRY_DATA, date(2026, 10, 7))] == ["2026-10-13", "2026-10-19"]
    assert [x["date"] for x in U.select_expiries(EXPIRY_DATA, date(2026, 10, 19))] == ["2026-10-19", "2026-10-27", "2026-11-03"] and U.select_expiries(EXPIRY_DATA, date(2026, 10, 19))[2]["dte_days"] == 15
    assert U.select_expiries(EXPIRY_DATA, date(2026, 12, 1)) == [] and U.select_expiries([], date(2026, 10, 5)) == []


def test_future_symbol():
    assert U.future_symbol(EXPIRY_DATA, date(2026, 10, 5)) == FUT and U.future_symbol(EXPIRY_DATA, date(2026, 10, 27)) == FUT
    assert U.future_symbol(EXPIRY_DATA, date(2026, 10, 28)) is None
    assert U.future_symbol([dict(date="29-12-2026", expiry="1", expiry_flag="M")], date(2026, 12, 1)) == "NSE:NIFTY26DECFUT"
    assert U.future_symbol([dict(date="28-01-2027", expiry="1", expiry_flag="M")], date(2026, 12, 30)) == "NSE:NIFTY27JANFUT"


def test_atm_strike_round_half_up():
    assert U.atm_strike(22524.9) == 22500 and U.atm_strike(22525.0) == 22550 and U.atm_strike(22550.0) == 22550 and U.atm_strike(22501.45) == 22500


def chain_for(w, ts):
    return U.parse_chain(w.chain(dict(timestamp=ts, strikecount=16)).json())


def test_parse_chain_and_windows():
    w = World(FakeClock(start()))
    p = chain_for(w, "1791281400")
    assert p["spot"] == 22501.45 and p["vix"] == 14.5 and p["fp"] == 22611.0 and len(p["rows"]) == 66 and len(p["expiries"]) == 5
    r0 = p["rows"][0]
    assert set(r0) == {"symbol", "strike", "type", "bid", "ask", "ltp", "volume", "oi", "oich", "prev_oi"} and isinstance(r0["strike"], float)
    assert len(U.in_window(p["rows"], 22500, 12)) == 50 and len(U.in_window(p["rows"], 22500, 16)) == 66 and len(U.in_window(p["rows"], 22500, 0)) == 2
    assert U.parse_chain(dict(data=dict(optionsChain=[]))) is None and U.parse_chain(None) is None
    by = {i: chain_for(w, ts) for i, ts in enumerate(["1791281400", "1791886200", "1792404600"])}
    syms = U.subscription_symbols(by, 22500, CFG)
    assert len(syms) == 198 and len(set(syms)) == 198 and len(syms) + 2 < C.WS_SYMBOL_LIMIT


# ================================================================================================ quality flags
def flags(**kw):
    base = dict(bid=99.9, ask=100.1, ltp=100.0, volume=10, oi=5, quote_feed_ts=1000, capture_ts=1002.0, data_source="fyers:ws-full")
    base.update(kw)
    return Q.row_flags(base, kw.pop("skew", 0.0) if False else 0.0, CFG, None, "option")


def test_row_flags_by_hand():
    assert flags() == ""
    assert flags(bid=0) == "NO_BID" and flags(ask=None) == "NO_ASK" and "BAD_PRICE" in flags(bid=-1) and flags(ltp=0) == "ZERO_LTP"
    assert flags(bid=100.2, ask=100.1) == "CROSSED" and flags(bid=100.0, ask=100.0) == "LOCKED"
    assert flags(bid=70.0, ask=130.0) == "WIDE_SPREAD" and flags(bid=90.0, ask=110.0) == "" and flags(bid=0.05, ask=1.2) == "WIDE_SPREAD" and flags(bid=0.05, ask=0.9) == ""
    assert flags(capture_ts=1000 + 301) == "STALE_QUOTE" and flags(capture_ts=1000 + 300) == "" and flags(quote_feed_ts=1010) == "FEED_IN_FUTURE"
    assert flags(oi=None) == "NO_OI" and flags(volume=0) == "NO_VOLUME_TODAY" and flags(volume=None) == "NO_VOLUME_TODAY"
    assert flags(data_source="fyers:options-chain-v3", quote_feed_ts=None) == "AGE_UNKNOWN" and flags(quote_feed_ts=None) == "NO_FEED" and flags(data_source="none", quote_feed_ts=None) == "AGE_UNKNOWN"


def test_row_flags_skew_prev_feed_and_kinds():
    base = dict(bid=99.9, ask=100.1, ltp=100.0, volume=10, oi=5, quote_feed_ts=1000, capture_ts=1000 + 301, data_source="fyers:ws-full")
    assert Q.row_flags(base, 0.0, CFG) == "STALE_QUOTE" and Q.row_flags(base, 2.0, CFG) == "" and Q.row_flags(dict(base, capture_ts=1000 + 302.5), 2.0, CFG) == "STALE_QUOTE"     # age = capture - feed - skew
    assert Q.row_flags(dict(base, capture_ts=1002.0), 0.0, CFG, prev_feed_ts=1001) == "FEED_REGRESSED" and Q.row_flags(dict(base, capture_ts=1002.0), 0.0, CFG, prev_feed_ts=1000) == ""
    idx = dict(ltp=22500.0, quote_feed_ts=1000, capture_ts=1002.0, data_source="fyers:ws-full")
    assert Q.row_flags(idx, 0.0, CFG, None, "index") == ""
    fut = dict(bid=22610.0, ask=22612.0, ltp=22611.0, volume=100, quote_feed_ts=1000, capture_ts=1002.0, data_source="fyers:ws-full")
    assert Q.row_flags(fut, 0.0, CFG, None, "future") == ""            # futures: no NO_OI, spread irrelevant at this size
    assert Q.row_flags(dict(fut, volume=0), 0.0, CFG, None, "future") == "NO_VOLUME_TODAY"


# ================================================================================================ store
def test_store_schema_dedupe_atomicity_immutability(tmp_path):
    s = ST.Store(str(tmp_path / "d"), date(2026, 10, 5), {"x": 1})
    assert s.path.endswith("micro_20261005.sqlite") and s.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    s.write_cycle(dict(cycle_id="2026-10-05T11:01", status="ok", scheduled_ts=10.0, n_rows=2), [dict(symbol="A", bid=1.0, flags=""), dict(symbol="B")])
    assert s.counts() == dict(cycles=1, quotes=2) and s.cycle_exists("2026-10-05T11:01") and not s.cycle_exists("x") and s.cycle_ids() == ["2026-10-05T11:01"]
    assert s.last_cycle() == dict(cycle_id="2026-10-05T11:01", scheduled_ts=10.0, status="ok")
    with pytest.raises(ST.DuplicateCycle):
        s.write_cycle(dict(cycle_id="2026-10-05T11:01", status="ok"), [dict(symbol="Z")])
    assert s.counts() == dict(cycles=1, quotes=2)
    with pytest.raises(Exception):                                         # a bad row aborts the WHOLE cycle (no cycle row, no partial quotes)
        s.write_cycle(dict(cycle_id="2026-10-05T11:06", status="ok"), [dict(symbol="A"), dict(symbol="A")])
    assert s.counts() == dict(cycles=1, quotes=2) and not s.cycle_exists("2026-10-05T11:06")
    for sql in ("UPDATE quotes SET bid=2", "DELETE FROM quotes", "UPDATE cycles SET status='x'", "DELETE FROM cycles", "UPDATE meta SET value='x'", "DELETE FROM meta"):
        with pytest.raises(sqlite3.IntegrityError):
            s.db.execute(sql)
    assert json.loads(s.db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0]) == {"x": 1}
    man = s.finalize()
    assert man["counts"] == dict(cycles=1, quotes=2) and man["cycle_status"] == {"ok": 1} and man["sha256"] == hashlib.sha256(open(s.path, "rb").read()).hexdigest()
    s.close()
    s2 = ST.Store(str(tmp_path / "d"), date(2026, 10, 5))                  # reopening keeps data
    assert s2.counts() == dict(cycles=1, quotes=2) and s2.last_quote_feed_ts("A") is None


def test_store_last_quote_feed_ts(tmp_path):
    s = ST.Store(str(tmp_path), date(2026, 10, 5))
    s.write_cycle(dict(cycle_id="2026-10-05T11:01", status="ok"), [dict(symbol="A", quote_feed_ts=100)])
    s.write_cycle(dict(cycle_id="2026-10-05T11:06", status="ok"), [dict(symbol="A", quote_feed_ts=160), dict(symbol="B")])
    assert s.last_quote_feed_ts("A") == 160 and s.last_quote_feed_ts("B") is None


# ================================================================================================ sources
def test_rest_result_classification():
    r = SRC.RestResult("x", 200, dict(s="ok"))
    assert r.ok and not r.rate_limited and not r.auth_failed
    assert SRC.RestResult("x", 429, None).rate_limited and SRC.RestResult("x", 200, dict(s="error", code=-429, message="x")).rate_limited and SRC.RestResult("x", 200, dict(s="error", message="Rate limit exceeded")).rate_limited
    assert not SRC.RestResult("x", 429, dict(s="ok")).ok and SRC.RestResult("x", 401, None).auth_failed and SRC.RestResult("x", 200, dict(s="error", message="Invalid token")).auth_failed
    assert not SRC.RestResult("x", 500, None).ok and not SRC.RestResult("x", error="boom").ok
    assert SRC.RestResult("x", 200, dict(s="ok"), server_date="Mon, 05 Oct 2026 03:52:52 GMT", rx_wall=1791172374.5).skew_s() == pytest.approx(1791172374.5 - 1791172372)
    assert SRC.RestResult("x", 200, dict(s="ok")).skew_s() is None


class GetOnly:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, params, headers))
        return self.fn(url, params)

    def __getattr__(self, n):
        raise AssertionError(f"REST client used session.{n}")


def test_rest_client_get_only_spacing_and_errors():
    clk = FakeClock(start())
    s = GetOnly(lambda u, p: FakeResp(200, dict(s="ok"), {"Date": "Mon, 05 Oct 2026 05:30:00 GMT"}))
    c = SRC.RestClient(lambda: "APP:TOKEN", s, clk.sleep, clk.monotonic, clk.wall, 2.0)
    a = c.get("a", "https://x/y", dict(p=1))
    t1 = clk.t
    b = c.get("b", "https://x/y")
    assert a.ok and clk.t - t1 == pytest.approx(2.0) and c.calls == 2 and s.calls[0][2] == {"Authorization": "APP:TOKEN"} and "TOKEN" not in repr(vars(a))
    boom = SRC.RestClient(lambda: None, GetOnly(lambda u, p: (_ for _ in ()).throw(ConnectionError("down"))), clk.sleep, clk.monotonic, clk.wall, 0)
    e = boom.get("a", "https://x")
    assert not e.ok and "ConnectionError" in e.error and e.status is None
    nojson = SRC.RestClient(lambda: "a", GetOnly(lambda u, p: FakeResp(502, None)), clk.sleep, clk.monotonic, clk.wall, 0).get("a", "x")
    assert nojson.status == 502 and nojson.body is None and not nojson.ok


def test_token_provider_waits_retries_and_never_logs_in():
    clk = FakeClock(start(9, 10))
    auth = FakeAuth(available_after_calls=3)
    tp = SRC.TokenProvider(auth)
    sleeps = []
    tok = tp.wait_for_token(clk.now, lambda s: (sleeps.append(s), clk.sleep(s)), datetime(2026, 10, 5, 9, 30, tzinfo=C.IST), 60)
    assert tok == ("APP-100", "TOKEN.SECRET") and sleeps == [60, 60, 60] and auth.calls == 4
    never = FakeAuth(available_after_calls=10 ** 9)
    clk2 = FakeClock(start(9, 10))
    sl2 = []
    assert SRC.TokenProvider(never).wait_for_token(clk2.now, lambda s: (sl2.append(s), clk2.sleep(s)), datetime(2026, 10, 5, 9, 30, tzinfo=C.IST), 60) is None
    assert len(sl2) == 20 and clk2.now().strftime("%H:%M") == "09:30"
    clk4 = FakeClock(datetime(2026, 10, 5, 9, 25, 30, tzinfo=C.IST))                  # the last wait is shortened to the deadline
    sl4 = []
    assert SRC.TokenProvider(FakeAuth(available_after_calls=10 ** 9)).wait_for_token(clk4.now, lambda s: (sl4.append(s), clk4.sleep(s)), datetime(2026, 10, 5, 9, 30, tzinfo=C.IST), 60) is None
    assert sl4 == [60, 60, 60, 60, 30] and clk4.now().strftime("%H:%M:%S") == "09:30:00"
    clk3 = FakeClock(start(9, 35))
    n3 = FakeAuth(available_after_calls=10 ** 9)
    assert SRC.TokenProvider(n3).wait_for_token(clk3.now, clk3.sleep, datetime(2026, 10, 5, 9, 30, tzinfo=C.IST), 60) is None and n3.calls == 1       # past the deadline: one attempt only
    with pytest.raises(AssertionError):
        n3.login()


def test_token_provider_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "fyers_auth", None)
    with pytest.raises(SRC.AuthImportError):
        SRC.TokenProvider().current()


def test_ws_feed_subscribe_add_limit_and_snapshot():
    clk = FakeClock(start())
    w = World(clk)
    f = SRC.WsFeed(lambda: "APP:TOK", w.factory(), clk.wall, lambda fn: fn())
    f.start(["NSE:A", "NSE:B"])
    assert f.connected and w.sock.subscribed == ["NSE:A", "NSE:B"] and f.ticks_total >= 2
    snap = f.snapshot()
    assert set(snap) == {"NSE:A", "NSE:B"} and snap["NSE:A"][0] == clk.t and snap["NSE:A"][1]["bid_price"] == 99.9
    assert f.add_symbols(["NSE:A", "NSE:C"]) == ["NSE:C"] and w.sock.sub_calls[-1] == ["NSE:C"] and f.add_symbols(["NSE:A", "NSE:C"]) == []
    f._on_error({"invalid_symbols": ["NSE:C"]})
    assert "NSE:C" not in f.subscribed and f.add_symbols(["NSE:C"]) == []
    big = SRC.WsFeed(lambda: "a:b", w.factory(), clk.wall, lambda fn: fn())
    with pytest.raises(ValueError):
        big.start([f"NSE:S{i}" for i in range(C.WS_SYMBOL_LIMIT)])
    f.subscribed.update(f"NSE:S{i}" for i in range(C.WS_SYMBOL_LIMIT - 2 - len(f.subscribed)))        # 4998 subscribed
    assert f.add_symbols(["NSE:X1", "NSE:X2", "NSE:X3"]) == [] and f.events[-1]["kind"] == "limit"
    f.subscribed.clear()
    f.subscribed.update(f"NSE:S{i}" for i in range(C.WS_SYMBOL_LIMIT - 3))                        # 4997 + 3 == 5000 is refused (conservative: the library says "less than 5000")
    assert f.add_symbols(["NSE:X1", "NSE:X2", "NSE:X3"]) == [] and f.add_symbols(["NSE:X1", "NSE:X2"]) == ["NSE:X1", "NSE:X2"]
    f.stop()
    assert w.sock.closed and not f.connected


# ================================================================================================ recorder
def make(tmp_path, clk=None, world=None, auth=None, cfg=CFG, max_cycles=None, hh=11, mm=0):
    clk = clk or FakeClock(start(hh, mm))
    w = world or World(clk)
    if not clk.hooks:
        clk.hooks.append(w.emit)
    rec = R.Recorder(cfg, str(tmp_path), SRC.TokenProvider(auth or FakeAuth()), clk, clk.stop, rest_session=w.session(), ws_factory=w.factory(), max_cycles=max_cycles, ws_spawn=lambda fn: fn())
    return rec, clk, w


def db(tmp_path, day="20261005"):
    return sqlite3.connect(os.path.join(str(tmp_path), f"micro_{day}.sqlite"))


def test_full_day_run(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="collector")
    rec, clk, w = make(tmp_path, hh=9, mm=10)
    assert rec.run() == 0
    con = db(tmp_path)
    assert con.execute("SELECT status, COUNT(*) FROM cycles GROUP BY status").fetchall() == [("ok", 77)]
    assert con.execute("SELECT DISTINCT n_rows, n_expected FROM cycles").fetchall() == [(152, 152)]
    assert con.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] == 77 * 152 and con.execute("SELECT COUNT(*) FROM (SELECT 1 FROM quotes GROUP BY cycle_id, symbol HAVING COUNT(*)>1)").fetchone()[0] == 0
    assert con.execute("SELECT MIN(cycle_id), MAX(cycle_id) FROM cycles").fetchone() == ("2026-10-05T09:16", "2026-10-05T15:36")
    assert con.execute("SELECT kind, COUNT(*) FROM quotes WHERE cycle_id='2026-10-05T10:01' GROUP BY kind ORDER BY kind").fetchall() == [("future", 1), ("index", 1), ("option", 150)]
    assert con.execute("SELECT MAX(ABS(offset_strikes)) FROM quotes WHERE kind='option'").fetchone()[0] == 12 and con.execute("SELECT COUNT(DISTINCT expiry_date) FROM quotes WHERE kind='option'").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM quotes WHERE kind='option' AND (data_source<>'fyers:ws-full' OR oi IS NULL OR bid_size IS NULL OR quote_feed_ts IS NULL)").fetchone()[0] == 0
    assert os.path.exists(os.path.join(str(tmp_path), "micro_20261005.sqlite.manifest.json")) and json.load(open(tmp_path / "health.json"))["pid"] == os.getpid()
    assert w.sock.closed and clk.now().strftime("%H:%M") >= "15:36"
    assert sorted(os.listdir(tmp_path)) == sorted(f for f in os.listdir(tmp_path) if f.startswith(("micro_", "health")))       # nothing but its own files
    rows_per_call = [c for c in w.chain_calls]
    assert len(rows_per_call) == 3 + 77 * 3 and all(c[1] == 16 for c in rows_per_call)                                      # 3 calls per cycle on top of the 3-call bootstrap
    # restart after the day: nothing duplicated, nothing new
    rec2, clk2, w2 = make(tmp_path, clk=FakeClock(start(16, 0)))
    assert rec2.run() == 0 and db(tmp_path).execute("SELECT COUNT(*) FROM cycles").fetchone()[0] == 77 and w2.rest_calls == 0
    assert "NO CACHED" not in caplog.text


def test_cycle_row_contents(tmp_path):
    rec, clk, w = make(tmp_path, max_cycles=1)
    assert rec.run() == 0
    con = db(tmp_path)
    con.row_factory = sqlite3.Row
    c = con.execute("SELECT * FROM cycles").fetchone()
    assert c["cycle_id"] == "2026-10-05T11:01" and c["status"] == "ok" and c["spot"] == 22501.45 and c["atm_strike"] == 22500 and c["india_vix"] == 14.5 and c["future_fp"] == 22611.0
    assert c["rest_calls"] == 3 and c["ws_connected"] == 1 and c["ws_subscribed"] == 200 and c["index_feed_age_s"] == 0.0 and c["skew_est_s"] == pytest.approx(2.0) and json.loads(c["expiries_json"]) == ["2026-10-06", "2026-10-13", "2026-10-19"]
    assert c["scheduled_ts"] == datetime(2026, 10, 5, 11, 1, tzinfo=C.IST).timestamp() and c["capture_start_ts"] >= c["scheduled_ts"] and c["capture_end_ts"] >= c["capture_start_ts"]
    o = con.execute("SELECT * FROM quotes WHERE symbol='NSE:NIFTY26O0622500CE'").fetchone()
    assert (o["kind"], o["expiry_date"], o["expiry_ts"], o["dte_days"], o["strike"], o["option_type"], o["atm_strike"], o["offset_strikes"]) == ("option", "2026-10-06", 1791281400, 1, 22500.0, "CE", 22500, 0)
    assert (o["bid"], o["ask"], o["bid_size"], o["ask_size"], o["last_traded_qty"], o["tot_buy_qty"], o["avg_trade_price"], o["data_source"]) == (99.9, 100.1, 130, 195, 65, 10, 99.5, "fyers:ws-full")
    assert o["oi"] > 1000000 and o["oi_prev"] == 999000 and o["oi_change"] == 5 and o["oi_capture_ts"] == pytest.approx(clk.t, abs=60) and o["chain_bid"] == 99.9 and o["chain_volume"] == 5000
    assert o["capture_ts"] == c["capture_start_ts"] and o["capture_minus_feed_s"] == pytest.approx(o["capture_ts"] - o["quote_feed_ts"], abs=1e-3) and o["capture_minus_last_trade_s"] == pytest.approx(o["capture_ts"] - o["last_trade_ts"], abs=1e-3)
    assert o["rx_ts"] <= o["capture_ts"] and o["flags"] == "" and o["spot"] == 22501.45
    i = con.execute("SELECT * FROM quotes WHERE kind='index'").fetchone()
    assert i["symbol"] == INDEX and i["ltp"] == 22501.45 and i["bid"] is None and i["flags"] == ""
    f = con.execute("SELECT * FROM quotes WHERE kind='future'").fetchone()
    assert f["symbol"] == FUT and f["bid"] == 99.9 and f["oi"] is None


def test_token_arrives_late_and_never(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="collector")
    rec, clk, w = make(tmp_path, auth=FakeAuth(available_after_calls=3), hh=9, mm=10, max_cycles=2)
    assert rec.run() == 0 and "found after 4 attempts" in caplog.text and db(tmp_path).execute("SELECT COUNT(*) FROM cycles WHERE status='ok'").fetchone()[0] == 2
    p2 = tmp_path / "never"
    rec, clk, w = make(p2, auth=FakeAuth(available_after_calls=10 ** 9), hh=9, mm=10)
    assert rec.run() == 4 and clk.now().strftime("%H:%M") == "09:30" and w.rest_calls == 0 and "NO CACHED FYERS TOKEN by 09:30" in caplog.text and "never logs in" in caplog.text
    p3 = tmp_path / "late"
    rec, clk, w = make(p3, auth=FakeAuth(available_after_calls=10 ** 9), hh=9, mm=45)
    assert rec.run() == 4 and w.rest_calls == 0


def test_auth_import_error_and_unusable_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fyers_auth", None)
    stop = threading.Event()
    rec = R.Recorder(CFG, str(tmp_path), SRC.TokenProvider(), FakeClock(start()), stop)
    assert rec.run() == 3
    f = tmp_path / "afile"
    f.write_text("x")
    assert R.Recorder(CFG, str(f / "sub"), SRC.TokenProvider(FakeAuth()), FakeClock(start()), stop).run() == 5
    assert R.main(["--data-dir", str(f / "sub2")]) == 5
    assert R.main(["--data-dir", str(tmp_path / "ok"), "--max-cycles", "1"]) == 3          # fyers_auth import fails in this environment -> exit 3


def test_restart_midday_records_gap_as_missed_without_duplicates(tmp_path):
    rec, clk, w = make(tmp_path, hh=9, mm=10, max_cycles=4)
    assert rec.run() == 0
    assert db(tmp_path).execute("SELECT MAX(cycle_id) FROM cycles").fetchone()[0] == "2026-10-05T09:31"
    rec2, clk2, w2 = make(tmp_path, clk=FakeClock(start(10, 0)), max_cycles=2)
    assert rec2.run() == 0
    con = db(tmp_path)
    st = con.execute("SELECT cycle_id, status, reason FROM cycles ORDER BY cycle_id").fetchall()
    missed = [x for x in st if x[1] == "missed"]
    assert [m[0][-5:] for m in missed] == ["09:36", "09:41", "09:46", "09:51", "09:56"] and all(m[2] == "recorder was not running" for m in missed)
    assert [x[0][-5:] for x in st if x[1] == "ok"] == ["09:16", "09:21", "09:26", "09:31", "10:01", "10:06"]
    assert con.execute("SELECT COUNT(*) FROM quotes WHERE cycle_id IN (SELECT cycle_id FROM cycles WHERE status='missed')").fetchone()[0] == 0


def test_supervised_start_midday_marks_nothing_before_session_start(tmp_path):
    rec, clk, w = make(tmp_path, max_cycles=2)
    rec.run()
    assert db(tmp_path).execute("SELECT COUNT(*) FROM cycles WHERE status='missed'").fetchone()[0] == 0


def test_rate_limit_skips_rest_backs_off_and_recovers(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="collector")
    cfg = C.Config(backoff_start_s=400.0, backoff_cap_s=900.0)
    clk = FakeClock(start())
    w = World(clk)
    w.rate_limit_on = {7}                      # bootstrap = calls 1-3, cycle 1 = 4-6, cycle 2 starts at call 7
    rec, clk, w = make(tmp_path, clk=clk, world=w, cfg=cfg, max_cycles=5)
    assert rec.run() == 0
    con = db(tmp_path)
    con.row_factory = sqlite3.Row
    cyc = {r["cycle_id"][-5:]: r for r in con.execute("SELECT * FROM cycles ORDER BY cycle_id")}
    assert [cyc[k]["status"] for k in ("11:01", "11:06", "11:11", "11:16", "11:21")] == ["ok", "partial_429", "partial_backoff", "ok", "ok"]
    assert cyc["11:06"]["rest_calls"] == 1 and cyc["11:06"]["rate_limited"] == 1 and cyc["11:11"]["rest_calls"] == 0 and cyc["11:16"]["rest_calls"] == 3 and "backoff" in cyc["11:06"]["reason"]
    n = con.execute("SELECT COUNT(*), SUM(oi IS NULL), SUM(data_source='fyers:ws-full') FROM quotes WHERE cycle_id='2026-10-05T11:06' AND kind='option'").fetchone()
    assert tuple(n) == (150, 150, 150)                                       # ws rows are kept; OI is simply absent and flagged
    assert con.execute("SELECT COUNT(*) FROM quotes WHERE cycle_id='2026-10-05T11:06' AND flags LIKE '%NO_OI%'").fetchone()[0] == 150
    assert "HTTP 429" in caplog.text and "skipping REST" in caplog.text and w.rest_calls == 3 + 3 + 1 + 0 + 3 + 3


def test_rate_limit_backoff_doubles(tmp_path):
    cfg = C.Config(backoff_start_s=100.0)
    clk = FakeClock(start())
    w = World(clk)
    w.rate_limit_on = {4, 5}                   # cycle 1 429; cycle 2 (backoff 100 s over) 429 again -> 200 s
    rec, clk, w = make(tmp_path, clk=clk, world=w, cfg=cfg, max_cycles=2)
    rec.run()
    assert rec.backoff_n == 2 and 195 <= (rec.backoff_until - datetime.fromtimestamp(db(tmp_path).execute("SELECT MAX(capture_start_ts) FROM cycles").fetchone()[0], C.IST)).total_seconds() <= 200    # 100 s, then 200 s
    p3 = tmp_path / "again"
    clk = FakeClock(start())
    w = World(clk)
    w.rate_limit_on = {4, 5}
    rec, clk, w = make(p3, clk=clk, world=w, cfg=cfg, max_cycles=3)
    rec.run()
    assert rec.backoff_n == 0 and rec.backoff_until is None                   # reset after the successful third cycle
    st = [r[0] for r in db(p3).execute("SELECT status FROM cycles ORDER BY cycle_id")]
    assert st == ["partial_429", "partial_429", "ok"]


def test_rest_failure_and_auth_failure_keep_ws_rows(tmp_path):
    for status, expect in ((500, "partial_rest"), (401, "partial_auth")):
        p = tmp_path / str(status)
        clk = FakeClock(start())
        w = World(clk)
        rec, clk, w = make(p, clk=clk, world=w, max_cycles=1)
        rec.rest = None
        orig = R.Recorder._bootstrap
        w.fail_status = None
        rec._bootstrap_done = True
        # bootstrap succeeds normally; then every REST call fails
        calls = {"n": 0}
        real_chain = w.chain

        def chain(params, real_chain=real_chain, calls=calls, status=status, w=w):
            calls["n"] += 1
            if calls["n"] > 3:
                w.fail_status = status
            return real_chain(params)
        w.chain = chain
        rec.run()
        con = db(p)
        s = con.execute("SELECT status, n_rows, n_ws_rows, rest_calls FROM cycles").fetchone()
        assert s[0] == expect and s[1] == 152 and s[2] == 152 and s[3] == 3
        assert con.execute("SELECT COUNT(*) FROM quotes WHERE kind='option' AND oi IS NULL").fetchone()[0] == 150


def test_stale_websocket_is_flagged_and_restarted(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="collector")
    clk = FakeClock(start())
    w = World(clk)
    w.drop_index = True
    rec, clk, w = make(tmp_path, clk=clk, world=w, max_cycles=2)
    rec.run()
    st = db(tmp_path).execute("SELECT status, index_feed_age_s, reason FROM cycles ORDER BY cycle_id").fetchall()
    assert [s[0] for s in st] == ["partial_ws", "partial_ws"] and st[0][1] is None and "index feed age" in st[0][2]
    assert caplog.text.count("websocket feed restarted") == 2 and "looks stale" in caplog.text           # one restart per stale cycle (throttle is 120 s; cycles are 300 s apart)
    assert caplog.text.count("looks stale") == 2


def test_atm_move_extends_subscription_and_uses_chain_for_the_gap(tmp_path):
    clk = FakeClock(start())
    w = World(clk)
    rec, clk, w = make(tmp_path, clk=clk, world=w, max_cycles=3)
    moved = {"done": False}

    def move():
        if clk.now() >= start(11, 3) and not moved["done"]:
            moved["done"] = True
            w.spot = 22800.0
    clk.hooks.append(move)
    rec.run()
    con = db(tmp_path)
    con.row_factory = sqlite3.Row
    assert [c["atm_strike"] for c in con.execute("SELECT atm_strike FROM cycles ORDER BY cycle_id")] == [22500, 22800, 22800]
    cyc = {r["cycle_id"][-5:]: r for r in con.execute("SELECT * FROM cycles")}
    assert cyc["11:06"]["n_chain_rows"] == 12 and cyc["11:11"]["n_chain_rows"] == 0 and cyc["11:01"]["n_chain_rows"] == 0
    assert len(w.sock.sub_calls) == 2 and len(w.sock.sub_calls[1]) == 36 and cyc["11:11"]["ws_subscribed"] == 200 + 36
    gap = con.execute("SELECT strike, flags, data_source FROM quotes WHERE cycle_id='2026-10-05T11:06' AND data_source<>'fyers:ws-full' ORDER BY strike").fetchall()
    assert {g["strike"] for g in gap} == {23350.0, 23400.0} and all(g["flags"].count("AGE_UNKNOWN") == 1 and g["data_source"] == "fyers:options-chain-v3" for g in gap)
    assert con.execute("SELECT MIN(strike), MAX(strike) FROM quotes WHERE cycle_id='2026-10-05T11:06' AND kind='option'").fetchone()[:] == (22200.0, 23400.0)


def test_late_cycle_is_recorded_missed_not_backfilled(tmp_path):
    clk = FakeClock(start())
    w = World(clk)
    rec, clk, w = make(tmp_path, clk=clk, world=w, max_cycles=3)
    jumped = {"n": 0}

    def sleep_through():
        if clk.now() >= start(11, 3) and not jumped["n"]:
            jumped["n"] = 1
            clk.t += 400                                                  # machine suspended: 11:06 is more than 120 s late
    clk.hooks.append(sleep_through)
    rec.run()
    st = db(tmp_path).execute("SELECT cycle_id, status, reason FROM cycles ORDER BY cycle_id").fetchall()
    assert [s[0][-5:] for s in st][:4] == ["11:01", "11:06", "11:11", "11:16"] and st[1][1] == "missed" and st[1][2].startswith("late by ") and [s[1] for s in st if s[0][-5:] in ("11:01", "11:11", "11:16")] == ["ok"] * 3


def test_cycle_exception_never_kills_the_day(tmp_path, monkeypatch):
    rec, clk, w = make(tmp_path, max_cycles=2)
    orig = R.Recorder._option_row
    state = {"n": 0}

    def boom(self, *a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("synthetic failure")
        return orig(self, *a, **k)
    monkeypatch.setattr(R.Recorder, "_option_row", boom)
    assert rec.run() == 0
    st = db(tmp_path).execute("SELECT cycle_id, status, reason FROM cycles ORDER BY cycle_id").fetchall()
    assert st[0][1] == "missed" and "RuntimeError" in st[0][2] and st[1][1] == "ok"


def test_graceful_stop_finishes_cycle_and_finalizes(tmp_path):
    clk = FakeClock(start())
    w = World(clk)
    rec, clk, w = make(tmp_path, clk=clk, world=w)

    def stopper():
        if clk.now() >= start(11, 3):
            clk.stop.set()
    clk.hooks.append(stopper)
    assert rec.run() == 0
    assert db(tmp_path).execute("SELECT COUNT(*) FROM cycles WHERE status='ok'").fetchone()[0] == 1 and w.sock.closed
    assert json.load(open(os.path.join(str(tmp_path), "micro_20261005.sqlite.manifest.json")))["counts"]["cycles"] == 1


def test_bootstrap_failure_exit_code(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="collector")
    clk = FakeClock(start())
    w = World(clk)
    w.fail_status = 500
    rec, clk, w = make(tmp_path, clk=clk, world=w)
    assert rec.run() == 6 and "could not build the universe" in caplog.text and db(tmp_path).execute("SELECT COUNT(*) FROM cycles").fetchone()[0] == 0


# ================================================================================================ export / report / safety
def test_report_and_export_roundtrip(tmp_path):
    rec, clk, w = make(tmp_path, max_cycles=3)
    rec.run()
    dbp = os.path.join(str(tmp_path), "micro_20261005.sqlite")
    before = hashlib.sha256(open(dbp, "rb").read()).hexdigest()
    S = REP.summarize(dbp)
    assert S["cycle_status"] == {"ok": 3} and S["share_ok"] == 1.0 and S["rows_per_ok_cycle"] == [152] * 3 and S["duplicate_keys"] == 0 and S["acceptance"]["rows_match_expected"] and S["acceptance"]["no_rate_limit"]
    assert S["ws"]["subscribed"] == [200, 200, 200] and S["rest"]["calls"] == [3, 3, 3] and S["oi_missing_option_rows"] == 0 and S["atm_spread_rs_median"] == pytest.approx(0.2)
    out = EP.export(dbp, str(tmp_path / "pq"))
    import pandas as pd
    q = pd.read_parquet(tmp_path / "pq" / "quotes_20261005.parquet")
    c = pd.read_parquet(tmp_path / "pq" / "cycles_20261005.parquet")
    assert len(q) == 456 == out["quotes"]["rows"] and len(c) == 3 and q.symbol.nunique() == 152 and set(q.columns) >= {"bid", "ask", "bid_size", "oi", "quote_feed_ts", "capture_ts", "flags"}
    assert hashlib.sha256(open(dbp, "rb").read()).hexdigest() == before                      # export is read-only on the database


def test_collector_source_is_read_only_and_never_logs_in():
    d = os.path.join(os.path.dirname(__file__), "..", "collector")
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".py"):
            src = open(os.path.join(d, fn), encoding="utf-8").read()
            assert not any(w in src for w in (".post(", "requests.post", ".put(", ".delete(", "place_order", "/orders", "multiorder", "get_access_token", "get_auth_header", "fyers_auth.login", "_save_cached_token")), fn


def test_nothing_in_the_app_imports_the_collector():
    root = os.path.join(os.path.dirname(__file__), "..")
    for fn in os.listdir(root):
        if fn.endswith(".py"):
            assert "collector" not in open(os.path.join(root, fn), encoding="utf-8", errors="ignore").read().replace("historical_recorder", ""), fn


# ================================================================================================ single instance, hard exit
def test_lock_refuses_second_recorder_and_takes_over_stale_lock(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="collector")
    (tmp_path / "recorder.lock").write_text("4242")                              # fresh lock (mtime = now), NO health.json yet: still a live recorder
    rec, clk, w = make(tmp_path, max_cycles=1)
    assert rec.run() == 7 and "another recorder (pid 4242) is running" in caplog.text and w.rest_calls == 0 and (tmp_path / "recorder.lock").read_text() == "4242"
    old = os.path.getmtime(tmp_path / "recorder.lock") - 600
    os.utime(tmp_path / "recorder.lock", (old, old))                             # heartbeat 10 minutes old -> stale
    rec, clk, w = make(tmp_path, max_cycles=1)
    assert rec.run() == 0 and "stale recorder.lock" in caplog.text and not (tmp_path / "recorder.lock").exists()


def test_second_start_within_the_first_120_seconds_is_refused(tmp_path, caplog):
    """Regression (found live 2026-10-05): a copy started seconds after the first, before any cycle, used to see no health.json, call the lock stale and overwrite it."""
    caplog.set_level(logging.INFO, logger="collector")
    a, clka, wa = make(tmp_path / "x")
    os.makedirs(tmp_path / "x", exist_ok=True)
    assert a._acquire_lock() is True
    assert (tmp_path / "x" / "recorder.lock").read_text() == str(os.getpid()) and (tmp_path / "x" / "health.json").exists()      # heartbeat written immediately on acquire
    b, clkb, wb = make(tmp_path / "x", max_cycles=1)
    assert b.run() == 7 and f"another recorder (pid {os.getpid()}) is running" in caplog.text and wb.rest_calls == 0
    assert (tmp_path / "x" / "recorder.lock").read_text() == str(os.getpid())                                                   # not overwritten
    assert "taking over" not in caplog.text


def test_lock_is_created_atomically_and_stale_takeover_is_rechecked(tmp_path, monkeypatch):
    rec, clk, w = make(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    opened = []
    real_open = os.open

    def spy(path, flags, *a, **k):
        if str(path).endswith("recorder.lock"):
            opened.append(flags)
        return real_open(path, flags, *a, **k)
    monkeypatch.setattr(os, "open", spy)
    assert rec._acquire_lock() and opened and all(f & os.O_EXCL and f & os.O_CREAT for f in opened)
    monkeypatch.undo()
    (tmp_path / "recorder.lock").write_text("777")
    old = os.path.getmtime(tmp_path / "recorder.lock") - 600
    os.utime(tmp_path / "recorder.lock", (old, old))
    rec2, clk2, w2 = make(tmp_path)
    ages = iter([1000.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])                           # looked stale, but was refreshed by someone else before we removed it
    monkeypatch.setattr(R.Recorder, "_lock_age", lambda self: next(ages))
    assert rec2._acquire_lock() is False and (tmp_path / "recorder.lock").read_text() == "777"


def test_heartbeat_refreshes_the_lock_and_release_only_removes_own_lock(tmp_path):
    rec, clk, w = make(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    assert rec._acquire_lock()
    old = os.path.getmtime(tmp_path / "recorder.lock") - 600
    os.utime(tmp_path / "recorder.lock", (old, old))
    assert rec._lock_age() > 500
    rec._health()
    assert rec._lock_age() < 5
    (tmp_path / "recorder.lock").write_text("999")
    rec._release_lock()
    assert (tmp_path / "recorder.lock").read_text() == "999"                     # another recorder's lock is left alone
    (tmp_path / "recorder.lock").write_text(str(os.getpid()))
    rec._release_lock()
    assert not (tmp_path / "recorder.lock").exists()


def test_lock_released_after_normal_run_and_after_token_exit(tmp_path):
    rec, clk, w = make(tmp_path, max_cycles=1)
    assert rec.run() == 0 and not (tmp_path / "recorder.lock").exists()
    p = tmp_path / "n"
    rec, clk, w = make(p, auth=FakeAuth(available_after_calls=10 ** 9), hh=9, mm=10)
    assert rec.run() == 4 and not (p / "recorder.lock").exists() and json.load(open(p / "health.json"))["waiting"] == "token"


def test_hard_exit_time_stops_the_loop(tmp_path):
    cfg = C.Config(hard_exit=datetime(2000, 1, 1, 11, 3).time())
    rec, clk, w = make(tmp_path, cfg=cfg)
    assert rec.run() == 0 and db(tmp_path).execute("SELECT COUNT(*) FROM cycles WHERE status='ok'").fetchone()[0] == 1
