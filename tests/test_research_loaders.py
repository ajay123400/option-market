"""Loader behaviour on tiny synthetic parquet files (read-only, look-ahead-free, IST-aware)."""
import hashlib
import math
from datetime import date

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from optionsengine.research import loaders  # noqa: E402

DAY = date(2026, 9, 21)


def write_options(path, rows):
    df = pd.DataFrame(rows, columns=["symbol", "type", "strike", "ts", "open", "high", "low", "close", "volume", "oi"])
    df = df.astype({"strike": "int32", "ts": "int64", "open": "float32", "high": "float32", "low": "float32",
                    "close": "float32", "volume": "int64", "oi": "int64"})
    df.to_parquet(path)
    return path


def bar(sym, typ, k, ts, close, vol):
    return (sym, typ, k, ts, close, close, close, close, vol, 100)


T10 = loaders.snapshot_ts(DAY, "10:00")


def test_snapshot_ts_is_ist_bar_start():
    # 10:00 IST == 04:30 UTC
    assert T10 % 86400 == (4 * 3600 + 30 * 60)
    assert loaders.snapshot_ts(DAY, "15:00") - T10 == 5 * 3600


def test_last_trade_age_forward_fill_and_rounding(tmp_path):
    rows = []
    for m in range(-40, 1):                                  # 09:20 .. 10:00 bar starts
        ts = T10 + 60 * m
        rows.append(bar("A", "CE", 24500, ts, 123.456789, 5 if m in (-30, 0) else 0))     # trades at 09:30 and 10:00
        rows.append(bar("B", "PE", 24400, ts, 50.05, 5 if m == -30 else 0))                # last trade 09:30
        rows.append(bar("C", "PE", 24300, ts, 20.1, 0))                                      # never trades
    df = loaders.load_expiry_frame(write_options(tmp_path / "e.parquet", rows))
    q = {(x.strike): x for x in loaders.quotes_at(df, T10)}
    assert q[24500.0].price == 123.46 and q[24500.0].age_minutes == pytest.approx(1.0)       # traded in this very bar
    assert q[24400.0].age_minutes == pytest.approx(31.0)                                      # 30 bars + this one
    assert q[24300.0].age_minutes is None                                                    # never traded -> missing, not 0
    assert q[24500.0].kind.value == "call" and q[24400.0].kind.value == "put"


def test_no_look_ahead_future_bars_do_not_change_quotes(tmp_path):
    base = [bar("A", "CE", 24500, T10 + 60 * m, 100.0, 5 if m == -10 else 0) for m in range(-20, 1)]
    future_a = base + [bar("A", "CE", 24500, T10 + 60 * m, 999.0, 7) for m in range(1, 30)]
    pa = write_options(tmp_path / "a.parquet", base)
    pb = write_options(tmp_path / "b.parquet", future_a)
    qa = loaders.quotes_at(loaders.load_expiry_frame(pa), T10)
    qb = loaders.quotes_at(loaders.load_expiry_frame(pb), T10)
    assert qa == qb and qa[0].price == 100.0 and qa[0].age_minutes == pytest.approx(11.0)


def test_exact_minute_only_and_missing_snapshot(tmp_path):
    rows = [bar("A", "CE", 24500, T10 - 60, 100.0, 5)]          # no bar AT 10:00
    df = loaders.load_expiry_frame(write_options(tmp_path / "e.parquet", rows))
    assert loaders.quotes_at(df, T10) == []                     # no silent substitution of an earlier bar


def test_trading_days_use_ist_date(tmp_path):
    late = loaders.snapshot_ts(date(2026, 9, 21), "23:45")      # still 21 Sep in IST (18:15 UTC)
    early = loaders.snapshot_ts(date(2026, 9, 22), "00:10")     # 22 Sep IST although 21 Sep 18:40 UTC
    df = loaders.load_expiry_frame(write_options(tmp_path / "e.parquet", [bar("A", "CE", 24500, late, 1.0, 1),
                                                                         bar("A", "CE", 24500, early, 1.0, 1)]))
    assert loaders.trading_days(df) == [date(2026, 9, 21), date(2026, 9, 22)]


def test_loaders_never_write_to_source(tmp_path):
    p = write_options(tmp_path / "e.parquet", [bar("A", "CE", 24500, T10, 100.0, 5)])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    loaders.quotes_at(loaders.load_expiry_frame(p), T10)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h


def test_spot_closes_lookup(tmp_path):
    sp = pd.DataFrame({"ts": [T10, T10 + 60], "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [24500.5, 24501.0]})
    p = tmp_path / "spot.parquet"; sp.to_parquet(p)
    s = loaders.load_spot_closes(str(p))
    assert s[T10] == 24500.5 and (T10 + 120) not in s
