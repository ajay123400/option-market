"""Append-only SQLite store: one database file per trading day (stdlib only, WAL, synchronous=FULL). A cycle and all its rows are written in ONE transaction; (cycle_id, symbol) is unique; UPDATE and DELETE are
blocked by triggers, so recorded data is immutable. Daily Parquet export is a separate offline script (collector/export_parquet.py)."""
import hashlib
import json
import os
import sqlite3
from datetime import date

from . import SCHEMA_VERSION

QUOTE_COLS = [
    ("cycle_id", "TEXT NOT NULL"), ("symbol", "TEXT NOT NULL"), ("kind", "TEXT"), ("expiry_date", "TEXT"), ("expiry_ts", "INTEGER"), ("dte_days", "INTEGER"), ("strike", "REAL"), ("option_type", "TEXT"),
    ("atm_strike", "INTEGER"), ("offset_strikes", "INTEGER"),
    ("ltp", "REAL"), ("bid", "REAL"), ("ask", "REAL"), ("bid_size", "INTEGER"), ("ask_size", "INTEGER"), ("volume", "INTEGER"), ("last_traded_qty", "INTEGER"), ("avg_trade_price", "REAL"),
    ("tot_buy_qty", "INTEGER"), ("tot_sell_qty", "INTEGER"),
    ("oi", "INTEGER"), ("oi_prev", "INTEGER"), ("oi_change", "INTEGER"), ("oi_capture_ts", "REAL"),
    ("quote_feed_ts", "INTEGER"), ("last_trade_ts", "INTEGER"), ("rx_ts", "REAL"), ("capture_ts", "REAL"), ("capture_minus_feed_s", "REAL"), ("capture_minus_last_trade_s", "REAL"),
    ("chain_bid", "REAL"), ("chain_ask", "REAL"), ("chain_ltp", "REAL"), ("chain_volume", "INTEGER"),
    ("spot", "REAL"), ("data_source", "TEXT"), ("flags", "TEXT"), ("extra_json", "TEXT"),
]
CYCLE_COLS = [
    ("cycle_id", "TEXT PRIMARY KEY"), ("scheduled_ts", "REAL"), ("capture_start_ts", "REAL"), ("capture_end_ts", "REAL"), ("status", "TEXT NOT NULL"), ("reason", "TEXT"),
    ("n_expected", "INTEGER"), ("n_rows", "INTEGER"), ("n_ws_rows", "INTEGER"), ("n_chain_rows", "INTEGER"), ("spot", "REAL"), ("atm_strike", "INTEGER"), ("india_vix", "REAL"), ("future_fp", "REAL"),
    ("skew_est_s", "REAL"), ("ws_connected", "INTEGER"), ("ws_ticks_total", "INTEGER"), ("ws_subscribed", "INTEGER"), ("index_feed_age_s", "REAL"),
    ("rest_calls", "INTEGER"), ("rest_latency_ms_max", "REAL"), ("rate_limited", "INTEGER"), ("expiries_json", "TEXT"), ("notes", "TEXT"),
]
IMMUTABLE = ["quotes", "cycles", "meta"]


class DuplicateCycle(Exception):
    pass


class Store:
    def __init__(self, directory: str, day: date, config_dict: dict = None):
        os.makedirs(directory, exist_ok=True)
        self.day = day
        self.path = os.path.join(directory, f"micro_{day.strftime('%Y%m%d')}.sqlite")
        self.db = sqlite3.connect(self.path, isolation_level=None)         # explicit BEGIN/COMMIT
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self._init(config_dict or {})

    def _init(self, config_dict):
        q = ", ".join(f"{n} {t}" for n, t in QUOTE_COLS)
        c = ", ".join(f"{n} {t}" for n, t in CYCLE_COLS)
        self.db.execute(f"CREATE TABLE IF NOT EXISTS quotes ({q}, PRIMARY KEY (cycle_id, symbol))")
        self.db.execute(f"CREATE TABLE IF NOT EXISTS cycles ({c})")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        for t in IMMUTABLE:
            for op in ("UPDATE", "DELETE"):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} BEGIN SELECT RAISE(ABORT, 'recorded data is immutable'); END")
        self.db.execute("CREATE INDEX IF NOT EXISTS quotes_symbol ON quotes(symbol, cycle_id)")
        for k, v in (("schema_version", str(SCHEMA_VERSION)), ("day", self.day.isoformat()), ("config", json.dumps(config_dict, sort_keys=True))):
            self.db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (k, v))

    def cycle_exists(self, cycle_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM cycles WHERE cycle_id=?", (cycle_id,)).fetchone() is not None

    def cycle_ids(self):
        return [r[0] for r in self.db.execute("SELECT cycle_id FROM cycles ORDER BY cycle_id")]

    def last_cycle(self):
        r = self.db.execute("SELECT cycle_id, scheduled_ts, status FROM cycles ORDER BY scheduled_ts DESC LIMIT 1").fetchone()
        return None if r is None else dict(cycle_id=r[0], scheduled_ts=r[1], status=r[2])

    def last_quote_feed_ts(self, symbol):
        r = self.db.execute("SELECT quote_feed_ts FROM quotes WHERE symbol=? AND quote_feed_ts IS NOT NULL ORDER BY cycle_id DESC LIMIT 1", (symbol,)).fetchone()
        return None if r is None else r[0]

    def write_cycle(self, cycle: dict, rows: list):
        """Atomic: the cycle row and every quote row, or nothing. Raises DuplicateCycle if the cycle was already recorded."""
        cc = [n for n, _ in CYCLE_COLS]
        qc = [n for n, _ in QUOTE_COLS]
        self.db.execute("BEGIN IMMEDIATE")
        try:
            try:
                self.db.execute(f"INSERT INTO cycles ({','.join(cc)}) VALUES ({','.join('?' * len(cc))})", [cycle.get(k) for k in cc])
            except sqlite3.IntegrityError as e:
                raise DuplicateCycle(cycle["cycle_id"]) from e
            self.db.executemany(f"INSERT INTO quotes ({','.join(qc)}) VALUES ({','.join('?' * len(qc))})", [[dict(r, cycle_id=cycle["cycle_id"]).get(k) for k in qc] for r in rows])
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def counts(self):
        return dict(cycles=self.db.execute("SELECT COUNT(*) FROM cycles").fetchone()[0], quotes=self.db.execute("SELECT COUNT(*) FROM quotes").fetchone()[0])

    def finalize(self):
        """Checkpoint the WAL and write <db>.manifest.json (row counts, status counts, SHA-256 of the database file)."""
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        st = dict(self.db.execute("SELECT status, COUNT(*) FROM cycles GROUP BY status").fetchall())
        with open(self.path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        man = dict(db=os.path.basename(self.path), day=self.day.isoformat(), schema_version=SCHEMA_VERSION, counts=self.counts(), cycle_status=st, sha256=sha)
        tmp = self.path + ".manifest.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1)
        os.replace(tmp, self.path + ".manifest.json")
        return man

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass
