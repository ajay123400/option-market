"""Crash-safe 5-minute recorder. READ-ONLY: it reads the app's cached Fyers token, calls GET endpoints and a websocket, and writes only under its own data directory.

    python -m collector.recorder [--data-dir DIR] [--max-cycles N]        (run from the repo root, with the app's Python environment)

Exit codes: 0 normal, 3 cannot import fyers_auth, 4 no cached token by the deadline, 5 data directory unusable, 6 fatal (no universe / unexpected), 7 another recorder is already running.
"""
import argparse
import json
import logging
import logging.handlers
import os
import signal
import statistics
import sys
import threading
import time
from datetime import date, datetime, timedelta

from . import config as C
from .quality import row_flags
from .sources import AuthImportError, RestClient, TokenProvider, WsFeed
from .store import DuplicateCycle, Store
from .universe import atm_strike, future_symbol, grid_instants, in_window, parse_chain, select_expiries, subscription_symbols

log = logging.getLogger("collector")


class Clock:
    """Single time source (aware IST now, epoch wall, interruptible sleep); replaced by a fake in tests."""

    def __init__(self, stop: threading.Event):
        self.stop = stop

    def now(self):
        return datetime.now(C.IST)

    def wall(self):
        return time.time()

    def monotonic(self):
        return time.monotonic()

    def sleep(self, seconds):
        if seconds > 0:
            self.stop.wait(seconds)


class Recorder:
    def __init__(self, cfg: C.Config, directory: str, tokens: TokenProvider, clock, stop: threading.Event, rest_session=None, ws_factory=None, max_cycles=None, ws_spawn=None):
        self.cfg, self.dir, self.tokens, self.clock, self.stop, self.max_cycles = cfg, directory, tokens, clock, stop, max_cycles
        self._rest_session, self._ws_factory, self._ws_spawn = rest_session, ws_factory, ws_spawn
        self.store = self.rest = self.ws = None
        self.expiries, self.expiry_data, self.fut_symbol = [], [], None
        self.cache = {}                         # expiry epoch -> last parsed chain (symbols/strikes for the universe even when a REST call is skipped)
        self.prev_feed, self.skew = {}, 0.0
        self.backoff_n, self.backoff_until = 0, None
        self.last_restart = 0.0
        self.session_start = None
        self.n_done = 0
        self.last_status = None

    # -------------------------------------------------------------------------------------------- helpers
    def _auth(self):
        t = self.tokens.current()
        return f"{t[0]}:{t[1]}" if t else None

    def _health(self, extra=None):
        h = dict(pid=os.getpid(), updated_ist=self.clock.now().isoformat(timespec="seconds"), last_cycle=self.last_status, ws_connected=bool(self.ws and self.ws.connected), ws_ticks=self.ws.ticks_total if self.ws else 0,
                 ws_subscribed=len(self.ws.subscribed) if self.ws else 0, backoff_until=self.backoff_until.isoformat(timespec="seconds") if self.backoff_until else None, skew_s=self.skew)
        h.update(extra or {})
        try:
            tmp = os.path.join(self.dir, "health.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(h, f)
            os.replace(tmp, os.path.join(self.dir, "health.json"))
        except OSError as e:
            log.warning("health file not written: %s", e)

    def _wait_until(self, t: datetime):
        while not self.stop.is_set():
            rem = (t - self.clock.now()).total_seconds()
            if rem <= 0:
                return True
            self.clock.sleep(min(rem, self.cfg.heartbeat_s))
            self._health()
        return False

    # -------------------------------------------------------------------------------------------- REST
    def _fetch_chain(self, exp):
        r = self.rest.get(f"chain_{exp['date']}", C.CHAIN_URL, {"symbol": C.INDEX_SYMBOL, "strikecount": self.cfg.subscribe_strikes, "timestamp": exp["timestamp_param"]})
        return r

    def _bootstrap(self, today):
        """Universe for the day: nearest chain call -> expiry list -> chain per selected expiry -> websocket subscription."""
        r = self.rest.get("chain_nearest", C.CHAIN_URL, {"symbol": C.INDEX_SYMBOL, "strikecount": self.cfg.subscribe_strikes, "timestamp": ""})
        parsed = parse_chain(r.body) if r.ok else None
        if parsed is None:
            log.error("bootstrap chain call failed (status=%s rate_limited=%s auth_failed=%s error=%s)", r.status, r.rate_limited, r.auth_failed, r.error)
            return False
        self.expiry_data = parsed["expiries"]
        self.expiries = select_expiries(self.expiry_data, today, self.cfg)
        self.fut_symbol = future_symbol(self.expiry_data, today)
        for e in self.expiries:
            if e["epoch"] == int(self.expiry_data[0]["expiry"]):
                self.cache[e["epoch"]] = parsed
            else:
                rr = self._fetch_chain(e)
                if rr.ok and parse_chain(rr.body):
                    self.cache[e["epoch"]] = parse_chain(rr.body)
                else:
                    log.warning("bootstrap chain for %s failed (status=%s)", e["date"], rr.status)
        if not self.cache:
            return False
        atm = atm_strike(parsed["spot"], self.cfg.strike_step)
        syms = subscription_symbols(self.cache_by_index(), atm, self.cfg) + [C.INDEX_SYMBOL] + ([self.fut_symbol] if self.fut_symbol else [])
        log.info("universe: expiries=%s atm=%s subscribe=%d symbols fut=%s", [e["date"] for e in self.expiries], atm, len(syms), self.fut_symbol)
        self.ws = WsFeed(self._auth, self._ws_factory, self.clock.wall, self._ws_spawn)
        self.ws.start(syms)
        for _ in range(20):                      # up to ~10 s for the first ticks
            if self.ws.ticks_total > 0 or self.stop.is_set():
                break
            self.clock.sleep(0.5)
        log.info("websocket: connected=%s ticks=%d subscribed=%d", self.ws.connected, self.ws.ticks_total, len(self.ws.subscribed))
        return True

    def cache_by_index(self):
        return {e["index"]: self.cache[e["epoch"]] for e in self.expiries if e["epoch"] in self.cache}

    # -------------------------------------------------------------------------------------------- one cycle
    def cycle(self, T: datetime):
        cfg, cid = self.cfg, T.strftime("%Y-%m-%dT%H:%M")
        t0 = self.clock.wall()
        snap = self.ws.snapshot() if self.ws else {}
        idx_tick = snap.get(C.INDEX_SYMBOL)
        idx_age = None if idx_tick is None else round(t0 - idx_tick[0], 3)
        notes, status, rest_lat, rate_limited, skews, fresh = [], "ok", [], False, [], {}
        if self.ws and (idx_tick is None or idx_age > cfg.ws_stale_index_s):
            status = "partial_ws"
            notes.append(f"index feed age {idx_age}")
            log.warning("cycle %s: websocket looks stale (index tick age %s s, connected=%s)", cid, idx_age, self.ws.connected)
            if t0 - self.last_restart > 120:
                self.last_restart = t0
                try:
                    self.ws.restart()
                    log.warning("websocket feed restarted")
                except Exception as e:
                    log.error("websocket restart failed: %s", e)
        # ---- REST (skipped during a 429 backoff; stops at the first 429)
        calls0 = self.rest.calls
        if self.backoff_until and self.clock.now() < self.backoff_until:
            status = "partial_backoff"
            notes.append(f"REST skipped: backoff until {self.backoff_until.isoformat(timespec='seconds')}")
            log.warning("cycle %s: REST skipped (429 backoff until %s)", cid, self.backoff_until.strftime("%H:%M:%S"))
        else:
            for e in self.expiries:
                r = self._fetch_chain(e)
                rest_lat.append(r.latency_ms or 0)
                if r.skew_s() is not None:
                    skews.append(r.skew_s())
                if r.rate_limited:
                    rate_limited = True
                    delay = min(cfg.backoff_cap_s, cfg.backoff_start_s * (2 ** self.backoff_n))
                    self.backoff_n += 1
                    self.backoff_until = self.clock.now() + timedelta(seconds=delay)
                    status = "partial_429"
                    notes.append(f"HTTP 429 on {e['date']}: skipping the remaining REST calls, backoff {delay:.0f} s")
                    log.warning("cycle %s: HTTP 429 (rate limit) on chain %s; skipping REST for %.0f s", cid, e["date"], delay)
                    break
                p = parse_chain(r.body) if r.ok else None
                if p is None:
                    if status == "ok":
                        status = "partial_auth" if r.auth_failed else "partial_rest"
                    notes.append(f"chain {e['date']} failed: status={r.status} err={r.error or r.message}")
                    log.warning("cycle %s: chain %s failed (status=%s auth_failed=%s error=%s)", cid, e["date"], r.status, r.auth_failed, r.error)
                    continue
                fresh[e["epoch"]] = (p, r.rx_wall)
                self.cache[e["epoch"]] = p
            if fresh and not rate_limited:
                self.backoff_n = 0
                self.backoff_until = None
        if skews:
            self.skew = round(statistics.median(skews), 3)
        # ---- context
        any_fresh = next(iter(fresh.values()))[0] if fresh else None
        spot = any_fresh["spot"] if any_fresh else (idx_tick[1].get("ltp") if idx_tick else None)
        vix = any_fresh["vix"] if any_fresh else None
        fp = any_fresh["fp"] if any_fresh else None
        atm = atm_strike(spot, cfg.strike_step) if spot else None
        rows = []
        if atm is not None:
            for e in self.expiries:
                p = self.cache.get(e["epoch"])
                if not p:
                    continue
                f = fresh.get(e["epoch"])
                for cr in in_window(p["rows"], atm, cfg.record_strikes, cfg.strike_step):
                    rows.append(self._option_row(cid, e, cr, atm, spot, snap.get(cr["symbol"]), cr if f else None, f[1] if f else None, t0))
        rows.append(self._simple_row(cid, "index", C.INDEX_SYMBOL, snap.get(C.INDEX_SYMBOL), spot, t0))
        if self.fut_symbol:
            rows.append(self._simple_row(cid, "future", self.fut_symbol, snap.get(self.fut_symbol), spot, t0))
        for r in rows:
            r["flags"] = row_flags(r, self.skew, cfg, self._prev_feed(r["symbol"]), kind=r["kind"])
            if r.get("quote_feed_ts") is not None:
                self.prev_feed[r["symbol"]] = r["quote_feed_ts"]
        n_ws = sum(1 for r in rows if r["data_source"] == "fyers:ws-full")
        if status == "ok" and rows and n_ws < len(rows) * 0.9:
            status = "partial_ws"
            notes.append(f"only {n_ws}/{len(rows)} rows from the websocket")
        if len(rows) <= 2:
            status = "failed" if status == "ok" else status
        # ---- grow the websocket subscription if the ATM window moved (best effort; REST fallback otherwise)
        if self.ws and atm is not None:
            new = self.ws.add_symbols(subscription_symbols(self.cache_by_index(), atm, cfg))
            if new:
                log.info("cycle %s: subscribed %d more symbols (ATM now %s)", cid, len(new), atm)
        n_opt = sum(1 for r in rows if r["kind"] == "option")
        cycle = dict(cycle_id=cid, scheduled_ts=T.timestamp(), capture_start_ts=t0, capture_end_ts=self.clock.wall(), status=status, reason="; ".join(notes) or None, n_expected=n_opt + 1 + (1 if self.fut_symbol else 0),
                     n_rows=len(rows), n_ws_rows=n_ws, n_chain_rows=sum(1 for r in rows if r["data_source"] == "fyers:options-chain-v3"), spot=spot, atm_strike=atm, india_vix=vix, future_fp=fp, skew_est_s=self.skew,
                     ws_connected=int(bool(self.ws and self.ws.connected)), ws_ticks_total=self.ws.ticks_total if self.ws else 0, ws_subscribed=len(self.ws.subscribed) if self.ws else 0, index_feed_age_s=idx_age,
                     rest_calls=self.rest.calls - calls0, rest_latency_ms_max=max(rest_lat) if rest_lat else None, rate_limited=int(rate_limited), expiries_json=json.dumps([e["date"] for e in self.expiries]), notes=None)
        try:
            self.store.write_cycle(cycle, rows)
        except DuplicateCycle:
            log.warning("cycle %s already recorded; not written again", cid)
            return None
        self.last_status = f"{cid}:{status}"
        log.info("cycle %s %s rows=%d ws=%d chain=%d idx_age=%s skew=%+.2fs rest_calls=%d lat_max=%s%s", cid, status, len(rows), n_ws, cycle["n_chain_rows"], idx_age, self.skew, cycle["rest_calls"],
                 cycle["rest_latency_ms_max"], " 429!" if rate_limited else "")
        return cycle

    def _prev_feed(self, symbol):
        if symbol not in self.prev_feed:
            v = self.store.last_quote_feed_ts(symbol)
            if v is not None:
                self.prev_feed[symbol] = v
        return self.prev_feed.get(symbol)

    def _option_row(self, cid, exp, cr, atm, spot, tick, chain_row, oi_ts, t0):
        r = dict(cycle_id=cid, symbol=cr["symbol"], kind="option", expiry_date=exp["date"], expiry_ts=exp["epoch"], dte_days=exp["dte_days"], strike=cr["strike"], option_type=cr["type"], atm_strike=atm,
                 offset_strikes=int(round((cr["strike"] - atm) / self.cfg.strike_step)), spot=spot, capture_ts=t0)
        if chain_row:
            r.update(oi=chain_row.get("oi"), oi_prev=chain_row.get("prev_oi"), oi_change=chain_row.get("oich"), oi_capture_ts=oi_ts, chain_bid=chain_row.get("bid"), chain_ask=chain_row.get("ask"),
                     chain_ltp=chain_row.get("ltp"), chain_volume=chain_row.get("volume"))
        if tick:
            self._from_tick(r, tick, t0)
        elif chain_row:
            r.update(ltp=chain_row.get("ltp"), bid=chain_row.get("bid"), ask=chain_row.get("ask"), volume=chain_row.get("volume"), data_source="fyers:options-chain-v3")
        else:
            r["data_source"] = "none"
        return r

    def _simple_row(self, cid, kind, symbol, tick, spot, t0):
        r = dict(cycle_id=cid, symbol=symbol, kind=kind, spot=spot, capture_ts=t0)
        if tick:
            self._from_tick(r, tick, t0)
        else:
            r["data_source"] = "none"
        return r

    @staticmethod
    def _from_tick(r, tick, t0):
        rx, m = tick
        feed, ltt = m.get("exch_feed_time"), m.get("last_traded_time")
        r.update(ltp=m.get("ltp"), bid=m.get("bid_price"), ask=m.get("ask_price"), bid_size=m.get("bid_size"), ask_size=m.get("ask_size"), volume=m.get("vol_traded_today"), last_traded_qty=m.get("last_traded_qty"),
                 avg_trade_price=m.get("avg_trade_price"), tot_buy_qty=m.get("tot_buy_qty"), tot_sell_qty=m.get("tot_sell_qty"), quote_feed_ts=feed, last_trade_ts=ltt, rx_ts=rx, data_source="fyers:ws-full",
                 capture_minus_feed_s=None if feed is None else round(t0 - feed, 3), capture_minus_last_trade_s=None if not ltt else round(t0 - ltt, 3))

    # -------------------------------------------------------------------------------------------- single-instance lock
    def _acquire_lock(self) -> bool:
        """recorder.lock holds the PID; a lock whose health.json heartbeat is older than 120 s (real time) is stale and is taken over."""
        lock, health = os.path.join(self.dir, "recorder.lock"), os.path.join(self.dir, "health.json")
        if os.path.exists(lock):
            try:
                fresh = os.path.exists(health) and time.time() - os.path.getmtime(health) < 120
            except OSError:
                fresh = False
            if fresh:
                try:
                    other = open(lock).read().strip()
                except OSError:
                    other = "?"
                log.error("another recorder (pid %s) is running in %s (heartbeat fresh); exiting (exit code 7)", other, self.dir)
                return False
            log.warning("stale recorder.lock found; taking over")
        with open(lock, "w") as f:
            f.write(str(os.getpid()))
        return True

    def _release_lock(self):
        try:
            os.remove(os.path.join(self.dir, "recorder.lock"))
        except OSError:
            pass

    # -------------------------------------------------------------------------------------------- day loop
    def _missed_row(self, T, reason):
        return dict(cycle_id=T.strftime("%Y-%m-%dT%H:%M"), scheduled_ts=T.timestamp(), status="missed", reason=reason, n_rows=0, n_expected=None)

    def _record_missed(self, T, reason):
        try:
            self.store.write_cycle(self._missed_row(T, reason), [])
            log.warning("cycle %s recorded as missed: %s", T.strftime("%H:%M"), reason)
        except DuplicateCycle:
            pass

    def run(self) -> int:
        cfg = self.cfg
        now = self.clock.now()
        today = now.date()
        try:
            os.makedirs(self.dir, exist_ok=True)
            self.store = Store(self.dir, today, cfg.as_dict())
        except OSError as e:
            log.error("data directory %s is not usable: %s", self.dir, e)
            return 5
        if not self._acquire_lock():
            self.store.close()
            return 7
        try:
            return self._run(today, now)
        finally:
            self._release_lock()

    def _run(self, today, now) -> int:
        cfg = self.cfg
        self.session_start = now
        log.info("recorder starting: day=%s dir=%s pid=%d", today, self.dir, os.getpid())
        deadline = datetime.combine(today, cfg.token_deadline, tzinfo=C.IST)
        try:
            tok = self.tokens.wait_for_token(self.clock.now, lambda sec: (self._health({"waiting": "token"}), self.clock.sleep(sec)), deadline, cfg.token_retry_s)
        except AuthImportError as e:
            log.error("%s", e)
            return 3
        if tok is None:
            log.error("NO CACHED FYERS TOKEN by %s: the app's daily login did not happen. The recorder never logs in; exiting (exit code 4).", cfg.token_deadline.strftime("%H:%M"))
            return 4
        self.rest = RestClient(self._auth, self._rest_session, self.clock.sleep, self.clock.monotonic, self.clock.wall, cfg.rest_min_gap_s)
        instants = grid_instants(today, cfg)
        # missed cycles between the last recorded cycle (or this session's start) and now
        last = self.store.last_cycle()
        ref = datetime.fromtimestamp(last["scheduled_ts"], C.IST) if last else self.session_start
        for T in instants:
            if ref < T <= self.clock.now() - timedelta(seconds=cfg.max_cycle_lag_s) and not self.store.cycle_exists(T.strftime("%Y-%m-%dT%H:%M")):
                self._record_missed(T, "recorder was not running")
        todo = [T for T in instants if T > self.clock.now() - timedelta(seconds=cfg.max_cycle_lag_s) and not self.store.cycle_exists(T.strftime("%Y-%m-%dT%H:%M"))]
        if not todo:
            log.info("no grid instants left today; nothing to record")
            self.store.finalize()
            return 0
        if not self._bootstrap(today):
            log.error("could not build the universe (see the lines above); exiting (exit code 6)")
            self.store.finalize()
            return 6
        try:
            for T in todo:
                if self.stop.is_set():
                    break
                if not self._wait_until(T) or self.clock.now().time() >= cfg.hard_exit:
                    break
                lag = (self.clock.now() - T).total_seconds()
                if lag > cfg.max_cycle_lag_s:
                    self._record_missed(T, f"late by {lag:.0f} s")
                    continue
                try:
                    if self.cycle(T) is not None:
                        self.n_done += 1
                except Exception as e:                              # a bad cycle must never kill the day
                    log.exception("cycle %s failed: %s", T.strftime("%H:%M"), e)
                    self._record_missed(T, f"error: {type(e).__name__}: {str(e)[:150]}")
                self._health()
                if self.max_cycles and self.n_done >= self.max_cycles:
                    log.info("max cycles (%d) reached", self.max_cycles)
                    break
        finally:
            if self.ws:
                self.ws.stop()
            man = self.store.finalize()
            log.info("recorder finished: %s", man["counts"])
        return 0


def setup_logging(directory, level="INFO"):
    os.makedirs(os.path.join(directory, "logs"), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    lg = logging.getLogger("collector")
    lg.setLevel(level)
    lg.handlers.clear()
    fh = logging.handlers.RotatingFileHandler(os.path.join(directory, "logs", f"recorder_{datetime.now(C.IST).strftime('%Y%m%d')}.log"), maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        lg.addHandler(h)


def main(argv=None):
    ap = argparse.ArgumentParser(description="NIFTY options bid/ask recorder (read-only)")
    ap.add_argument("--data-dir", default=None, help="default: NIFTY_MICRO_DIR, else E:\\nifty_microstructure on Windows")
    ap.add_argument("--max-cycles", type=int, default=None, help="stop after N recorded cycles (supervised runs)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    directory = a.data_dir or C.data_dir()
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        print(f"data directory {directory} is not usable: {e}")
        return 5
    setup_logging(directory, a.log_level)
    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stop.set())
    rec = Recorder(C.Config(), directory, TokenProvider(), Clock(stop), stop, max_cycles=a.max_cycles)
    try:
        return rec.run()
    except Exception as e:
        log.exception("fatal: %s", e)
        return 6


if __name__ == "__main__":
    sys.exit(main())
