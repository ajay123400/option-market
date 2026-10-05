"""Mutation check for the collector: applies each deliberate fault to a COPY of the repository and runs tests/test_collector.py (usage, from the repo root: python collector/validation/mutation_check.py WORKDIR)."""
import os
import shutil
import subprocess
import sys

SRC = os.getcwd()
W = sys.argv[1]
if os.path.exists(W):
    shutil.rmtree(W)
os.makedirs(W)
for d in ("collector", "tests", "optionsengine"):
    shutil.copytree(os.path.join(SRC, d), os.path.join(W, d), ignore=shutil.ignore_patterns("__pycache__"))
for f in ("pytest.ini",):
    if os.path.exists(os.path.join(SRC, f)):
        shutil.copy(os.path.join(SRC, f), W)
M = [
    ("universe: last grid instant dropped", "collector/universe.py", "while t <= end:", "while t < end:"),
    ("universe: expiry window off by one", "collector/universe.py", "days < 0 or days > cfg.max_expiry_days", "days < 0 or days >= cfg.max_expiry_days"),
    ("universe: expiry day excluded", "collector/universe.py", "days < 0 or days > cfg.max_expiry_days", "days <= 0 or days > cfg.max_expiry_days"),
    ("universe: future month index", "collector/universe.py", "_MONTHS[d.month - 1]", "_MONTHS[d.month % 12]"),
    ("universe: ATM banker's rounding", "collector/universe.py", "int(math.floor(spot / step + 0.5) * step)", "int(round(spot / step) * step)"),
    ("universe: window exclusive", "collector/universe.py", "abs(r[\"strike\"] - atm) <= lim", "abs(r[\"strike\"] - atm) < lim"),
    ("quality: stale threshold inclusive", "collector/quality.py", "if age > cfg.stale_quote_s:", "if age >= cfg.stale_quote_s:"),
    ("quality: skew sign", "collector/quality.py", "age = cap - feed - (skew_s or 0.0)", "age = cap - feed + (skew_s or 0.0)"),
    ("quality: wide spread uses min", "collector/quality.py", "max(cfg.wide_spread_abs, cfg.wide_spread_pct * mid)", "min(cfg.wide_spread_abs, cfg.wide_spread_pct * mid)"),
    ("quality: crossed/locked swapped", "collector/quality.py", "if bid > ask:\n            f.append(\"CROSSED\")", "if bid >= ask:\n            f.append(\"CROSSED\")"),
    ("store: UPDATE trigger missing", "collector/store.py", "for op in (\"UPDATE\", \"DELETE\"):", "for op in (\"DELETE\",):"),
    ("store: no rollback", "collector/store.py", "self.db.execute(\"ROLLBACK\")", "pass"),
    ("store: duplicate not detected", "collector/store.py", "except sqlite3.IntegrityError as e:\n                raise DuplicateCycle(cycle[\"cycle_id\"]) from e", "except sqlite3.IntegrityError as e:\n                raise"),
    ("sources: REST spacing removed", "collector/sources.py", "wait = self._gap - (self._clock() - self._last)", "wait = 0"),
    ("sources: 429 status ignored", "collector/sources.py", "status == 429 or (isinstance", "status == 4290 or (isinstance"),
    ("sources: token wait ignores deadline remainder", "collector/sources.py", "sleep(min(retry_s, remaining))", "sleep(retry_s)"),
    ("sources: resubscribe everything", "collector/sources.py", "new = sorted(set(symbols) - self.subscribed - self.invalid_symbols)", "new = sorted(set(symbols) - self.invalid_symbols)"),
    ("sources: symbol limit off by one", "collector/sources.py", "if len(self.subscribed) + len(new) >= WS_SYMBOL_LIMIT:", "if len(self.subscribed) + len(new) > WS_SYMBOL_LIMIT:"),
    ("recorder: backoff does not double", "collector/recorder.py", "cfg.backoff_start_s * (2 ** self.backoff_n)", "cfg.backoff_start_s * (1 ** self.backoff_n)"),
    ("recorder: 429 does not stop the cycle's REST calls", "collector/recorder.py", "                    break\n                p = parse_chain", "                    pass\n                p = parse_chain"),
    ("recorder: late cycles are recorded anyway", "collector/recorder.py", "if lag > cfg.max_cycle_lag_s:", "if lag > 10 ** 9:"),
    ("recorder: record window uses the subscribe width", "collector/recorder.py", "in_window(p[\"rows\"], atm, cfg.record_strikes, cfg.strike_step)", "in_window(p[\"rows\"], atm, cfg.subscribe_strikes, cfg.strike_step)"),
    ("recorder: backoff never applied", "collector/recorder.py", "self.clock.now() < self.backoff_until:", "self.clock.now() > self.backoff_until:"),
    ("recorder: stale index threshold inverted", "collector/recorder.py", "idx_age > cfg.ws_stale_index_s", "idx_age < cfg.ws_stale_index_s"),
    ("recorder: feed age sign", "collector/recorder.py", "round(t0 - feed, 3)", "round(feed - t0, 3)"),
    ("recorder: previous OI column", "collector/recorder.py", "oi_prev=chain_row.get(\"prev_oi\")", "oi_prev=chain_row.get(\"oich\")"),
    ("recorder: lock freshness inverted", "collector/recorder.py", "if age is not None and age < self.LOCK_STALE_S:", "if age is not None and age > self.LOCK_STALE_S:"),
    ("recorder: lock not created atomically", "collector/recorder.py", "os.O_CREAT | os.O_EXCL | os.O_WRONLY", "os.O_CREAT | os.O_WRONLY"),
    ("recorder: no immediate heartbeat on acquire", "collector/recorder.py", "            self._health()                                                   # heartbeat immediately", "            pass                                                   # heartbeat immediately"),
    ("recorder: stale takeover not re-checked", "collector/recorder.py", "if again is not None and again >= self.LOCK_STALE_S:", "if True:"),
    ("recorder: heartbeat does not refresh the lock", "collector/recorder.py", "os.utime(self._lock_path(), None)", "pass"),
    ("recorder: release removes foreign lock", "collector/recorder.py", "if open(lock).read().strip() == str(os.getpid()):", "if True:"),
    ("recorder: token exit code", "collector/recorder.py", "            return 4\n        self.rest", "            return 5\n        self.rest"),
    ("recorder: hard exit ignored", "collector/recorder.py", "if not self._wait_until(T) or self.clock.now().time() >= cfg.hard_exit:", "if not self._wait_until(T):"),
    ("recorder: missed cycles never marked", "collector/recorder.py", "if ref < T <= self.clock.now() - timedelta(seconds=cfg.max_cycle_lag_s)", "if False and ref < T <= self.clock.now() - timedelta(seconds=cfg.max_cycle_lag_s)"),
]
res = []
for name, f, a, b in M:
    p = os.path.join(W, f)
    s = open(p).read()
    if a not in s:
        res.append((name, "PATTERN NOT FOUND"))
        continue
    open(p, "w").write(s.replace(a, b, 1))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_collector.py", "-x", "-q", "-p", "no:cacheprovider"], cwd=W, capture_output=True, text=True, timeout=240)
        res.append((name, "KILLED" if r.returncode != 0 else "SURVIVED"))
    except subprocess.TimeoutExpired:
        res.append((name, "KILLED (timeout)"))
    open(p, "w").write(s)
for n, r in res:
    print(f"{r:20s} {n}")
