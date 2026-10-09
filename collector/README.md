# collector — Phase A Step 1: forward bid/ask + microstructure recorder

**Purpose.** Record future NIFTY option market quality (bid, ask, sizes, last trade, volume, OI, exchange and recorder timestamps) so that we can later test whether the historical IV-vs-RV gap survives
realistic bid/ask and execution costs. **Data collection only**: no orders, no trading actions, no strategy maths, no authentication of its own. Nothing in the existing app imports this package.

## Broker: Arrow from 2026-10-09
From 9 Oct 2026 Fyers' Standard plan allows 5,000 data calls a day and 50 websocket symbols, so the app (and this recorder) moved to
Arrow (iRage). With the app's `.env` `DATA_SOURCE=arrow`, `collector/arrow_sources.py` provides the same three adapters:
* login via the app's `arrow_auth` (shared `.arrow_session.json`; it logs in once a day if nobody has yet);
* the websocket is `wss://ds.arrow.trade` "full" mode (bid/ask/sizes/volume/OI/exchange times; 1,024 symbols per account; a second
  connection beside the app's was verified to work), rows labelled `data_source=arrow:ws-full`;
* the per-expiry "chain call" is `/info/option-chain` (strikes + `openingOI` = previous-day OI) + `/info/quotes/full` (<= 100 contracts a
  call), returned in options-chain-v3 shape; fallback rows are labelled `arrow:rest-chain`. Arrow has no REST quote for the index, so
  spot and India VIX come from this recorder's websocket (1-min candle close as fallback). REST spacing 0.25 s (10 req/s limit).
Symbols in the database stay Fyers-style (`NSE:NIFTY26O1322650CE`), so older and newer days line up. Days before 2026-10-09 are Fyers data.

## What it records
* Every 5 minutes at minute ≡ 1 (mod 5), 09:16 … 15:36 IST (77 cycles/day; includes 10:01, 13:01, 15:01 = the historical observation instants).
* Per cycle: the nearest up-to-3 listed expiries within 15 calendar days × ATM ± 12 strikes × CE/PE (150 contracts) + the NIFTY index + the nearest-month future = 152 rows.
* Websocket (full mode, one connection): subscribed to ATM ± 16 (about 200 symbols; the fyers_apiv3 client limit is 5000 per connection). Bid/ask/sizes/volume/last-trade and exchange times come from the websocket
  state captured at the cycle instant. Open interest, India VIX and the futures price come from `options-chain-v3` (3 calls per cycle, `strikecount=16`, one per expiry).
* If the ATM window moves, new strikes are subscribed on the live connection; until their first tick the row falls back to the chain's bid/ask (`data_source=fyers:options-chain-v3`, flagged `AGE_UNKNOWN`, no sizes).
* Both clocks are stored and never mixed: `quote_feed_ts` / `last_trade_ts` are EXCHANGE times (whole seconds); `capture_ts` / `rx_ts` / `oi_capture_ts` are OUR clock. `cycles.skew_est_s` = local clock minus the server
  `Date` header (about +1 to +1.5 s on the development machine; the header has 1 s resolution). The skew is only logged and stored — the recorder never changes the system clock.
* Descriptive quality flags per row (nothing is filtered or altered): NO_BID, NO_ASK, BAD_PRICE, CROSSED, LOCKED, WIDE_SPREAD, STALE_QUOTE, FEED_IN_FUTURE, FEED_REGRESSED, NO_FEED, AGE_UNKNOWN, NO_OI, NO_VOLUME_TODAY, ZERO_LTP.
  Thresholds are in `config.py` and stored in each database's `meta` table.

## Storage
`<data dir>/micro_YYYYMMDD.sqlite` (one per trading day; SQLite WAL, `synchronous=FULL`; a cycle and all its rows are one transaction; `(cycle_id, symbol)` is unique; UPDATE/DELETE are blocked by triggers).
At the end of the day `micro_YYYYMMDD.sqlite.manifest.json` records counts and the file's SHA-256. Also in the data dir: `health.json` (heartbeat), `recorder.lock`, `logs/recorder_YYYYMMDD.log`.

Data dir: `--data-dir`, else `NIFTY_MICRO_DIR`, else `E:\nifty_microstructure` on Windows (`~/nifty_microstructure` elsewhere). Estimated size at a 5-minute grid: about 12k rows/day, roughly 3–4 MB/day in SQLite.

Parquet export (research side only): `python -m collector.export_parquet <db> [--out DIR]`.  Acceptance/health summary: `python -m collector.report <db>`.

## Authentication
The Fyers token comes from the app's daily login (`.fyers_session.json`), read through `fyers_auth._load_cached_token()`. **The recorder never logs in.** If no valid token exists at start it retries every 60 s until
09:30 IST, then exits with exit code 4 and the log line `NO CACHED FYERS TOKEN by 09:30 …`. (Started after 09:30 without a token: one attempt, then exit 4.)

## Rate limits
The app already polls REST continuously, so the recorder is deliberately light: 3 chain calls per 5-minute cycle (+ 3 at start), at least 2 s apart, GET only. On HTTP 429 (or a "rate limit" body) the recorder stops its
remaining REST calls for that cycle, skips REST for following cycles during an exponential backoff (65 s, 130 s, 260 s, cap 300 s) and logs it. Websocket rows are still recorded during that time (they involve no REST call);
the cycle is marked `partial_429` / `partial_backoff` and OI is NULL (flag NO_OI).

## Cycle status
`ok`, `partial_429`, `partial_backoff`, `partial_rest`, `partial_auth`, `partial_ws`, `failed`, `missed` (recorder not running, late by more than 120 s, or an internal error; never back-filled — a live quote cannot be recreated).

## Run
From the repo root, with the app's Python environment (the one that can `import fyers_auth`):
```
python -m collector.recorder                      # normal day (Task Scheduler: start 09:10, stop-after 15:45; the recorder also exits by itself after the last cycle)
python -m collector.recorder --max-cycles 3       # supervised run: stops after 3 recorded cycles
```
Exit codes: 0 normal, 3 cannot import fyers_auth, 4 no cached token by 09:30, 5 data directory unusable, 6 no universe / fatal, 7 another recorder already running (`recorder.lock` is created atomically, holds the PID, and its own mtime is the heartbeat; a lock older than 120 s is stale and is taken over).
SIGINT/SIGTERM/Ctrl+C finish the current cycle, close the websocket and write the manifest.

Windows Task Scheduler: trigger Mon–Fri 09:10; action `python -m collector.recorder` with "Start in" = the repo root; "Stop the task if it runs longer than" 7 hours (a 09:10 start would otherwise be killed at 10:45 by a short limit; 6 h 35 min is the exact 15:45 mark; the recorder exits by itself after its last cycle and at 15:45 at the latest, so 7 h is only a safety net); do NOT configure any time resync (needs admin).
The recorder logs the clock skew every cycle.

## Supervised 3-cycle run (suggested acceptance)
1. Start at about 10:58–11:00: `python -m collector.recorder --max-cycles 3` and watch the log.
2. `python -m collector.report E:\nifty_microstructure\micro_YYYYMMDD.sqlite` — expect: all cycles `ok`, 152 rows per cycle, 0 duplicates, `ws_subscribed` about 200, `rest_calls` 3 per cycle, no 429, plausible `skew_est_s`.
3. Run it again for a minute: it must refuse with exit code 7 while another recorder is running, and a restart after a stop must not duplicate cycles.

## Tests
`tests/test_collector.py` (offline; fake clock, fake Fyers world), `collector/validation/` (mutation checks, an independent recomputation of a recorded database).
