"""Recorder configuration: every constant that shapes the data lives here (and is written into each daily database's `meta` table)."""
import os
from dataclasses import asdict, dataclass
from datetime import time as dtime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
INDEX_SYMBOL = "NSE:NIFTY50-INDEX"
WS_SYMBOL_LIMIT = 5000                      # fyers_apiv3 data_ws.symbol_limit (checked in fyers_apiv3 3.1.18: "Please provide less than 5000 symbols")
CHAIN_URL = "https://api-t1.fyers.in/data/options-chain-v3"
DEFAULT_WINDOWS_DIR = r"E:\nifty_microstructure"


@dataclass(frozen=True)
class Config:
    first_cycle: dtime = dtime(9, 16)       # grid: minute = 1 (mod 5); contains 10:01, 13:01 and 15:01 (bar start + 60 s, the historical observation instants)
    last_cycle: dtime = dtime(15, 36)
    grid_minutes: int = 5
    hard_exit: dtime = dtime(15, 45)
    token_deadline: dtime = dtime(9, 30)    # retry the cached token every token_retry_s until here, then exit with a clear log line
    token_retry_s: float = 60.0
    record_strikes: int = 12                # recorded window: ATM +/- 12 strikes (25 strikes x CE/PE)
    subscribe_strikes: int = 16             # websocket subscription / chain request window: ATM +/- 16 (about 200 symbols, far below the 5000 limit)
    strike_step: int = 50
    max_expiries: int = 3
    max_expiry_days: int = 15               # research population is <= 14 DTE
    rest_min_gap_s: float = 2.0             # conservative spacing: the app already polls the same endpoints
    backoff_start_s: float = 65.0           # on HTTP 429: skip REST for this and following cycles until the backoff expires (doubling, capped)
    backoff_cap_s: float = 300.0
    max_cycle_lag_s: float = 120.0          # a grid instant later than this when we get to it is recorded as `missed`, never back-filled
    stale_quote_s: float = 300.0            # descriptive flag only
    wide_spread_abs: float = 1.0            # descriptive flag only: spread > max(abs, pct * mid)
    wide_spread_pct: float = 0.25
    ws_stale_index_s: float = 30.0          # index tick older than this at the cycle instant: websocket considered unhealthy (logged, feed restarted)
    heartbeat_s: float = 30.0

    def as_dict(self):
        d = asdict(self)
        return {k: (v.strftime("%H:%M") if isinstance(v, dtime) else v) for k, v in d.items()}


def data_dir(env=None, os_name=None):
    """NIFTY_MICRO_DIR if set; else E:\\nifty_microstructure on Windows (the machine this runs on); else ~/nifty_microstructure."""
    env = os.environ if env is None else env
    if env.get("NIFTY_MICRO_DIR"):
        return env["NIFTY_MICRO_DIR"]
    if (os_name or os.name) == "nt":
        return DEFAULT_WINDOWS_DIR
    return os.path.join(os.path.expanduser("~"), "nifty_microstructure")
