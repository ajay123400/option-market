"""
Highest-Volume-Strike Cross Strangle Backtest
================================================================

Thin day-loop driver around strategy_engine.StrategyEngine -- the actual
strategy rules (leadership smoothing, overshoot fallback, ITM-neighbor
fallback, hedge, roll-only-on-genuine-leader-change, EOD square-off) all
live there now, in ONE place shared with simulator.py and paper_trade.py.
(This file used to carry its own separate copy of that logic, which
silently drifted out of sync with every later refinement -- confirmed live
when this file's results didn't match the simulator's for the same day.
Never duplicate the engine again; always import it.)

DATA LIMITATION: Kite's/Fyers' historical API only serves currently-listed
(not yet expired) contracts, so this only backtests an expiry's life so far
from whenever it was fetched, and only days where that contract was
genuinely liquid (see instruments_wk0908.json's fetch notes).
"""
import json
import os
from datetime import time as dtime

import sys

import pandas as pd

import paths
from strategy_engine import StrategyEngine

BASE = paths.BASE_DIR
SQUARE_OFF_TIME = dtime(15, 25)  # close out 15 min before NSE's actual F&O close (15:40 since 2026-08-03)


def load_instruments(instruments_file=None):
    """instruments_file defaults to the most recent expiry in expiries.json
    (was hardcoded to instruments_wk0908.json). Pass another one as the
    first CLI argument: python backtest.py instruments_wk0929.json"""
    if instruments_file is None:
        with open(f"{BASE}/expiries.json") as f:
            instruments_file = json.load(f)[-1]["instruments_file"]
    with open(f"{BASE}/{instruments_file}") as f:
        return json.load(f)


def load_all_candles(instruments):
    data = {}
    for inst in instruments:
        path = f"{BASE}/data/{inst['symbol']}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            continue
        data[inst["symbol"]] = df.sort_values("date").reset_index(drop=True)
    return data


def build_day_index(candle_data):
    days = set()
    for df in candle_data.values():
        days.update(df["date"].dt.date.unique())
    return sorted(days)


def run_day(day, frames, strike_of, side_of, ce_strikes, pe_strikes, trade_log):
    all_ts = sorted(ts for ts in set(ts for d in frames.values() for ts in d.index) if ts.time() < SQUARE_OFF_TIME)
    if len(all_ts) < 2:
        return None

    engine = StrategyEngine(ce_strikes, pe_strikes)

    def log_events(events, ts):
        for e in events:
            if e.type.startswith(("ENTRY", "ROLL_EXIT", "EOD_EXIT")):
                trade_log.append([day, e.type, e.strike, ts, e.price, e.hedge_strike, e.hedge_price, e.pnl])

    for ts in all_ts:
        ce_candles, pe_candles = {}, {}
        for sym, d in frames.items():
            if ts not in d.index:
                continue
            row = d.loc[ts]
            rec = {"open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"],
                   "volume": row["volume"], "oi": row["oi"]}
            (ce_candles if side_of[sym] == "CE" else pe_candles)[strike_of[sym]] = rec
        log_events(engine.on_candle_close(ce_candles, pe_candles), ts)

    log_events(engine.force_eod_exit(), all_ts[-1])

    status = engine.status_snapshot()
    return {
        "date": day,
        "ce_leader_open": status["ce_leader"],
        "pe_leader_open": status["pe_leader"],
        "num_candles": len(all_ts),
        "num_rolls": status["num_rolls"],
        "total_pnl": status["day_pnl"],
    }


def run_backtest():
    instruments = load_instruments(sys.argv[1] if len(sys.argv) > 1 else None)
    strike_of = {i["symbol"]: i["strike"] for i in instruments}
    side_of = {i["symbol"]: i["type"] for i in instruments}
    ce_strikes = sorted({i["strike"] for i in instruments if i["type"] == "CE"})
    pe_strikes = sorted({i["strike"] for i in instruments if i["type"] == "PE"})

    candle_data = load_all_candles(instruments)
    if not candle_data:
        print("No data found in E:/Option Market/data/ -- run the data fetch first.")
        return

    days = build_day_index(candle_data)
    print(f"Loaded {len(candle_data)}/{len(instruments)} strikes with data across {len(days)} candidate day(s).")

    daily_rows = []
    trade_log = []

    for day in days:
        day_frames = {}
        for sym, df in candle_data.items():
            d = df[df["date"].dt.date == day]
            if not d.empty:
                day_frames[sym] = d.set_index("date")
        if not day_frames:
            continue

        row = run_day(day, day_frames, strike_of, side_of, ce_strikes, pe_strikes, trade_log)
        if row:
            daily_rows.append(row)

    os.makedirs(f"{BASE}/results", exist_ok=True)
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(f"{BASE}/results/daily_pnl.csv", index=False)

    trade_df = pd.DataFrame(trade_log, columns=["date", "action", "strike", "time", "price",
                                                 "hedge_strike", "hedge_price", "pnl"])
    trade_df.to_csv(f"{BASE}/results/trade_log.csv", index=False)

    if daily_df.empty:
        print("No tradable days found (insufficient data).")
        return

    total_pnl = daily_df["total_pnl"].sum()
    win_days = (daily_df["total_pnl"] > 0).sum()
    total_days = len(daily_df)
    cum = daily_df["total_pnl"].cumsum()
    running_max = cum.cummax()
    drawdown = (cum - running_max).min()

    print("\n=== BACKTEST SUMMARY (via shared strategy_engine) ===")
    print(f"Trading days tested : {total_days}")
    print(f"Total P&L           : Rs {total_pnl:,.2f}")
    print(f"Win days            : {win_days}/{total_days} ({100*win_days/total_days:.1f}%)")
    print(f"Avg daily P&L       : Rs {total_pnl/total_days:,.2f}")
    print(f"Best day            : Rs {daily_df['total_pnl'].max():,.2f}")
    print(f"Worst day           : Rs {daily_df['total_pnl'].min():,.2f}")
    print(f"Max drawdown (cum)  : Rs {drawdown:,.2f}")
    print(f"Avg rolls/day       : {daily_df['num_rolls'].mean():.1f}")
    print("\nPer-day breakdown:")
    print(daily_df.to_string(index=False))


if __name__ == "__main__":
    run_backtest()
