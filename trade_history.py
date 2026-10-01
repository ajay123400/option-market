"""trade_history.py -- reconstructs complete round-trip trades (entry paired
with its roll/EOD exit) from paper_trade.py's flat event log
(results/paper_trade_log.jsonl), for the Trade History page. This is the
one place that actually persists across app restarts and days -- the
simulator's event log is intentionally in-memory only (a re-run), and
backtest.py's CSVs get overwritten on every run.
"""
import json
import os
from collections import defaultdict

import charges as charges_mod
import paths
from strategy_engine import LOT_SIZE

LOG_PATH = os.path.join(paths.BASE_DIR, "results", "paper_trade_log.jsonl")


def load_raw_events():
    if not os.path.exists(LOG_PATH):
        return []
    events = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def build_trades():
    """Pairs each ENTRY_* with the ROLL_EXIT_*/EOD_EXIT_* that later closes
    the same side on the same date, producing one row per complete
    round-trip (a day with 3 rolls on one side yields 3 separate rows)."""
    events = load_raw_events()
    open_leg = {}  # (date, side) -> entry event
    trades = []
    for e in events:
        time_str = e.get("time") or ""
        date = time_str[:10]
        etype = e.get("type", "")
        if etype.startswith("ENTRY_"):
            side = etype.split("_")[-1]
            open_leg[(date, side)] = e
        elif etype.startswith(("ROLL_EXIT_", "EOD_EXIT_", "SL_EXIT_", "RISK_EXIT_")):
            side = etype.split("_")[-1]
            entry = open_leg.pop((date, side), None)
            trades.append({
                "date": date,
                "side": side,
                "strike": e.get("strike"),
                "entry_time": entry.get("time") if entry else None,
                "entry_price": entry.get("price") if entry else None,
                "exit_time": e.get("time"),
                "exit_price": e.get("price"),
                "hedge_strike": e.get("hedge_strike"),
                "hedge_entry_price": entry.get("hedge_price") if entry else None,
                "hedge_exit_price": e.get("hedge_price"),
                "exit_type": etype.rsplit("_", 2)[0].replace("_EXIT", "").replace("RISK", "LOSS CAP"),
                "pnl": e.get("pnl"),
            })
            trades[-1]["charges"] = _trade_charges(trades[-1])
            trades[-1]["net_pnl"] = (round(trades[-1]["pnl"] - trades[-1]["charges"], 2)
                                     if trades[-1]["pnl"] is not None else None)
    trades.sort(key=lambda t: (t["date"], t["exit_time"] or ""))
    return trades


def _trade_charges(t):
    """Short leg + its hedge, each entered and exited: 4 orders at 1 lot."""
    c = 0.0
    if t["entry_price"] is not None and t["exit_price"] is not None:
        c += charges_mod.round_trip("SELL", t["entry_price"], t["exit_price"], LOT_SIZE)
    if t["hedge_entry_price"] is not None and t["hedge_exit_price"] is not None:
        c += charges_mod.round_trip("BUY", t["hedge_entry_price"], t["hedge_exit_price"], LOT_SIZE)
    return round(c, 2)


def summary_by_date(trades):
    by_date = defaultdict(lambda: {"num_trades": 0, "pnl": 0.0, "charges": 0.0, "net_pnl": 0.0})
    for t in trades:
        d = by_date[t["date"]]
        d["num_trades"] += 1
        d["pnl"] += t["pnl"] or 0.0
        d["charges"] += t.get("charges") or 0.0
        d["net_pnl"] += (t["pnl"] or 0.0) - (t.get("charges") or 0.0)
    return [{"date": d, **v} for d, v in sorted(by_date.items())]


if __name__ == "__main__":
    trades = build_trades()
    print(f"{len(trades)} trades found.")
    for t in trades[:5]:
        print(t)
    print("\nBy date:")
    for row in summary_by_date(trades):
        print(row)
