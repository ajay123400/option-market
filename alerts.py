"""alerts.py -- user price / P&L alerts, delivered on Telegram.

Checked by the risk monitor (manual_monitor.py) on every market-hours pass,
so they fire whether or not a browser tab is open. One-shot: an alert
disarms itself when it triggers.

Kinds:
  spot_above / spot_below   -- NIFTY 50 index level
  ltp_above  / ltp_below    -- a contract's LTP (needs `symbol`)
  pnl_above  / pnl_below    -- total LIVE P&L of all open Strategy Builder legs (Rs)
"""
import os
import threading
import time
import json

import paths

ALERTS_FILE = os.path.join(paths.BASE_DIR, "results", "alerts.json")
KINDS = {"spot_above", "spot_below", "ltp_above", "ltp_below", "pnl_above", "pnl_below"}
_lock = threading.Lock()


def load():
    try:
        with open(ALERTS_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _save(rows):
    os.makedirs(os.path.dirname(ALERTS_FILE), exist_ok=True)
    paths.atomic_write_json(ALERTS_FILE, rows)


def add(kind, value, symbol=None, note=""):
    if kind not in KINDS:
        raise ValueError(f"Unknown alert kind {kind!r}")
    if kind.startswith("ltp_") and not symbol:
        raise ValueError("An LTP alert needs a contract symbol.")
    row = {"id": f"al_{int(time.time() * 1000)}", "kind": kind, "value": float(value),
           "symbol": symbol, "note": note, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "triggered_at": None, "triggered_value": None}
    with _lock:
        rows = load()
        rows.append(row)
        _save(rows)
    return row


def delete(alert_id):
    with _lock:
        rows = load()
        _save([r for r in rows if r["id"] != alert_id])


def describe(a):
    what = {"spot": "NIFTY", "ltp": (a.get("symbol") or "").replace("NSE:", ""), "pnl": "Open P&L"}[a["kind"].split("_")[0]]
    return f"{what} {'≥' if a['kind'].endswith('above') else '≤'} {a['value']:g}"


def check(spot, ltp_lookup, open_pnl, notify):
    """Fires every armed alert whose condition holds now. `notify(msg)`
    delivers it. Returns the alerts triggered this pass."""
    fired = []
    with _lock:
        rows = load()
        for a in rows:
            if a.get("triggered_at"):
                continue
            base = a["kind"].split("_")[0]
            cur = spot if base == "spot" else open_pnl if base == "pnl" else ltp_lookup.get(a.get("symbol"))
            if cur is None:
                continue
            hit = cur >= a["value"] if a["kind"].endswith("above") else cur <= a["value"]
            if hit:
                a["triggered_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                a["triggered_value"] = cur
                fired.append(a)
        if fired:
            _save(rows)
    for a in fired:
        notify(f"⏰ Alert: {describe(a)} -- now {a['triggered_value']:,.2f}" + (f" ({a['note']})" if a.get("note") else ""))
    return fired
