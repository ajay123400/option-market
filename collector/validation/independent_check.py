"""Independent verification of a recorder database (sqlite3 + stdlib only; no collector imports): recomputes the universe rules, the atm/offset/DTE columns, every quality flag and the cycle counters
from the stored raw columns and the configuration stored in the database itself, and compares them with what the recorder wrote.

    python collector/validation/independent_check.py <micro_YYYYMMDD.sqlite>
"""
import json
import math
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
WS = ("fyers:ws-full", "arrow:ws-full")            # websocket-row labels (either broker; deliberately not imported from collector)
CHAIN = ("fyers:options-chain-v3", "arrow:rest-chain")

IST = timezone(timedelta(hours=5, minutes=30))
bad = 0


def check(name, cond, detail=""):
    global bad
    bad += not cond
    print(f"  {'ok ' if cond else 'BAD'} {name} {detail}")


def main(path):
    con = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    cfg = json.loads(meta["config"])
    day = date.fromisoformat(meta["day"])
    cyc = {r["cycle_id"]: r for r in con.execute("SELECT * FROM cycles")}
    print(f"database {path}: day {day}, {len(cyc)} cycles, {con.execute('SELECT COUNT(*) FROM quotes').fetchone()[0]} quote rows")
    check("schema version recorded", meta["schema_version"] == "1")
    check("no duplicate (cycle_id, symbol)", con.execute("SELECT COUNT(*) FROM (SELECT 1 FROM quotes GROUP BY cycle_id, symbol HAVING COUNT(*)>1)").fetchone()[0] == 0)
    check("every quote row belongs to a recorded cycle", con.execute("SELECT COUNT(*) FROM quotes WHERE cycle_id NOT IN (SELECT cycle_id FROM cycles)").fetchone()[0] == 0)
    check("missed cycles carry no quote rows", con.execute("SELECT COUNT(*) FROM quotes q JOIN cycles c USING(cycle_id) WHERE c.status='missed'").fetchone()[0] == 0)
    # grid
    first_h, first_m = map(int, cfg["first_cycle"].split(":"))
    last_h, last_m = map(int, cfg["last_cycle"].split(":"))
    okgrid = True
    for cid, c in cyc.items():
        t = datetime.fromtimestamp(c["scheduled_ts"], IST)
        lo, hi = t.replace(hour=first_h, minute=first_m, second=0), t.replace(hour=last_h, minute=last_m, second=0)
        okgrid &= (t.date() == day and t.second == 0 and (t.minute - first_m) % cfg["grid_minutes"] == 0 and lo <= t <= hi and cid == t.strftime("%Y-%m-%dT%H:%M"))
    check("every cycle sits on the 5-minute grid 09:16..15:36 with a matching id", okgrid)
    half_up = lambda x, step: int(math.floor(x / step + 0.5) * step)
    step, rec_n, max_days = cfg["strike_step"], cfg["record_strikes"], cfg["max_expiry_days"]
    universe_problems, counter_problems = [], []
    for cid, c in cyc.items():
        if c["status"] == "missed":
            continue
        rows = con.execute("SELECT * FROM quotes WHERE cycle_id=?", (cid,)).fetchall()
        opts = [r for r in rows if r["kind"] == "option"]
        label = cid[-5:]
        if c["n_rows"] != len(rows) or c["n_ws_rows"] != sum(r["data_source"] in WS for r in rows) or c["n_chain_rows"] != sum(r["data_source"] in CHAIN for r in rows):
            counter_problems.append((label, c["n_rows"], len(rows)))
        if c["spot"] and opts:
            atm = half_up(c["spot"], step)
            exp = set(json.loads(c["expiries_json"]))
            per = {}
            problems = []
            for r in opts:
                if r["atm_strike"] != atm or r["offset_strikes"] != round((r["strike"] - atm) / step) or abs(r["offset_strikes"]) > rec_n:
                    problems.append(("atm/offset", r["symbol"]))
                if r["expiry_date"] not in exp or not (0 <= (date.fromisoformat(r["expiry_date"]) - day).days <= max_days) or r["dte_days"] != (date.fromisoformat(r["expiry_date"]) - day).days:
                    problems.append(("expiry/dte", r["symbol"]))
                per.setdefault(r["expiry_date"], set()).add((r["strike"], r["option_type"]))
            for e, s in per.items():
                if s != {(k, t) for k in {x[0] for x in s} for t in ("CE", "PE")} or len({x[0] for x in s}) != 2 * rec_n + 1:
                    problems.append(("window", e))
            if c["status"] in ("ok",) and (len(per) != len(exp) or c["n_expected"] != len(rows)):
                problems.append(("completeness", len(per)))
            universe_problems += [(label,) + p for p in problems]
    check("cycle counters (n_rows, n_ws_rows, n_chain_rows) equal the stored rows", not counter_problems, str(counter_problems[:3]))
    check("universe rules: atm/offset columns, |offset| <= 12, 25 strikes x CE/PE per expiry, DTE 0..15, rows == expected on ok cycles", not universe_problems, str(universe_problems[:3]))
    # timestamps & derived columns
    bad_t = con.execute("SELECT COUNT(*) FROM quotes WHERE quote_feed_ts IS NOT NULL AND ABS(capture_minus_feed_s - (capture_ts - quote_feed_ts)) > 0.001").fetchone()[0]
    bad_l = con.execute("SELECT COUNT(*) FROM quotes WHERE last_trade_ts IS NOT NULL AND last_trade_ts > 0 AND ABS(capture_minus_last_trade_s - (capture_ts - last_trade_ts)) > 0.001").fetchone()[0]
    bad_rx = con.execute("SELECT COUNT(*) FROM quotes WHERE rx_ts IS NOT NULL AND rx_ts > capture_ts + 0.001").fetchone()[0]
    check("capture_minus_feed / last_trade columns equal capture_ts minus the exchange timestamps", bad_t == 0 and bad_l == 0, f"({bad_t}, {bad_l})")
    check("websocket receive time never after the capture instant", bad_rx == 0, f"({bad_rx})")
    # flags, independently
    skew = {cid: (c["skew_est_s"] or 0.0) for cid, c in cyc.items()}
    prev = {}
    mism = 0
    for r in con.execute("SELECT q.* FROM quotes q ORDER BY symbol, cycle_id"):
        f = set()
        bid, ask, ltp, kind = r["bid"], r["ask"], r["ltp"], r["kind"]
        if kind != "index":
            if bid is None or bid <= 0:
                f.add("NO_BID")
            if ask is None or ask <= 0:
                f.add("NO_ASK")
        if any(x is not None and x < 0 for x in (bid, ask, ltp)):
            f.add("BAD_PRICE")
        if ltp is not None and ltp == 0:
            f.add("ZERO_LTP")
        if bid and ask and bid > 0 and ask > 0:
            if bid > ask:
                f.add("CROSSED")
            elif bid == ask:
                f.add("LOCKED")
            elif ask - bid > max(cfg["wide_spread_abs"], cfg["wide_spread_pct"] * (ask + bid) / 2):
                f.add("WIDE_SPREAD")
        feed, cap = r["quote_feed_ts"], r["capture_ts"]
        if r["data_source"] not in WS or feed is None:
            f.add("NO_FEED" if feed is None and r["data_source"] in WS else "AGE_UNKNOWN")
        elif cap is not None:
            if cap - feed - skew[r["cycle_id"]] > cfg["stale_quote_s"]:
                f.add("STALE_QUOTE")
            if feed - cap > 2.0:
                f.add("FEED_IN_FUTURE")
            p = prev.get(r["symbol"])
            if p is not None and feed < p:
                f.add("FEED_REGRESSED")
        if feed is not None:
            prev[r["symbol"]] = feed
        if kind == "option" and r["oi"] is None:
            f.add("NO_OI")
        if kind != "index" and not r["volume"]:
            f.add("NO_VOLUME_TODAY")
        if ",".join(sorted(f)) != (r["flags"] or ""):
            mism += 1
    check("every stored quality flag string equals the independent recomputation", mism == 0, f"({mism} mismatches)")
    st = dict(con.execute("SELECT status, COUNT(*) FROM cycles GROUP BY status").fetchall())
    print("  cycle status counts:", st)
    print("\nRESULT:", "ALL CHECKS PASSED" if bad == 0 else f"{bad} CHECKS FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
