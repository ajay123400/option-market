"""Read-only acceptance summary of a daily recorder database (for the supervised run and for daily health checks).

    python -m collector.report E:\\nifty_microstructure\\micro_20261005.sqlite
"""
import argparse
import json
import os
import sqlite3
import statistics
import sys


def _q(con, sql, *a):
    return con.execute(sql, a).fetchall()


def summarize(db_path: str) -> dict:
    uri = "file:" + os.path.abspath(db_path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        S = dict(db=os.path.basename(db_path))
        S["cycle_status"] = dict(_q(con, "SELECT status, COUNT(*) FROM cycles GROUP BY status"))
        n_cycles = sum(S["cycle_status"].values())
        n_ok = S["cycle_status"].get("ok", 0)
        S["n_cycles"], S["share_ok"] = n_cycles, (n_ok / n_cycles if n_cycles else None)
        S["rows_per_ok_cycle"] = [r[0] for r in _q(con, "SELECT n_rows FROM cycles WHERE status='ok' ORDER BY cycle_id")]
        S["expected_per_cycle"] = [r[0] for r in _q(con, "SELECT n_expected FROM cycles WHERE status='ok' ORDER BY cycle_id")]
        S["duplicate_keys"] = _q(con, "SELECT COUNT(*) FROM (SELECT cycle_id, symbol FROM quotes GROUP BY cycle_id, symbol HAVING COUNT(*)>1)")[0][0]
        S["rows_by_source"] = dict(_q(con, "SELECT data_source, COUNT(*) FROM quotes GROUP BY data_source"))
        flags = {}
        for (f,) in _q(con, "SELECT flags FROM quotes WHERE flags IS NOT NULL AND flags<>''"):
            for x in f.split(","):
                flags[x] = flags.get(x, 0) + 1
        S["flag_counts"] = dict(sorted(flags.items(), key=lambda kv: -kv[1]))
        S["skew_est_s"] = [r[0] for r in _q(con, "SELECT skew_est_s FROM cycles WHERE skew_est_s IS NOT NULL ORDER BY cycle_id")]
        S["index_feed_age_s"] = [r[0] for r in _q(con, "SELECT index_feed_age_s FROM cycles WHERE index_feed_age_s IS NOT NULL ORDER BY cycle_id")]
        S["ws"] = dict(subscribed=[r[0] for r in _q(con, "SELECT ws_subscribed FROM cycles ORDER BY cycle_id")], ticks_total=[r[0] for r in _q(con, "SELECT ws_ticks_total FROM cycles ORDER BY cycle_id")])
        S["rest"] = dict(calls=[r[0] for r in _q(con, "SELECT rest_calls FROM cycles ORDER BY cycle_id")], latency_ms_max=[r[0] for r in _q(con, "SELECT rest_latency_ms_max FROM cycles ORDER BY cycle_id")],
                         rate_limited_cycles=_q(con, "SELECT COUNT(*) FROM cycles WHERE rate_limited=1")[0][0])
        S["oi_missing_option_rows"] = _q(con, "SELECT COUNT(*) FROM quotes WHERE kind='option' AND oi IS NULL")[0][0]
        S["option_rows_without_ws"] = _q(con, "SELECT COUNT(*) FROM quotes WHERE kind='option' AND data_source<>'fyers:ws-full'")[0][0]
        ages = [r[0] for r in _q(con, "SELECT capture_minus_feed_s FROM quotes WHERE kind='option' AND capture_minus_feed_s IS NOT NULL")]
        if ages:
            ages.sort()
            S["option_capture_minus_feed_s"] = dict(n=len(ages), p50=ages[len(ages) // 2], p90=ages[int(0.9 * len(ages))], max=ages[-1])
        sp = [r[0] for r in _q(con, "SELECT ask-bid FROM quotes WHERE kind='option' AND bid>0 AND ask>bid AND offset_strikes=0")]
        if sp:
            S["atm_spread_rs_median"] = statistics.median(sp)
        S["notes"] = [r[0] for r in _q(con, "SELECT reason FROM cycles WHERE reason IS NOT NULL ORDER BY cycle_id")][:20]
        S["acceptance"] = dict(
            share_ok_at_least_95pct=(S["share_ok"] or 0) >= 0.95, zero_duplicates=S["duplicate_keys"] == 0,
            rows_match_expected=all(a == b for a, b in zip(S["rows_per_ok_cycle"], S["expected_per_cycle"])), universe_rows_per_ok_cycle=sorted(set(S["rows_per_ok_cycle"])),
            no_rate_limit=S["rest"]["rate_limited_cycles"] == 0)
        return S
    finally:
        con.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    a = ap.parse_args(argv)
    print(json.dumps(summarize(a.db), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
