"""Phase A, Step 0b: SECOND live probe. READ-ONLY. Same safety rules as probe_step0.py (GET only, cached token read without logging in, token never printed/written, no orders, no database, no strategy maths).

Purpose (what the first probe could not tell): behaviour of illiquid / far-OTM / next-expiry contracts over a 10-minute window, what actually triggers a websocket tick, what `exch_feed_time` does on quiet
contracts, whether `options-chain-v3` honours strikecount=12 (and for a second expiry), how often OI changes between chain calls, whether chain bid/ask agree with the websocket, per-call depth latency, and a
conservative REST-spacing / rate check. About 14 REST calls in 10 minutes, >= 2 s apart; ONE websocket connection with ~25 symbols.

Run in the middle of a normal session (suggested 11:00-14:30 IST; the script refuses outside 09:30-15:15 IST unless --allow-closed):
    python tools/phase_a_probe/probe_step0b.py --ws-seconds 600
Send back the output folder (default ~/phase_a_probe_output/step0b_<timestamp>/). It contains no token.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("probe_step0", os.path.join(_here, "probe_step0.py"))
P = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("probe_step0", P)
_spec.loader.exec_module(P)

IST = P.IST
STRIKE_STEP = 50
# (expiry index, offset in index points from ATM, option type, role).  expiry index 0 = nearest listed expiry, 1 = next, 2 = third.
UNIVERSE_SPEC = [
    (0, 0, "CE", "atm"), (0, 0, "PE", "atm"),
    (0, 300, "PE", "itm"), (0, -300, "CE", "itm"),                       # ITM put above spot, ITM call below spot
    (0, 200, "CE", "otm"), (0, -200, "PE", "otm"),
    (0, 600, "CE", "wing"), (0, -600, "PE", "wing"),
    (0, 1000, "CE", "far_wing"), (0, -1000, "PE", "far_wing"),
    (0, 1500, "CE", "far_wing"), (0, -1500, "PE", "far_wing"),
    (1, 0, "CE", "atm"), (1, 0, "PE", "atm"),
    (1, 300, "CE", "otm"), (1, -300, "PE", "otm"),
    (1, 800, "CE", "wing"), (1, -800, "PE", "wing"),
    (2, 0, "CE", "atm"), (2, 0, "PE", "atm"),
    (2, 1000, "CE", "wing"), (2, -1000, "PE", "wing"),
]
DEPTH_ROLES = [(0, 0, "CE"), (0, 0, "PE"), (0, 600, "CE"), (0, -1000, "PE"), (1, 0, "CE")]   # depth is called for these 5 only


# ------------------------------------------------------------------------------------------------ pure helpers
def parse_master(text):
    """{iso_expiry_date: {(strike, 'CE'|'PE'): fyers_symbol}} for NIFTY from Fyers' public NSE_FO symbol master (same columns as fyers_option_symbols.py: 8 expiry epoch, 9 symbol, 13 underlying, 15 strike, 16 type)."""
    out = {}
    for line in text.splitlines():
        p = line.split(",")
        if len(p) < 17 or p[13] != "NIFTY" or p[16] not in ("CE", "PE"):
            continue
        try:
            d = datetime.fromtimestamp(int(p[8]), IST).strftime("%Y-%m-%d")
            out.setdefault(d, {})[(float(p[15]), p[16])] = p[9]
        except (ValueError, OSError):
            continue
    return out


def iso_from_chain_date(s):
    d, m, y = s.split("-")
    return f"{y}-{m}-{d}"


def build_universe(spot, expiries, master, spec=UNIVERSE_SPEC, fallback_syms=None):
    """expiries: chain `expiryData` list (nearest first). Returns (rows, skipped). A target strike that is not listed is replaced by the nearest listed strike within 200 points, else skipped (and recorded).
    `fallback_syms`: {iso_date: {(strike, type): symbol}} built from chain responses, used only if the master has no entry for that expiry."""
    atm = int(round(spot / STRIKE_STEP) * STRIKE_STEP)
    rows, skipped, seen = [], [], set()
    for ei, off, typ, role in spec:
        if ei >= len(expiries):
            skipped.append(dict(expiry_index=ei, offset=off, type=typ, reason="expiry not listed")); continue
        iso = iso_from_chain_date(expiries[ei]["date"])
        listed = (master or {}).get(iso) or (fallback_syms or {}).get(iso) or {}
        strikes = sorted({k for (k, t) in listed if t == typ})
        want = atm + off
        if not strikes:
            skipped.append(dict(expiry_index=ei, offset=off, type=typ, reason="no listed strikes known for this expiry")); continue
        got = min(strikes, key=lambda k: abs(k - want))
        if abs(got - want) > 200:
            skipped.append(dict(expiry_index=ei, offset=off, type=typ, reason=f"nearest listed strike {got} is > 200 pts from target {want}")); continue
        if listed[(got, typ)] in seen:
            skipped.append(dict(expiry_index=ei, offset=off, type=typ, reason=f"nearest listed strike {got} is already in the universe (no duplicates)")); continue
        seen.add(listed[(got, typ)])
        rows.append(dict(symbol=listed[(got, typ)], expiry_index=ei, expiry_date=iso, strike=got, type=typ, role=role, offset_points=int(got - atm), offset_nominal=off, target_strike=want))
    return rows, skipped


def pair_change_signatures(ticks):
    """What changed between consecutive ticks of ONE symbol: Counter of tuples of field-group names. Groups: ltp, volume, ltt, quote_px, quote_size, tot_qty, atp, feed_time."""
    groups = dict(ltp=("ltp",), volume=("vol_traded_today", "last_traded_qty"), ltt=("last_traded_time",), quote_px=("bid_price", "ask_price"), quote_size=("bid_size", "ask_size"),
                  tot_qty=("tot_buy_qty", "tot_sell_qty"), atp=("avg_trade_price",), feed_time=("exch_feed_time",))
    c = Counter()
    for (_, a), (_, b) in zip(ticks, ticks[1:]):
        sig = tuple(g for g, fs in groups.items() if any(a.get(f) != b.get(f) for f in fs))
        c["+".join(sig) if sig else "NOTHING_CHANGED"] += 1
    return dict(c.most_common())


def symbol_liquidity_row(sym, meta, ticks, t_end):
    """One line per symbol: tick counts, trades, quote activity, spread, sizes, ages -- descriptive only."""
    m = [x for _, x in ticks]
    row = dict(symbol=sym, **{k: meta.get(k) for k in ("expiry_date", "strike", "type", "role", "offset_points")}, n_ticks=len(m))
    if not m or "bid_price" not in m[0]:
        return row
    caps = [c for c, _ in ticks]
    gaps = [b - a for a, b in zip(caps, caps[1:])]
    vols = [x["vol_traded_today"] for x in m]
    q = [(x["bid_price"], x["ask_price"], x["bid_size"], x["ask_size"]) for x in m]
    sp = [x["ask_price"] - x["bid_price"] for x in m if 0 < x["bid_price"] < x["ask_price"]]          # crossed / locked / one-sided quotes are counted separately, not in the spread stats
    mid = [(x["ask_price"] + x["bid_price"]) / 2 for x in m if 0 < x["bid_price"] < x["ask_price"]]
    row.update(
        tick_gap_median_s=round(sorted(gaps)[len(gaps) // 2], 3) if gaps else None, tick_gap_max_s=round(max(gaps), 3) if gaps else None,
        volume_increments=sum(1 for a, b in zip(vols, vols[1:]) if b > a), contracts_traded=vols[-1] - vols[0],
        distinct_quote_states=len(set(q)), zero_bid_ticks=sum(1 for x in m if x["bid_price"] <= 0), zero_ask_ticks=sum(1 for x in m if x["ask_price"] <= 0),
        crossed_ticks=sum(1 for x in m if x["bid_price"] > x["ask_price"] > 0), locked_ticks=sum(1 for x in m if x["bid_price"] == x["ask_price"] > 0),
        spread_rs_median=round(sorted(sp)[len(sp) // 2], 3) if sp else None, spread_rs_max=round(max(sp), 3) if sp else None,
        spread_pct_median=round(100 * sorted(s / mm for s, mm in zip(sp, mid))[len(sp) // 2], 3) if sp else None,
        bid_size_median=sorted(x["bid_size"] for x in m)[len(m) // 2], ask_size_median=sorted(x["ask_size"] for x in m)[len(m) // 2],
        never_traded_today=vols[-1] == 0, last_trade_age_at_end_s=round(t_end - m[-1]["last_traded_time"], 1) if m[-1].get("last_traded_time") else None,
        feed_minus_last_trade_max_s=max(x["exch_feed_time"] - (x.get("last_traded_time") or 0) for x in m) if all(x.get("last_traded_time") for x in m) else None,
        feed_minus_last_trade_median_s=sorted(x["exch_feed_time"] - x["last_traded_time"] for x in m)[len(m) // 2] if all(x.get("last_traded_time") for x in m) else None,
        capture_minus_feed_median_s=round(sorted(c - x["exch_feed_time"] for c, x in ticks)[len(m) // 2], 3), ltp_ge_ask=sum(1 for x in m if x["ltp"] >= x["ask_price"] > 0), ltp_le_bid=sum(1 for x in m if 0 < x["ltp"] <= x["bid_price"]),
        change_signatures=pair_change_signatures(ticks))
    return row


def oi_cadence(chain_calls):
    """chain_calls: [(capture_epoch, {symbol: oi})] in time order. Per consecutive pair: how many symbols' OI changed."""
    out = []
    for (t0, a), (t1, b) in zip(chain_calls, chain_calls[1:]):
        common = set(a) & set(b)
        out.append(dict(from_ist=P.ist(t0), to_ist=P.ist(t1), seconds=round(t1 - t0, 1), n_symbols=len(common), n_oi_changed=sum(1 for s in common if a[s] != b[s]),
                        example_changes={s: [a[s], b[s]] for s in sorted(s2 for s2 in common if a[s2] != b[s2])[:5]}))
    return out


def chain_rows(body):
    rows = ((body or {}).get("data") or {}).get("optionsChain") or []
    return [r for r in rows if isinstance(r, dict) and r.get("strike_price", -1) != -1]


def chain_vs_ws(cap, rows, ticks_by_symbol, window=2.0):
    """Chain bid/ask vs the nearest websocket tick (within `window` s) of the same symbol."""
    res = []
    for r in rows:
        ts = ticks_by_symbol.get(r.get("symbol")) or []
        if not ts:
            continue
        c, m = min(ts, key=lambda t: abs(t[0] - cap))
        if abs(c - cap) > window:
            continue
        res.append(dict(symbol=r["symbol"], gap_s=round(c - cap, 3), bid_equal=r.get("bid") == m.get("bid_price"), ask_equal=r.get("ask") == m.get("ask_price"), ltp_equal=r.get("ltp") == m.get("ltp"),
                        volume_equal=r.get("volume") == m.get("vol_traded_today"), chain_volume_minus_ws=(r.get("volume") - m["vol_traded_today"]) if r.get("volume") is not None and m.get("vol_traded_today") is not None else None))
    n = len(res)
    return dict(n_compared=n, bid_equal=sum(x["bid_equal"] for x in res), ask_equal=sum(x["ask_equal"] for x in res), ltp_equal=sum(x["ltp_equal"] for x in res), volume_equal=sum(x["volume_equal"] for x in res), rows=res[:40])


def schedule(ws_seconds):
    """(fraction of ws window, kind, arg). Deliberately sparse; every REST call is >= min_gap apart (enforced by Http)."""
    ev = [(0.10, "chain_e0", 1), (0.30, "quotes", 1), (0.50, "chain_e0", 2), (0.60, "quotes", 2), (0.70, "depth", 0), (0.85, "quotes", 3), (0.93, "chain_e0", 3)]
    return [(f * ws_seconds, k, a) for f, k, a in ev]


# ------------------------------------------------------------------------------------------------ orchestration
def run(out_dir, http, ws_factory, ws_seconds, sleep=time.sleep, master_text=None, app_id="", token=""):
    os.makedirs(out_dir, exist_ok=True)
    S = dict(probe="phase_a_step0b", started_ist=datetime.now(IST).isoformat(timespec="seconds"), ws_seconds=ws_seconds, rest_min_gap_s=http._min_gap)
    # --- pre-flight REST: chain strikecount=12 for expiry 0 and 1
    rec, ch0 = http.get("chain_e0_sc12_pre", P.CHAIN_URL, {"symbol": P.INDEX_SYMBOL, "strikecount": 12, "timestamp": ""})
    if ch0 is None:
        S["fatal"] = "no chain response"; _save(out_dir, S, http); return S
    safe_write_json(out_dir, "raw_chain_e0_sc12_pre.json", ch0)
    exps = (ch0.get("data") or {}).get("expiryData") or []
    t_ch0 = time.time()
    rows0 = chain_rows(ch0)
    spot = next((r.get("ltp") for r in ((ch0.get("data") or {}).get("optionsChain") or []) if r.get("strike_price", -1) == -1), None)
    strikes0 = sorted({r["strike_price"] for r in rows0})
    S["strikecount_12_check"] = dict(n_option_rows=len(rows0), n_strikes=len(strikes0), min_strike=strikes0[0] if strikes0 else None, max_strike=strikes0[-1] if strikes0 else None,
                                     expected_if_2N_plus_1_strikes=50, honoured=len(rows0) == 50, expiry_requested="nearest (timestamp='')", n_expiries_listed=len(exps), expiries_first_4=exps[:4],
                                     india_vix=((ch0.get("data") or {}).get("indiavixData") or {}).get("ltp"), underlying_fp=next((r.get("fp") for r in (ch0.get("data") or {}).get("optionsChain") or [] if r.get("strike_price", -1) == -1), None))
    fallback = {}
    chain_by_exp = {0: ch0}
    if len(exps) > 1:
        rec, ch1 = http.get("chain_e1_sc12_pre", P.CHAIN_URL, {"symbol": P.INDEX_SYMBOL, "strikecount": 12, "timestamp": exps[1]["expiry"]})
        if ch1 is not None:
            safe_write_json(out_dir, "raw_chain_e1_sc12_pre.json", ch1)
            r1 = chain_rows(ch1)
            s1 = sorted({r["strike_price"] for r in r1})
            syms1 = {r["symbol"] for r in r1}
            S["second_expiry_chain_check"] = dict(requested_expiry=exps[1], n_option_rows=len(r1), n_strikes=len(s1), symbols_look_like_requested_expiry=sorted({re.sub(r"\d+(CE|PE)$", "", x) for x in syms1}),
                                                  differs_from_first_expiry_symbols=syms1.isdisjoint({r["symbol"] for r in rows0}))
            chain_by_exp[1] = ch1
    for ei, body in chain_by_exp.items():
        iso = iso_from_chain_date(exps[ei]["date"])
        fallback[iso] = {(float(r["strike_price"]), r["option_type"]): r["symbol"] for r in chain_rows(body) if r.get("symbol")}
    master = parse_master(master_text) if master_text else {}
    S["master_available"] = bool(master)
    uni, skipped = build_universe(spot, exps, master, fallback_syms=fallback)
    fut = P.find_future(master_text, time.time()) if master_text else None
    symbols = [u["symbol"] for u in uni] + [P.INDEX_SYMBOL] + ([fut] if fut else [])
    S["spot_at_start"], S["universe"], S["universe_skipped"], S["n_ws_symbols"] = spot, uni, skipped, len(symbols)
    meta = {u["symbol"]: u for u in uni}
    # --- websocket + scheduled REST samples
    ws = P.WsCollector(app_id, token, symbols, ws_factory)
    t0 = time.time()
    ws.start()
    chain_calls, chain_checks, quote_samples, depth_results = [(t_ch0, {r["symbol"]: r.get("oi") for r in rows0})], [], [], []
    depth_syms = [next((u["symbol"] for u in uni if u["expiry_index"] == ei and u["offset_nominal"] == off and u["type"] == typ), None) for ei, off, typ in DEPTH_ROLES]
    for at, kind, arg in schedule(ws_seconds):
        wait = t0 + at - time.time()
        if wait > 0:
            sleep(wait)
        cap = time.time()
        if kind == "chain_e0":
            r, b = http.get(f"chain_e0_sc12_during_{arg}", P.CHAIN_URL, {"symbol": P.INDEX_SYMBOL, "strikecount": 12, "timestamp": ""})
            if b is not None:
                rr = chain_rows(b)
                chain_calls.append((cap, {x["symbol"]: x.get("oi") for x in rr}))
                chain_checks.append(dict(call=arg, rows=rr, cap=cap))
        elif kind == "quotes":
            r, b = http.get(f"quotes_during_ws_{arg}", P.QUOTES_URL, {"symbols": ",".join(symbols)})
            for d in ((b or {}).get("d") or []):
                if isinstance(d, dict) and d.get("v"):
                    quote_samples.append((cap, d.get("n"), d["v"]))
        elif kind == "depth":
            for sym in depth_syms:
                if not sym:
                    continue
                c2 = time.time()
                r, b = http.get("depth", P.DEPTH_URL, {"symbol": sym, "ohlcv_flag": 1})
                a = P.analyze_depth(r, b)
                inner = ((b or {}).get("d") or {}).get(sym) or {}
                depth_results.append(dict(symbol=sym, capture_epoch=c2, latency_ms=(r or {}).get("latency_ms"), status=(r or {}).get("status"), levels_bid=a.get("bids_levels"), levels_ask=a.get("ask_levels"),
                                          top_bid=(inner.get("bids") or [{}])[0], top_ask=(inner.get("ask") or [{}])[0], ltt=inner.get("ltt"), oi=inner.get("oi"), volume=inner.get("v")))
    wait = t0 + ws_seconds - time.time()
    if wait > 0:
        sleep(wait)
    ws.stop()
    by = ws.by_symbol()
    t_end = time.time()
    tick_path = os.path.join(out_dir, "ws_ticks.jsonl.gz")
    with gzip.open(tick_path, "wt", encoding="utf-8") as f:
        for c, m in sorted(ws.ticks, key=lambda t: t[0]):
            line = json.dumps(dict(capture_epoch=c, capture_ist=P.ist(c), msg=m), default=str)
            for sec in P._SECRETS:
                if sec and sec in line:
                    raise RuntimeError("token in tick output")
            f.write(line + "\n")
    # --- analyses
    table = [symbol_liquidity_row(s, meta.get(s, {}), by.get(s, []), t_end) for s in symbols]
    with open(os.path.join(out_dir, "probe2_symbol_table.csv"), "w", newline="", encoding="utf-8") as f:
        cols = [c for c in dict.fromkeys(k for r in table for k in r) if c != "change_signatures"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(table)
    S["websocket"] = dict(events=ws.events, connected=any(e["kind"] == "connected" for e in ws.events), errors=[e for e in ws.events if e["kind"] in ("error", "close_failed")], n_ticks_total=len(ws.ticks),
                          symbols_with_no_ticks=[s for s in symbols if s not in by], per_symbol=table)
    agg = Counter()
    for s, t in by.items():
        if s in meta or s == fut:
            for k, v in pair_change_signatures(t).items():
                agg[k] += v
    S["tick_trigger_signatures_all_option_and_future_symbols"] = dict(agg.most_common())
    S["feed_time_semantics"] = dict(
        pairs_where_only_feed_time_changed=agg.get("feed_time", 0), pairs_with_nothing_changed=agg.get("NOTHING_CHANGED", 0),
        pairs_quote_changed_without_trade=sum(v for k, v in agg.items() if ("quote_px" in k or "quote_size" in k) and not any(x in k for x in ("ltp", "volume", "ltt"))),
        pairs_quote_changed_without_trade_and_feed_advanced=sum(v for k, v in agg.items() if ("quote_px" in k or "quote_size" in k) and not any(x in k for x in ("ltp", "volume", "ltt")) and "feed_time" in k),
        note="descriptive counts only; see per-symbol feed_minus_last_trade_* and change_signatures")
    S["rest_vs_ws"] = P.compare_rest_with_ws(quote_samples, by)
    S["quotes_tt_values"] = sorted({str(v.get("tt")) for _, _, v in quote_samples})
    S["chain_oi_cadence_e0"] = oi_cadence(chain_calls)
    S["chain_vs_ws_bid_ask"] = [dict(call=c["call"], **{k: v for k, v in chain_vs_ws(c["cap"], c["rows"], by).items() if k != "rows"}) for c in chain_checks]
    S["depth"] = dict(calls=depth_results, latency_ms=P._stats([d["latency_ms"] for d in depth_results if d.get("latency_ms") is not None]))
    S["feed_probe_window"] = dict(start_ist=P.ist(t0), end_ist=P.ist(t_end))
    _save(out_dir, S, http)
    return S


def safe_write_json(out_dir, name, obj):
    P.safe_write(os.path.join(out_dir, name), P.jdump(obj))


def _save(out_dir, S, http):
    S["http_calls"] = http.log
    S["n_rest_calls"] = sum(1 for r in http.log if "status" in r or "error" in r)
    S["rate_limited"] = any(r.get("rate_limited") for r in http.log)
    S["local_minus_server_date_header_s"] = P.clock_skew(http.log)
    S["finished_ist"] = datetime.now(IST).isoformat(timespec="seconds")
    P.safe_write(os.path.join(out_dir, "probe2_summary.json"), P.jdump(S))


def market_window_warning_b(ws_seconds, now=None):
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return "weekend"
    end = now + timedelta(seconds=ws_seconds + 120)
    lo, hi = datetime(2000, 1, 1, 9, 30).time(), datetime(2000, 1, 1, 15, 15).time()
    if not (lo <= now.time() and end.time() <= hi):
        return "needs a window inside 09:30-15:15 IST (mid-session)"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "phase_a_probe_output", "step0b_" + datetime.now(IST).strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--ws-seconds", type=float, default=600)
    ap.add_argument("--rest-min-gap", type=float, default=2.0)
    ap.add_argument("--allow-closed", action="store_true")
    a = ap.parse_args(argv)
    warn = market_window_warning_b(a.ws_seconds)
    if warn and not a.allow_closed:
        print(f"Not a mid-session window ({warn}). Re-run in market hours, or pass --allow-closed (timestamp findings are then meaningless).")
        return 2
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
    try:
        import fyers_auth
    except (KeyError, ImportError) as e:
        print(f"Cannot import the app's fyers_auth ({type(e).__name__}: {e}); run from the app's Python environment with its .env present. Nothing was sent.")
        return 3
    token = fyers_auth._load_cached_token()            # read-only: never logs in
    if not token:
        print("No valid cached token for today (run the app / `python fyers_auth.py` first). The probe does not log in.")
        return 4
    P._SECRETS.extend([token, f"{fyers_auth.APP_ID}:{token}"])
    print(f"Cached token found (not printed). Output dir: {a.out}. Running {a.ws_seconds:.0f} s; about 14 REST calls >= {a.rest_min_gap}s apart.")
    master = None
    try:
        import requests
        master = requests.get(P.MASTER_URL, timeout=30).text
    except Exception as e:
        print(f"symbol master not downloaded ({type(e).__name__}); far-OTM strikes beyond the chain window will be skipped")
    http = P.Http(fyers_auth.APP_ID, token, min_gap=a.rest_min_gap)
    S = run(a.out, http, None, a.ws_seconds, master_text=master, app_id=fyers_auth.APP_ID, token=token)
    print(f"done. ticks={S.get('websocket', {}).get('n_ticks_total')} symbols={S.get('n_ws_symbols')} rest_calls={S.get('n_rest_calls')} rate_limited={S.get('rate_limited')}  files in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
