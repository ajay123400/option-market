"""app.py -- minimal Flask app for the live NIFTY Option Chain view.
Polls Fyers' options-chain-v3 endpoint (the only one that carries OI) fresh
on every /api/chain request; the browser re-polls every couple of seconds.
"""
import json
import os

from flask import Flask, jsonify, render_template, request

import data_health
import fyers_option_chain as chain_mod
import historical_recorder
import charges as charges_mod
import manual_monitor
import manual_trades
import market_calendar as mc
import paths
import simulator

# templates/ is a read-only bundled resource -- in a frozen .exe it lives
# under paths.BUNDLE_DIR (PyInstaller's temp extraction dir), not next to
# the .exe like the writable data/config files paths.BASE_DIR points at.
app = Flask(__name__, template_folder=os.path.join(paths.BUNDLE_DIR, "templates"),
            static_folder=os.path.join(paths.BUNDLE_DIR, "static"))


@app.before_request
def _block_cross_site_writes():
    """Every POST here changes positions/journal. Endpoints read JSON with
    force=True, so a page on ANY other website could fire a text/plain
    POST at localhost (no CORS preflight for that) -- e.g. close_all. The
    browser always sends Origin on cross-site POSTs, so only same-origin
    ones are accepted."""
    if request.method == "POST":
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            return jsonify({"ok": False, "error": "Cross-site request blocked."}), 403

_sim_session = None  # single-user local tool -- one active simulation at a time


@app.route("/")
def index():
    return render_template("option_chain.html")


@app.route("/charts")
def charts_page():
    return render_template("charts.html")


@app.route("/iv")
def iv_page():
    return render_template("iv.html")


@app.route("/plan")
def plan_page():
    return render_template("plan.html")


@app.route("/api/plan")
def api_plan():
    import plan
    try:
        return jsonify({"ok": True, **plan.payload()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _sell_status(chain):
    """Premium-selling checklist (sell_rules.py) for the chain being shown;
    None if it cannot be computed -- never breaks the caller."""
    try:
        import sell_rules
        return sell_rules.status(chain)
    except Exception as e:
        print(f"[sell_rules] {e}")
        return None


@app.route("/api/iv/history")
def api_iv_history():
    import sell_rules
    try:
        d = sell_rules.history_payload()
        return jsonify({"ok": d is not None, **(d or {"error": "IV history not built yet"})})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/charts/meta")
def api_charts_meta():
    import charts
    try:
        return jsonify({"ok": True, **charts.meta(request.args["expiry"])})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/charts/data")
def api_charts_data():
    import charts
    a = request.args
    try:
        return jsonify({"ok": True, **charts.data(
            a["expiry"], interval=a.get("interval", "5"), mode=a.get("mode", "single"),
            strike=a.get("strike"), typ=a.get("type", "CE"),
            ce_strike=a.get("ce_strike"), pe_strike=a.get("pe_strike"),
            date_from=a.get("from") or None, date_to=a.get("to") or None)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/simulator")
def simulator_page():
    return render_template("simulator.html")


@app.route("/api/sim/expiries")
def api_sim_expiries():
    try:
        return jsonify({"ok": True, "expiries": simulator.list_expiries()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sim/dates")
def api_sim_dates():
    expiry = request.args.get("expiry")
    try:
        return jsonify({"ok": True, "dates": simulator.list_available_dates(expiry),
                         "week_start": simulator.week_start(expiry) if expiry else None,
                         "missing_dates": simulator.list_missing_weekdays(expiry),
                         "timeframes": simulator.AVAILABLE_TIMEFRAMES})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sim/start", methods=["POST"])
def api_sim_start():
    global _sim_session
    body = request.get_json(force=True)
    date_str = body.get("date")
    timeframe = int(body.get("timeframe", 1))
    expiry = body.get("expiry")
    end_date = body.get("end_date")  # default: run on through expiry
    added, topup_err = 0, None
    if expiry and date_str:
        # the bulk download covered ATM +/-10 for the expiry week only --
        # fetch any strikes an earlier start date needs (once; then cached on disk)
        try:
            import history_downloader
            added, topup_err = history_downloader.ensure_strikes(expiry, date_str)
            if added:
                simulator.forget(expiry)
        except Exception as e:
            topup_err = f"{type(e).__name__}: {e}"
    try:
        _sim_session = simulator.SimulationSession(date_str, timeframe_min=timeframe, expiry=expiry, end_date=end_date)
        return jsonify({"ok": True, "total_candles": _sim_session.total_candles, "gaps": _sim_session.gaps,
                         "days": [d.isoformat() for d in _sim_session.days], "positional": _sim_session.positional,
                         "added_strikes": added, "topup_error": topup_err,
                         **_sim_session.leg_bundle()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sim/next", methods=["POST"])
def api_sim_next():
    global _sim_session
    if _sim_session is None:
        return jsonify({"ok": False, "error": "No active session -- call /api/sim/start first."}), 400
    try:
        minutes = max(1, min(375, int((request.get_json(silent=True) or {}).get("minutes", 1))))
    except (TypeError, ValueError):
        minutes = 1
    snap = _sim_session.step(minutes)
    if snap is None:
        return jsonify({"ok": True, "done": True})
    return jsonify({"ok": True, "done": False, **snap, **_sim_session.leg_bundle()})


@app.route("/api/sim/eod", methods=["POST"])
def api_sim_eod():
    if _sim_session is None:
        return jsonify({"ok": False, "error": "No active session -- call /api/sim/start first."}), 400
    snap = _sim_session.eod()
    if snap is None:
        return jsonify({"ok": True, "done": True})
    return jsonify({"ok": True, "done": False, **snap, **_sim_session.leg_bundle()})


@app.route("/api/sim/next_day", methods=["POST"])
def api_sim_next_day():
    if _sim_session is None:
        return jsonify({"ok": False, "error": "No active session -- call /api/sim/start first."}), 400
    snap = _sim_session.next_day()
    if snap is None:
        return jsonify({"ok": True, "done": True})
    return jsonify({"ok": True, "done": False, **snap, **_sim_session.leg_bundle()})


@app.route("/api/sim/add_leg", methods=["POST"])
def api_sim_add_leg():
    if _sim_session is None:
        return jsonify({"ok": False, "error": "No active session -- start a simulation first."}), 400
    body = request.get_json(force=True)
    try:
        leg = _sim_session.add_leg(
            float(body["strike"]), body["type"], body["side"],
            int(body.get("qty", 1)), body.get("entry_price"),
        )
        return jsonify({"ok": True, "leg": leg, **_sim_session.leg_bundle()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/sim/remove_leg", methods=["POST"])
def api_sim_remove_leg():
    if _sim_session is None:
        return jsonify({"ok": False, "error": "No active session."}), 400
    body = request.get_json(force=True)
    try:
        _sim_session.remove_leg(body["id"])
        return jsonify({"ok": True, **_sim_session.leg_bundle()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/sim/reset_legs", methods=["POST"])
def api_sim_reset_legs():
    if _sim_session is None:
        return jsonify({"ok": False, "error": "No active session."}), 400
    _sim_session.reset_legs()
    return jsonify({"ok": True, **_sim_session.leg_bundle()})


@app.route("/api/sim/close_leg", methods=["POST"])
def api_sim_close_leg():
    if _sim_session is None:
        return jsonify({"ok": False, "error": "No active session."}), 400
    body = request.get_json(force=True)
    try:
        record = _sim_session.close_leg(body["id"], body.get("exit_price"))
        return jsonify({"ok": True, "record": record, **_sim_session.leg_bundle()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/sim/edit_leg", methods=["POST"])
def api_sim_edit_leg():
    if _sim_session is None:
        return jsonify({"ok": False, "error": "No active session."}), 400
    body = request.get_json(force=True)
    try:
        leg = _sim_session.edit_leg(
            body["id"], float(body["strike"]), body["type"], body["side"],
            int(body.get("qty", 1)), float(body["entry_price"]),
        )
        return jsonify({"ok": True, "leg": leg, **_sim_session.leg_bundle()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/expiries")
def api_expiries():
    try:
        expiries = chain_mod.list_expiries()
        return jsonify({"ok": True, "expiries": expiries})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chain")
def api_chain():
    expiry_ts = request.args.get("expiry", "")
    strikecount = int(request.args.get("strikecount", 10))
    include_greeks = request.args.get("greeks", "") == "1"
    try:
        data = chain_mod.get_chain(strikecount=strikecount, expiry_timestamp=expiry_ts, include_greeks=include_greeks)
        stats = _chain_stats_with_iv(data)
        rv, view = None, None
        try:
            import market_view
            rv = market_view.range_for_chain(data, data.get("resolved_expiry_ts"))
            idx = chain_mod.get_quotes([chain_mod.INDEX_SYMBOL]).get(chain_mod.INDEX_SYMBOL) or {}
            view = market_view.market_view(data, stats, rv, idx.get("chp"))
        except Exception as e:
            print(f"[api_chain] range/market view failed: {e}")
        return jsonify({"ok": True, **data, "market": mc.market_state(), "lot_size": manual_trades.LOT_SIZE,
                        "stats": stats, "range": rv, "view": view, "sell": _sell_status(data)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _chain_stats_with_iv(data):
    """chain_stats() plus live ATM IV and where it sits in the last year's
    range (IV Rank / Percentile from NSE bhavcopy history)."""
    stats = chain_mod.chain_stats(data)
    if not stats:
        return stats
    try:
        import eod_analysis
        import greeks as g
        import time as _t
        rows = data["strikes"]
        spot, exp_ts = data["spot"], data.get("resolved_expiry_ts")
        near = sorted(rows, key=lambda r: abs(r["strike"] - spot))[:6]
        fwd = g.synthetic_forward([(r["strike"], (r["ce"] or {}).get("ltp"), (r["pe"] or {}).get("ltp")) for r in near]) or spot
        T = (exp_ts - _t.time()) / (365 * 86400) if exp_ts else 0
        atm = min(rows, key=lambda r: abs(r["strike"] - fwd))
        ivs = [g.implied_vol_fwd((atm[k] or {}).get("ltp"), fwd, atm["strike"], T, g.RISK_FREE_RATE, k == "ce")
               for k in ("ce", "pe")] if T > 0 else []
        ivs = [v for v in ivs if v]
        if ivs:
            stats["atm_iv"] = round(sum(ivs) / len(ivs) * 100, 2)
            stats.update(eod_analysis.iv_rank(stats["atm_iv"]))
            # support / resistance: delta-weighted OI walls near the market
            import oi_walls
            w = oi_walls.walls({r["strike"]: (r["ce"] or {}).get("oi") for r in rows},
                               {r["strike"]: (r["pe"] or {}).get("oi") for r in rows},
                               spot, fwd=fwd, T=T, iv_pct=stats["atm_iv"], straddle=stats.get("atm_straddle"))
            stats.update(resistance=w["resistance"], support=w["support"], walls=w,
                         max_ce_oi_strike=w["resistance"], max_pe_oi_strike=w["support"])
    except Exception:
        pass
    return stats


@app.route("/strategy")
def strategy_page():
    return render_template("strategy.html")


_chain_price_map = manual_trades.chain_price_map
_price_lookup_for_legs = manual_trades.price_lookup_for_legs


@app.route("/api/manual/positions")
def api_manual_positions():
    try:
        try:
            manual_trades.migrate_leg_expiries()  # no-op once every open leg has a concrete expiry
        except Exception:
            pass  # retried next poll; never block showing positions over it
        all_positions = manual_trades.load_positions()
        if all_positions:
            lookup, spot, expiry_ts_map, forward_map = _price_lookup_for_legs(all_positions)
        else:
            lookup, spot = _chain_price_map("")
            expiry_ts_map, forward_map = {}, {}
        # Priced over EVERY open leg regardless of the "viewing" filter
        # below -- checkAutoClose() on the frontend must keep watching
        # SL/Target/15:25 square-off on strategies not currently in view,
        # not just whichever one is on screen.
        watch_positions = manual_trades.positions_with_live(all_positions, lookup)

        view_batch = request.args.get("batch_id", "")
        if view_batch == "__draft__":
            raw_positions = [p for p in all_positions if not p.get("batch_id")]
        elif view_batch:
            raw_positions = [p for p in all_positions if p.get("batch_id") == view_batch]
        else:
            raw_positions = all_positions
        if view_batch:
            raw_ids = {p["id"] for p in raw_positions}
            positions = [p for p in watch_positions if p["id"] in raw_ids]
        else:
            positions = watch_positions

        # Legs that have already been booked/closed but whose strategy is
        # still partially open (e.g. one leg booked as an adjustment) --
        # that locked-in P&L must be folded into the payoff curve and
        # totals, or the chart/stats only ever show the remaining legs'
        # shape and silently drop the profit/loss already realized.
        closed_legs = []
        realized_pnl = None  # None (not 0) only when there's truly nothing open to scope this to
        if view_batch == "__draft__":
            history = manual_trades.load_history()
            closed_legs = [h for h in history if not h.get("batch_id")]
            realized_pnl = round(sum(h["pnl"] for h in closed_legs), 2)
        elif view_batch:
            history = manual_trades.load_history()
            closed_legs = [h for h in history if h.get("batch_id") == view_batch]
            realized_pnl = round(sum(h["pnl"] for h in closed_legs), 2)
        else:
            # "All Open Positions" -- fold in realized P&L from every batch
            # that currently has at least one leg still open, so this view
            # agrees with "Viewing: <that strategy>" when only one strategy
            # is open (the exact mismatch a user hit: the default "All"
            # view showed no booked profit while the specific-batch view
            # did). Batches that are ALREADY fully closed are deliberately
            # excluded -- that P&L is done and belongs to Journal/history,
            # not to "what's currently open".
            open_batch_ids = {p.get("batch_id") for p in all_positions if p.get("batch_id")}
            if open_batch_ids:
                history = manual_trades.load_history()
                closed_legs = [h for h in history if h.get("batch_id") in open_batch_ids]
                realized_pnl = round(sum(h["pnl"] for h in closed_legs), 2)

        curve = manual_trades.payoff_curve(raw_positions, spot) if spot else []
        if realized_pnl:
            curve = [{"price": c["price"], "pnl": round(c["pnl"] + realized_pnl, 2)} for c in curve]
        todays_curve = manual_trades.compute_todays_curve(
            raw_positions, spot, lookup, expiry_ts_map, forward_map, [c["price"] for c in curve]
        ) if curve else None
        if realized_pnl and todays_curve:
            todays_curve = [{"price": c["price"], "pnl": round(c["pnl"] + realized_pnl, 2)} for c in todays_curve]
        stats = manual_trades.summary(raw_positions, curve, realized_pnl=realized_pnl or 0.0)
        live_total = sum(p["live_pnl"] for p in positions if p["live_pnl"] is not None)
        greeks = manual_trades.greeks_lookup_for_legs(raw_positions) if raw_positions else {}
        total_pnl = round(live_total + realized_pnl, 2) if realized_pnl is not None else None
        # Estimated costs: booked legs' recorded charges + what the open legs
        # would pay in total if closed right now. Net = gross - charges.
        booked_charges = sum(h.get("charges") if h.get("charges") is not None else
                             charges_mod.round_trip(h["side"], h["entry_price"], h["exit_price"],
                                                    manual_trades.lot_of(h) * h["qty"], h.get("exit_reason"))
                             for h in closed_legs)
        open_charges = sum(p["est_charges"] for p in positions if p.get("est_charges") is not None)
        est_charges = round(booked_charges + open_charges, 2)
        net_pnl = round((total_pnl if total_pnl is not None else live_total) - est_charges, 2)
        return jsonify({"ok": True, "positions": positions, "watch_positions": watch_positions,
                         "spot": spot, "curve": curve, "todays_curve": todays_curve,
                         "stats": stats, "live_pnl_total": round(live_total, 2),
                         "realized_pnl": realized_pnl, "total_pnl": total_pnl,
                         "closed_legs": closed_legs, "greeks": greeks,
                         "est_charges": est_charges, "net_pnl": net_pnl, "lot_size": manual_trades.LOT_SIZE,
                         "market": mc.market_state(),
                         "monitor": manual_monitor.status()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/preview", methods=["POST"])
def api_manual_preview():
    """Payoff / today's curve / stats / Greeks for hypothetical legs (the
    basket) -- computed exactly like a saved strategy, but nothing is saved."""
    body = request.get_json(force=True) or {}
    try:
        legs = [l for l in (body.get("legs") or []) if l.get("price")]
        if not legs:
            return jsonify({"ok": False, "error": "The basket is empty."}), 400
        raw = manual_trades.preview_positions(legs)
        lookup, spot, expiry_ts_map, forward_map = _price_lookup_for_legs(raw)
        positions = manual_trades.positions_with_live(raw, lookup)
        curve = manual_trades.payoff_curve(raw, spot) if spot else []
        todays = manual_trades.compute_todays_curve(raw, spot, lookup, expiry_ts_map, forward_map,
                                                    [c["price"] for c in curve]) if curve else None
        stats = manual_trades.summary(raw, curve, realized_pnl=0.0)
        live_total = sum(p["live_pnl"] for p in positions if p["live_pnl"] is not None)
        est = round(sum(p["est_charges"] for p in positions if p.get("est_charges") is not None), 2)
        return jsonify({"ok": True, "positions": positions, "spot": spot, "curve": curve, "todays_curve": todays,
                        "stats": stats, "greeks": manual_trades.greeks_lookup_for_legs(raw),
                        "live_pnl_total": round(live_total, 2), "realized_pnl": 0.0, "total_pnl": round(live_total, 2),
                        "est_charges": est, "net_pnl": round(live_total - est, 2), "lot_size": manual_trades.LOT_SIZE})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/delete_strategy", methods=["POST"])
def api_manual_delete_strategy():
    """Deletes a strategy entered by mistake (open legs + booked legs), no P&L
    booked; a copy of every removed record goes to results/deleted_trades.jsonl."""
    body = request.get_json(force=True) or {}
    try:
        n_open, n_hist = manual_trades.delete_strategy(body.get("batch_id"))
        return jsonify({"ok": True, "open_removed": n_open, "history_removed": n_hist})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/manual/add_leg", methods=["POST"])
def api_manual_add_leg():
    body = request.get_json(force=True)
    try:
        entry_price = body.get("entry_price")
        if entry_price is None:
            entry_price, _basis = manual_trades.fill_price(body["symbol"], body["side"])
            if entry_price is None:
                return jsonify({"ok": False, "error": "Could not fetch a live price for this symbol."}), 400
        leg = manual_trades.add_leg(body["symbol"], float(body["strike"]), body["type"], body["side"],
                                     int(body.get("qty", 1)), float(entry_price), body.get("expiry", ""),
                                     trade_type=body.get("trade_type", "INTRADAY"),
                                     batch_id=body.get("batch_id"))
        return jsonify({"ok": True, "leg": leg})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/add_basket", methods=["POST"])
def api_manual_add_basket():
    body = request.get_json(force=True)
    try:
        legs_spec = body["legs"]
        if len(legs_spec) < 1:
            return jsonify({"ok": False, "error": "The basket is empty."}), 400
        expiry = body.get("expiry", "")
        for spec in legs_spec:
            if spec.get("entry_price") is None:
                price, _basis = manual_trades.fill_price(spec["symbol"], spec["side"])
                if price is None:
                    return jsonify({"ok": False, "error": f"No live price for {spec['symbol']}."}), 400
                spec["entry_price"] = price
        batch_id, new_legs = manual_trades.add_basket(legs_spec, trade_type=body.get("trade_type", "INTRADAY"),
                                                      name=body.get("name"))
        return jsonify({"ok": True, "batch_id": batch_id, "legs": new_legs})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/manual/rename_strategy", methods=["POST"])
def api_manual_rename_strategy():
    body = request.get_json(force=True) or {}
    try:
        manual_trades.rename_strategy(body["batch_id"], body.get("name"))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/manual/open_batches")
def api_manual_open_batches():
    try:
        data = manual_trades.list_open_batches()
        return jsonify({"ok": True, "batches": data["batches"], "has_draft": data["has_draft"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/set_risk", methods=["POST"])
def api_manual_set_risk():
    body = request.get_json(force=True)
    try:
        # Current LTP, so a wrong-side SL/target (one that would trigger on
        # the very next poll) is rejected instead of silently accepted.
        leg_now = next((p for p in manual_trades.load_positions() if p["id"] == body["id"]), None)
        ref_price = None
        if leg_now is not None:
            try:
                lookup, _, _, _ = _price_lookup_for_legs([leg_now])
                ref_price = lookup.get(leg_now["symbol"])
            except Exception:
                pass
        leg = manual_trades.update_leg_risk(
            body["id"],
            sl=body.get("sl"), target=body.get("target"),
            clear_sl=bool(body.get("clear_sl")), clear_target=bool(body.get("clear_target")),
            ref_price=ref_price,
        )
        if leg is None:
            return jsonify({"ok": False, "error": "Leg not found."}), 404
        return jsonify({"ok": True, "leg": leg})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/set_type", methods=["POST"])
def api_manual_set_type():
    body = request.get_json(force=True)
    try:
        leg = manual_trades.set_trade_type(body["id"], body["trade_type"])
        if leg is None:
            return jsonify({"ok": False, "error": "Leg not found."}), 404
        return jsonify({"ok": True, "leg": leg})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/close_leg", methods=["POST"])
def api_manual_close_leg():
    body = request.get_json(force=True)
    try:
        exit_price = body.get("exit_price")
        if exit_price is None:
            positions = {p["id"]: p for p in manual_trades.load_positions()}
            leg = positions.get(body["id"])
            if leg is None:
                return jsonify({"ok": False, "error": "Leg not found."}), 404
            lookup, _ = _chain_price_map(leg.get("expiry", ""))
            exit_price = lookup.get(leg["symbol"])
            if exit_price is None:
                return jsonify({"ok": False, "error": "Could not fetch a live price to close at."}), 400
        reason = body.get("reason") if body.get("reason") in ("SL", "TARGET", "SQUARE_OFF") else "MANUAL"
        record = manual_trades.close_leg(body["id"], float(exit_price), exit_reason=reason)
        if record is None:
            return jsonify({"ok": False, "error": "Leg not found."}), 404
        return jsonify({"ok": True, "record": record})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/close_all", methods=["POST"])
def api_manual_close_all():
    body = request.get_json(force=True) or {}
    try:
        raw_positions = manual_trades.load_positions()
        lookup, _, _, _ = _price_lookup_for_legs(raw_positions) if raw_positions else ({}, None, {}, {})
        records, skipped = manual_trades.close_all(lookup, batch_filter=body.get("batch_id") or None)
        return jsonify({"ok": True, "records": records, "skipped": skipped})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/quote")
def api_quote():
    """Order-ticket data for one contract: LTP, bid/ask, the paper fill
    price for the chosen side, and the spread."""
    symbol, side = request.args.get("symbol", ""), request.args.get("side", "BUY")
    try:
        q = chain_mod.get_quotes([symbol]).get(symbol)
        if not q:
            return jsonify({"ok": False, "error": "No quote for this symbol."}), 404
        price, basis = manual_trades.fill_price(symbol, side)
        return jsonify({"ok": True, **q, "fill_price": price, "fill_basis": basis,
                        "lot_size": manual_trades.LOT_SIZE, "market": mc.market_state()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


_ideas_cache = {}


@app.route("/api/ideas")
def api_ideas():
    """Auto-built strategies for one expiry with risk/reward + probability
    (strategy_ideas.py). Cached 60 s per expiry."""
    import time as _t
    import market_view
    import strategy_ideas
    expiry_ts = request.args.get("expiry", "")
    hit = _ideas_cache.get(expiry_ts)
    if hit and _t.time() - hit[0] < 60:
        return jsonify(hit[1])
    try:
        chain = chain_mod.get_chain(strikecount=30, expiry_timestamp=expiry_ts)
        stats = _chain_stats_with_iv(chain)
        rv = view = None
        try:
            rv = market_view.range_view(chain, chain.get("resolved_expiry_ts"))
            idx = chain_mod.get_quotes([chain_mod.INDEX_SYMBOL]).get(chain_mod.INDEX_SYMBOL) or {}
            view = market_view.market_view(chain, stats, rv, idx.get("chp"))
        except Exception as e:
            print(f"[api_ideas] range/market view failed: {e}")
        rng = ((rv or {}).get("official") or (rv or {}).get("live")) if (rv or {}).get("for_viewed_expiry", True) else None
        if rng is None:
            # not the expiry the range engine tracks (e.g. next week's): same
            # rules on this expiry's own leaders
            try:
                rng = market_view.expiry_range(chain)
            except Exception as e:
                print(f"[api_ideas] expiry range failed: {e}")
        res = strategy_ideas.build(chain, stats, rng, view)
        res.pop("ctx", None)
        out = {"ok": True, **res, "market": mc.market_state(), "sell": _sell_status(chain)}
        _ideas_cache[expiry_ts] = (_t.time(), out)
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


_track_trades = {"mtime": None, "df": None}


@app.route("/api/ideas/track")
def api_ideas_track():
    """Year-by-year + worst trades of one template at one days-to-expiry /
    entry time, from the 5-year template backtest (template_backtest.py)."""
    import os as _os
    import pandas as _pd
    import paths as _paths
    path = _os.path.join(_paths.BASE_DIR, "results", "template_bt", "trades.parquet")
    try:
        m = _os.path.getmtime(path)
        if _track_trades["mtime"] != m:
            _track_trades.update(mtime=m, df=_pd.read_parquet(path))
        t = _track_trades["df"]
        x = t[(t.tpl == request.args["tpl"]) & (t.dte == int(request.args["dte"])) & (t.slot == request.args["slot"])]
        x = x.assign(year=x.expiry.str[:4]).sort_values("expiry")
        years = [{"year": y, "n": int(len(g)), "win": round(float((g.hold_rs > 0).mean() * 100), 1),
                  "avg": round(float(g.hold_rs.mean())), "sum": round(float(g.hold_rs.sum())),
                  "avg_man": round(float(g.managed_rs.mean()))} for y, g in x.groupby("year")]
        worst = x.nsmallest(5, "hold_rs")[["expiry", "day", "premium_pts", "hold_rs", "managed_rs", "managed_exit"]].to_dict("records")
        eq = [[r.expiry, round(float(c))] for r, c in zip(x.itertuples(), x.hold_rs.cumsum())]
        eqm = [round(float(c)) for c in x.managed_rs.cumsum()]
        return jsonify({"ok": True, "years": years, "worst": worst, "equity": eq, "equity_managed": eqm})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/view_ideas", methods=["POST"])
def api_view_ideas():
    """My View: the user's view -> best defined-risk structures (view_ideas.py)."""
    import view_ideas
    body = request.get_json(force=True) or {}
    try:
        chain = chain_mod.get_chain(strikecount=30, expiry_timestamp=body.get("expiry", ""))
        return jsonify({"ok": True, **view_ideas.build(chain, body.get("view") or {})})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/margin", methods=["POST"])
def api_margin():
    """Margin + estimated round-trip charges for a proposed basket:
    body {legs: [{symbol, side, qty (lots), price}]}."""
    body = request.get_json(force=True) or {}
    legs = body.get("legs") or []
    try:
        m = chain_mod.get_margin(legs, manual_trades.LOT_SIZE)
        est = sum(charges_mod.round_trip(l["side"], float(l.get("price") or 0), float(l.get("price") or 0),
                                         manual_trades.LOT_SIZE * int(l["qty"])) for l in legs)
        return jsonify({"ok": True, **m, "est_round_trip_charges": round(est, 2)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/scenario")
def api_manual_scenario():
    """Scenario grid for the positions in view (same batch filter as
    /api/manual/positions): ?batch_id=&days=N."""
    try:
        view_batch = request.args.get("batch_id", "")
        days = max(0.0, float(request.args.get("days", 0) or 0))
        all_positions = manual_trades.load_positions()
        if view_batch == "__draft__":
            legs = [p for p in all_positions if not p.get("batch_id")]
        elif view_batch:
            legs = [p for p in all_positions if p.get("batch_id") == view_batch]
        else:
            legs = all_positions
        if not legs:
            return jsonify({"ok": True, "grid": None})
        lookup, spot, exp_map, fwd_map = _price_lookup_for_legs(legs)
        realized = 0.0
        if view_batch:
            realized = sum(h["pnl"] for h in manual_trades.load_history()
                           if (h.get("batch_id") or "__draft__") == view_batch)
        grid = manual_trades.scenario_grid(legs, spot, lookup, exp_map, fwd_map, days_forward=days, realized_pnl=realized)
        return jsonify({"ok": True, "grid": grid, "spot": spot})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/alerts", methods=["GET", "POST"])
def api_alerts():
    import alerts
    try:
        if request.method == "POST":
            body = request.get_json(force=True) or {}
            if body.get("delete"):
                alerts.delete(body["delete"])
            else:
                alerts.add(body["kind"], body["value"], body.get("symbol"), body.get("note", ""))
        rows = alerts.load()
        return jsonify({"ok": True, "alerts": [{**a, "label": alerts.describe(a)} for a in rows]})
    except (ValueError, KeyError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/range")
def range_page():
    return render_template("range.html")


@app.route("/api/range/state")
def api_range_state():
    import range_strategy
    try:
        with open(range_strategy.STATE_PATH) as f:
            return jsonify({"ok": True, "state": json.load(f)})
    except FileNotFoundError:
        return jsonify({"ok": True, "state": None})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/range/refresh", methods=["POST"])
def api_range_refresh():
    import range_strategy
    try:
        return jsonify({"ok": True, "state": range_strategy.run_once(notify=None)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


_range_bt_cache = {"at": 0, "rows": None}


@app.route("/api/range/backtest")
def api_range_backtest():
    import time as _t
    import range_strategy
    try:
        if _range_bt_cache["rows"] is None or _t.time() - _range_bt_cache["at"] > 600:
            _range_bt_cache.update(at=_t.time(), rows=range_strategy.backtest_all())
        return jsonify({"ok": True, "rows": _range_bt_cache["rows"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/diary")
def diary_page():
    return render_template("diary.html")


@app.route("/api/diary")
def api_diary_list():
    import range_strategy
    try:
        return jsonify({"ok": True, "cycles": range_strategy.list_diaries()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/diary/day/<day>")
def api_diary_day(day):
    import range_strategy
    try:
        if day == "today":
            day = mc.now_ist().date().isoformat()
        return jsonify({"ok": True, **range_strategy.day_detail(day)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/diary/expiry/<expiry>")
def api_diary_expiry(expiry):
    import range_strategy
    try:
        if expiry == "current":
            cycles = range_strategy.list_diaries()
            if not cycles:
                return jsonify({"ok": True, "days": [], "cycles": []})
            expiry = cycles[0]["expiry"]
        v = range_strategy.expiry_view(expiry)
        if v is None:
            return jsonify({"ok": False, "error": "No recorded data for that expiry."}), 404
        return jsonify({"ok": True, **v})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/diary/<expiry>")
def api_diary(expiry):
    import range_strategy
    try:
        d = range_strategy.get_diary(expiry)
        if d is None:
            return jsonify({"ok": False, "error": "No recorded data for that week."}), 404
        return jsonify({"ok": True, **d})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/market_state")
def api_market_state():
    return jsonify({"ok": True, **mc.market_state()})


@app.route("/api/monitor_status")
def api_monitor_status():
    return jsonify({"ok": True, **manual_monitor.status()})


@app.route("/api/manual/history")
def api_manual_history():
    try:
        history = manual_trades.load_history()
        view_batch = request.args.get("batch_id", "")
        if view_batch == "__draft__":
            history = [h for h in history if not h.get("batch_id")]
        elif view_batch:
            history = [h for h in history if h.get("batch_id") == view_batch]
        # Optional single-day filter (YYYY-MM-DD) -- the Journal-style
        # "closed trade history" list otherwise only ever grows, so the
        # frontend defaults this to today and lets the user pick an older
        # date from a calendar control to browse further back.
        date_filter = request.args.get("date", "")
        if date_filter:
            history = [h for h in history if (h.get("exit_time") or "").startswith(date_filter)]
        return jsonify({"ok": True, "history": history})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/save_strategy", methods=["POST"])
def api_manual_save_strategy():
    body = request.get_json(force=True) or {}
    try:
        batch_id = manual_trades.save_strategy(leg_ids=body.get("leg_ids"), name=body.get("name"))
        if batch_id is None:
            return jsonify({"ok": False, "error": "No unsaved open legs to save."}), 400
        return jsonify({"ok": True, "batch_id": batch_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/data-health")
def data_health_page():
    return render_template("data_health.html")


@app.route("/api/data_health")
def api_data_health():
    try:
        return jsonify({"ok": True, **data_health.get_health_report()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/data_health/backfill", methods=["POST"])
def api_data_health_backfill():
    try:
        import daily_history
        status = daily_history.update()
        return jsonify({"ok": True, "status": status})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/journal")
def journal_page():
    return render_template("journal.html")


@app.route("/api/manual/journal")
def api_manual_journal():
    try:
        lookup = {}
        try:
            open_legs = manual_trades.load_positions()
            if open_legs:
                lookup, _, _, _ = _price_lookup_for_legs(open_legs)
        except Exception:
            pass  # journal still works with realized P&L only if the live chain call fails
        return jsonify({"ok": True, "journal": manual_trades.get_journal(lookup)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/manual/journal_note", methods=["POST"])
def api_manual_journal_note():
    body = request.get_json(force=True)
    try:
        manual_trades.save_journal_note(body["batch_id"], body.get("note", ""))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    import broker
    print(f"Logging in to {broker.name()} (automated TOTP flow)...")
    broker.login()  # daily token refresh -- a no-op while today's cached token is valid
    print(f"{broker.name()} login OK.")
    # (historical data now comes from the evening job -- daily_history.py;
    # the old intraday recorder thread is no longer started)
    # Server-side SL/Target/15:25 square-off/expiry settlement -- runs
    # whether or not any browser tab is open. See manual_monitor.py.
    manual_monitor.start_background_monitor()
    print("Risk monitor running in the background.")
    import range_strategy
    range_strategy.start_background()  # the user's weekly range method, paper-traded
    import plan
    plan.start_background()            # "Aaj ka Plan": records + follows the plan's trades
    print("Range Strategy running in the background. Starting server...")
    # threaded=True is essential here, not optional: every page (Option
    # Chain, Strategy Builder, Journal) polls its own endpoint every few
    # seconds, and each of those calls out to Fyers over the network. The
    # default single-threaded dev server handles one request at a time, so
    # those calls queue up behind each other -- confirmed live as exactly
    # the "switching tabs feels laggy" symptom, since a page navigation's
    # request sat behind whatever other tab's in-flight poll was still
    # waiting on Fyers.
    # use_reloader=False is essential here -- Werkzeug's file-watcher walks
    # every imported module's path, and this machine's Python environment
    # is a shared one with numpy/pandas/torch/jupyter/pytest etc. installed
    # globally. Something keeps touching those unrelated site-packages
    # files' mtimes, so the reloader saw it as "code changed" and restarted
    # the whole server every 1-2 seconds -- wiping the auth-token and chain
    # caches constantly and abruptly killing in-flight requests (confirmed
    # live: a mid-restart fetch failed outright). debug=True is kept for
    # readable tracebacks; just the auto-restart-on-file-change is off, so
    # a code change now needs a manual restart of `python app.py`.
    # debug=False: the Werkzeug debugger allows code execution from the
    # browser and shouldn't be on in an app that holds broker credentials.
    app.run(debug=False, port=5057, threaded=True, use_reloader=False)
