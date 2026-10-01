"""straddle_prevclose_bt.py -- intraday short ATM straddle, entered with a
LIMIT at the previous day's closing straddle price.

Rules (per trading day D, the nearest weekly expiry):
  strike   the 50-pt strike nearest NIFTY's close on the previous trading day P
  limit    CE close + PE close of that strike at P's last minute
  entry    on D from 09:15 until ENTRY_CUTOFF: sold as soon as the combined
           1-min close reaches the limit (at the 09:15 candle a gap above the
           limit fills at the open; later crossings fill at the limit)
  stop     combined premium >= (1 + SL) x the sold premium, on 1-min closes
  exit     at a fixed clock time (grid below) if not stopped
Costs: 0.5 pt slippage per order + all charges; 1 lot of today's size (65).

Output: results/straddle_prevclose/report.html (+ trades.parquet)
"""
import json
import os
import sys
from datetime import time as dtime

import numpy as np
import pandas as pd

import charges
import manual_trades
import paths
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "straddle_prevclose")
LOT = manual_trades.LOT_SIZE
SLIP = 0.5
ENTRY_CUTOFF = "14:30"
EXITS = ["09:45", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:15", "15:25"]
SLS = [None, 0.3, 0.5]
TARGETS = [None, 0.2, 0.3, 0.4, 0.5]


def _hhmm(ts):
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert(S.IST).strftime("%H:%M")


def collect(years=5):
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") in ("done", "live"))
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=int(365.25 * years))).date().isoformat()
    sp = S._load_spot()
    sp_day = sp.index.date
    spot_close = sp.groupby(sp_day)["close"].last()
    day_open = sp.groupby(sp_day)["open"].first()
    days_all = sorted(spot_close.index)
    hm = sp.index.strftime("%H:%M")
    s920_by_day = dict(zip(sp_day[hm == "09:20"], sp.close.to_numpy()[hm == "09:20"]))
    out = []
    prev_exp = None
    for exp in exps:
        if exp < cutoff:
            prev_exp = exp
            continue
        df = S._load_options(exp)
        mins = df["date"].dt.hour * 60 + df["date"].dt.minute
        df = df[mins <= 15 * 60 + 40]
        exp_d = pd.Timestamp(exp).date()
        lo = pd.Timestamp(prev_exp).date() if prev_exp else None
        for D in sorted(df["day"].unique()):
            if (lo and D <= lo) or D > exp_d:
                continue
            i = days_all.index(D) if D in spot_close.index else None
            if not i:
                continue
            P = days_all[i - 1]
            k = int(round(spot_close[P] / 50) * 50)
            legs = {}
            for typ in ("CE", "PE"):
                x = df[(df.strike == k) & (df.type == typ)]
                xp, xd = x[x.day == P], x[x.day == D]
                if xp.empty or xd.empty:
                    break
                legs[typ] = (float(xp.sort_values("ts").close.iloc[-1]), xd.sort_values("ts"))
            if len(legs) < 2:
                continue
            limit = legs["CE"][0] + legs["PE"][0]
            # minute grid of D, forward-filled per leg
            ts = np.union1d(legs["CE"][1].ts.to_numpy(), legs["PE"][1].ts.to_numpy())
            px, op = {}, {}
            for typ, (_, xd) in legs.items():
                s_ = pd.Series(xd.close.to_numpy(), index=xd.ts.to_numpy()).reindex(ts).ffill()
                px[typ] = s_.to_numpy()
                op[typ] = float(xd.open.iloc[0])
            comb = px["CE"] + px["PE"]
            hh = np.array([_hhmm(t) for t in ts])
            ok = np.isfinite(comb) & (hh <= ENTRY_CUTOFF)
            hit = np.flatnonzero(ok & (comb >= limit))
            first_open = op["CE"] + op["PE"]
            # baseline 2: the ATM straddle at 09:20 (strike nearest NIFTY then)
            atm = None
            if D in s920_by_day:
                ka = int(round(float(s920_by_day[D]) / 50) * 50)
                xa = {t_: df[(df.strike == ka) & (df.type == t_) & (df.day == D)].sort_values("ts") for t_ in ("CE", "PE")}
                if not xa["CE"].empty and not xa["PE"].empty:
                    sa = (pd.Series(xa["CE"].close.to_numpy(), index=xa["CE"].ts.to_numpy()).reindex(ts).ffill().to_numpy()
                          + pd.Series(xa["PE"].close.to_numpy(), index=xa["PE"].ts.to_numpy()).reindex(ts).ffill().to_numpy())
                    atm = np.round(sa, 2).tolist()
            rec = {"day": D.isoformat(), "hh": hh.tolist(), "comb": np.round(comb, 2).tolist(), "atm": atm, "expiry": exp, "dte_is0": D == exp_d, "strike": k, "limit": round(limit, 2),
                   "spot_prev": float(spot_close[P]), "open_comb": round(first_open, 2), "gap": round(float(day_open[D] - spot_close[P]), 1) if D in day_open.index else None}
            if not len(hit):
                out.append({**rec, "filled": False})
                continue
            e = int(hit[0])
            sold = max(limit, first_open) if hh[e] == hh[0] and first_open >= limit else limit
            fill = sold - 2 * SLIP
            out.append({**rec, "filled": True, "entry_time": hh[e], "sold": round(sold, 2), "fill": round(fill, 2),
                        "entry_i": e})
        prev_exp = exp
    return out


def entry(r, mode):
    """(entry index, net sold premium, series) for one day under an entry mode:
    limit = the previous-close limit rule; open = the same strike sold at
    09:16 whatever the price; atm = today's ATM straddle at 09:20."""
    hh = r["hh"]
    if mode == "limit":
        return (r["entry_i"], r["fill"], r["comb"]) if r["filled"] else None
    ser = r["comb"] if mode == "open" else r["atm"]
    if ser is None:
        return None
    t0 = "09:16" if mode == "open" else "09:20"
    e = next((i for i, h in enumerate(hh) if h >= t0 and ser[i] is not None and np.isfinite(ser[i])), None)
    return None if e is None else (e, ser[e] - 2 * SLIP, ser)


def simulate(rows, exit_t, sl, tgt, mode="limit"):
    res = []
    for r in rows:
        en = entry(r, mode)
        if en is None or r["hh"][en[0]] >= exit_t:
            continue
        e, fill, ser = en
        path = np.array(ser[e + 1:], dtype=float)
        pt = np.array(r["hh"][e + 1:])
        live = (pt <= exit_t) & np.isfinite(path)
        idx = np.flatnonzero(live)
        why, xi = "TIME", (idx[-1] if len(idx) else None)
        cand = []
        if sl is not None:
            h = np.flatnonzero(live & (path >= fill * (1 + sl)))
            if len(h):
                cand.append((h[0], "SL"))
        if tgt is not None:
            h = np.flatnonzero(live & (path <= fill * (1 - tgt)))
            if len(h):
                cand.append((h[0], "TGT"))
        if cand:
            xi, why = min(cand)
        buy = (path[xi] if xi is not None else fill) + 2 * SLIP
        pts = fill - buy
        half = fill / 2
        ch = charges.order_charges("SELL", half, LOT) * 2 + charges.order_charges("BUY", buy / 2, LOT) * 2
        res.append({"day": r["day"], "pts": pts, "rs": pts * LOT - ch, "why": why, "dte0": r["dte_is0"],
                    "entry_time": r["hh"][e], "gap_fill": r["hh"][e] == r["hh"][0], "gap": r["gap"]})
    return pd.DataFrame(res)


def summary(d):
    if d.empty:
        return None
    y = d.assign(year=d.day.str[:4]).groupby("year").rs.sum()
    eq = d.rs.cumsum()
    return {"n": int(len(d)), "win": round(float((d.rs > 0).mean() * 100), 1), "avg": round(float(d.rs.mean())),
            "total": round(float(d.rs.sum())), "worst": round(float(d.rs.min())), "best": round(float(d.rs.max())),
            "dd": round(float((eq - eq.cummax()).min())), "sl_rate": round(float((d.why == "SL").mean() * 100), 1),
            "tgt_rate": round(float((d.why == "TGT").mean() * 100), 1),
            "a": round(float(d[d.day < "2024-01-01"].rs.sum())), "b": round(float(d[d.day >= "2024-01-01"].rs.sum())),
            "years": {k: round(float(v)) for k, v in y.items()}}


def run():
    import pickle
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "days.pkl")
    if os.path.exists(cache) and "--fresh" not in sys.argv:
        rows = pickle.load(open(cache, "rb"))
    else:
        rows = collect()
        pickle.dump(rows, open(cache, "wb"))
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("hh", "comb", "atm")} for r in rows]).to_parquet(
        os.path.join(OUT, "trades.parquet"), index=False)
    grid = []
    for mode in ("limit", "open", "atm"):
        for sl in SLS:
            for ex in EXITS:
                s = summary(simulate(rows, ex, sl, None, mode))
                if s:
                    grid.append({"mode": mode, "sl": sl, "exit": ex, "tgt": None, **s})
    for tgt in TARGETS[1:]:
        for sl in SLS:
            s = summary(simulate(rows, "15:15", sl, tgt))
            if s:
                grid.append({"mode": "limit", "sl": sl, "exit": "15:15", "tgt": tgt, **s})
    return rows, grid


MODES = {"limit": "Your rule — limit at yesterday's closing straddle price",
         "open": "Same strike sold at 09:16 at any price (no limit)",
         "atm": "Today's ATM straddle sold at 09:20"}


def report(rows, grid):
    main = simulate(rows, "15:25", 0.5, None, "limit")
    eq = {m: [[d, round(float(v))] for d, v in zip(x.day, x.rs.cumsum())]
          for m, x in ((m, simulate(rows, "15:25", 0.5, None, m)) for m in MODES)}
    filled = [r for r in rows if r["filled"]]
    unf = simulate([r for r in rows if not r["filled"]], "15:25", 0.5, None, "open")
    fil_open = simulate(filled, "15:25", 0.5, None, "open")
    gaps = main.assign(g=pd.cut(main.gap.abs(), [-1, 25, 50, 100, 1e9], labels=["under 25", "25–50", "50–100", "over 100"]))
    data = {
        "grid": grid, "modes": MODES, "eq": eq,
        "fill": {"days": len(rows), "filled": len(filled), "gap": int(main.gap_fill.sum())},
        "split": [
            {"k": "Filled at 09:15 (gap open)", **summary(main[main.gap_fill])},
            {"k": "Filled later in the day", **summary(main[~main.gap_fill])},
            {"k": "Expiry day", **summary(main[main.dte0])},
            {"k": "Other days", **summary(main[~main.dte0])},
            *[{"k": f"NIFTY gap {g} pts", **summary(x)} for g, x in gaps.groupby("g", observed=True)],
        ],
        "why": {"unfilled_avg": round(float(unf.rs.mean())), "unfilled_n": int(len(unf)),
                "filled_open_avg": round(float(fil_open.rs.mean())), "filled_limit_avg": round(float(main.rs.mean()))},
        "years": {m: summary(simulate(rows, "15:25", 0.5, None, m))["years"] for m in MODES},
        "meta": {"first": rows[0]["day"], "last": rows[-1]["day"], "lot": LOT},
    }
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path, data


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Straddle at Yesterday's Close</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--best:#e9f8f1;--grp:#eef0f6;--accent:#4f5fe0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--best:#16302a;--grp:#23262e;--accent:#8a95ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
.rules{display:grid;grid-template-columns:120px 1fr;gap:6px 14px;font-size:13.5px}.rules b{color:var(--muted)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px;font-size:12.5px}
button{font:inherit;font-size:12.5px;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:8px;padding:4px 10px;cursor:pointer}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;margin-top:6px}.legend i{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle}
svg text{fill:var(--muted);font-size:11px}
</style></head><body><div class="wrap">
<h1>Short straddle at yesterday's closing price — intraday</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="card"><h2>Rules tested</h2><div class="rules">
<b>Strike</b><span>The 50-pt strike nearest NIFTY's close on the previous trading day (nearest weekly expiry).</span>
<b>Entry</b><span>Limit sell at yesterday's closing straddle price (CE + PE at the last minute). From 09:15 to 14:30: sold when the combined 1-min price reaches it. A gap open above it fills at the open. Not reached → no trade that day.</span>
<b>Stop loss</b><span>Combined premium up 50% from the sold price (1-min closes). 30% and no stop shown too.</span>
<b>Exit</b><span>Same day, at every clock time from 09:45 to 15:25 — to find the best one.</span>
<b>Costs</b><span>0.5 pt slippage per order, all charges, 1 lot (65).</span>
</div></div>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>What the test says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>Exit time × stop loss</h2>
<div class="bar"><span class="note">Stop loss:</span><button data-sl="0.5" class="on">50%</button><button data-sl="0.3">30%</button><button data-sl="none">No SL</button></div>
<div class="scroll"><table id="tExit"></table></div>
<p class="note">A trade filled after the exit time is not counted for that exit (so n grows through the morning). Best exit per rule shaded.</p></div>
<div class="card"><h2>Cumulative ₹ — exit 15:25, SL 50%</h2><div id="eq"></div><div class="legend" id="lg"></div></div>
<div class="card"><h2>By year — exit 15:25, SL 50%</h2><div class="scroll"><table id="tYear"></table></div></div>
<div class="card"><h2>Your rule — where the loss comes from (exit 15:25, SL 50%)</h2><div class="scroll"><table id="tSplit"></table></div></div>
<div class="card"><h2>Adding a profit target (exit 15:15)</h2><div class="scroll"><table id="tTgt"></table></div></div>
<div class="card"><h2>Limits</h2><ul>
<li>1-min closing prices (no bid/ask history); a real limit order could fill a little better when the price jumps through it within a minute.</li>
<li>The straddle's "closing price" is the last traded 1-min close of the previous day.</li>
<li>Naked straddle: the worst day here is not the worst possible.</li>
</ul></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
const G = (m, sl, ex, tgt = null) => D.grid.find(g => g.mode === m && String(g.sl) === String(sl) && g.exit === ex && String(g.tgt) === String(tgt));
const F = D.fill, W = D.why;
document.getElementById("meta").textContent = `${F.days} trading days (${D.meta.first} → ${D.meta.last}) · 1-min option prices · 0.5 pt slippage per order · all charges · 1 lot (${D.meta.lot}).`;
const L = G("limit", 0.5, "15:25"), O = G("open", 0.5, "15:25"), A = G("atm", 0.5, "15:25");
const kp = (v, l, c = "") => `<div class="kpi"><b class="${c}">${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${Math.round(F.filled / F.days * 100)}%`, `days the limit was reached (${F.filled} of ${F.days}); ${F.gap} of them at the 09:15 open`) +
  kp(rs(L.total), `your rule, 5 years (exit 15:25, SL 50%) · ${rs(L.avg)} a trade`, cl(L.total)) +
  kp(rs(O.total), `same strike, no limit (sold 09:16) · ${rs(O.avg)} a trade`, cl(O.total)) +
  kp(rs(A.total), `today's ATM straddle at 09:20 · ${rs(A.avg)} a trade`, cl(A.total));
const bestL = D.grid.filter(g => g.mode === "limit" && g.tgt == null).reduce((a, g) => g.avg > a.avg ? g : a);
document.getElementById("verdict").innerHTML = [
  `<b>The limit-at-yesterday's-close rule loses at every exit time and every stop.</b> Its best case (exit ${bestL.exit}, ${bestL.sl == null ? "no SL" : "SL " + bestL.sl * 100 + "%"}) is still ${rs(bestL.avg)} a trade. With SL 50% and exit 15:25: ${rs(L.total)} over 5 years — ${rs(L.a)} in 2021–23 and ${rs(L.b)} in 2024–26.`,
  `<b>Why: the limit picks the wrong days.</b> The straddle only gets back to yesterday's price when NIFTY has moved away from the strike enough to undo the overnight time decay — a gapping or trending day, the worst kind for a straddle seller. On the ${W.unfilled_n} days the limit was never reached, the same straddle sold at 09:16 made ${rs(W.unfilled_avg)} a day on average: the quiet days the rule skips are exactly the good ones. On the days it did fill, selling at 09:16 instead would have averaged ${rs(W.filled_open_avg)} vs ${rs(W.filled_limit_avg)} with the limit — the loss comes from which days are traded, not from the fill price.`,
  `<b>Big gap days do most of the damage:</b> days NIFTY opened more than 100 pts from yesterday's close lost ${rs(D.split.find(x => x.k.includes("over 100")).avg)} a trade (${rs(D.split.find(x => x.k.includes("over 100")).total)} of the ${rs(L.total)}). The limit fills at the open on those days, straight into the move.`,
  `<b>Selling without the limit is about break-even</b> (${rs(O.avg)} a trade for the same strike at 09:16, ${rs(A.avg)} for the ATM straddle at 09:20) — the intraday straddle premium is roughly fairly priced once charges and slippage are paid.`,
  `<b>Best exit time: as late as possible (15:25).</b> Every rule does best when held into the close; the earliest exits (09:45–10:00) are the worst.`,
  `<b>The 50% stop rarely fires</b> (${L.sl_rate}% of trades) and changes little; a 30% stop and profit targets of 20–50% do not rescue the rule either.`,
  `<b>The signal may point the other way:</b> days that reach yesterday's straddle price are the moving days. Buying the straddle on that trigger is an idea to test — not a result yet.`,
].map(x => `<li>${x}</li>`).join("");

function exitTable(sl) {
  const exits = [...new Set(D.grid.filter(g => g.tgt == null).map(g => g.exit))];
  const modes = Object.keys(D.modes);
  let h = `<tr><th>Exit</th>${modes.map(m => `<th colspan="4" style="text-align:center">${m === "limit" ? "Your rule (limit)" : m === "open" ? "Same strike, 09:16" : "ATM at 09:20"}</th>`).join("")}</tr>
    <tr><th></th>${modes.map(() => `<th>n</th><th>Win %</th><th>₹ / trade</th><th>5-yr ₹</th>`).join("")}</tr>`;
  const best = {};
  modes.forEach(m => { best[m] = exits.map(e => G(m, sl, e)).filter(Boolean).reduce((a, g) => g.avg > a.avg ? g : a).exit; });
  exits.forEach(e => {
    h += `<tr><td>${e}</td>${modes.map(m => { const g = G(m, sl, e); if (!g) return "<td colspan=4></td>";
      const bs = best[m] === e ? ' style="background:var(--best)"' : "";
      return `<td${bs}>${g.n}</td><td${bs}>${g.win}</td><td${bs} class="${cl(g.avg)}"><b>${rs(g.avg)}</b></td><td${bs} class="${cl(g.total)}">${rs(g.total)}</td>`; }).join("")}</tr>`;
  });
  document.getElementById("tExit").innerHTML = h;
}
exitTable(0.5);
document.querySelectorAll("[data-sl]").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("[data-sl]").forEach(x => x.classList.toggle("on", x === b));
  exitTable(b.dataset.sl === "none" ? null : +b.dataset.sl);
}));

const years = [...new Set(Object.values(D.years).flatMap(y => Object.keys(y)))].sort();
document.getElementById("tYear").innerHTML = `<tr><th>Rule</th>${years.map(y => `<th>${y}</th>`).join("")}<th>Total</th></tr>` +
  Object.entries(D.modes).map(([m, lab]) => `<tr><td>${lab}</td>${years.map(y => `<td class="${cl(D.years[m][y])}">${rs(D.years[m][y])}</td>`).join("")}<td class="${cl(G(m, 0.5, "15:25").total)}"><b>${rs(G(m, 0.5, "15:25").total)}</b></td></tr>`).join("");

document.getElementById("tSplit").innerHTML = `<tr><th>Group</th><th>n</th><th>Win %</th><th>₹ / trade</th><th>Total</th><th>Worst</th><th>SL hit %</th></tr>` +
  D.split.map(s => `<tr><td>${s.k}</td><td>${s.n}</td><td>${s.win}</td><td class="${cl(s.avg)}">${rs(s.avg)}</td><td class="${cl(s.total)}">${rs(s.total)}</td><td class="down">${rs(s.worst)}</td><td>${s.sl_rate}</td></tr>`).join("");

document.getElementById("tTgt").innerHTML = `<tr><th>Target (of premium)</th><th>Stop</th><th>n</th><th>Win %</th><th>Target hit %</th><th>SL hit %</th><th>₹ / trade</th><th>5-yr ₹</th><th>Max DD</th></tr>` +
  D.grid.filter(g => g.mode === "limit" && g.exit === "15:15").sort((a, b) => (a.tgt ?? 0) - (b.tgt ?? 0) || String(a.sl).localeCompare(String(b.sl)))
    .map(g => `<tr><td>${g.tgt == null ? "none" : g.tgt * 100 + "%"}</td><td>${g.sl == null ? "none" : g.sl * 100 + "%"}</td><td>${g.n}</td><td>${g.win}</td><td>${g.tgt_rate}</td><td>${g.sl_rate}</td><td class="${cl(g.avg)}">${rs(g.avg)}</td><td class="${cl(g.total)}">${rs(g.total)}</td><td class="down">${rs(g.dd)}</td></tr>`).join("");

(function () {
  const COL = {limit: "#e0434a", open: "#4f5fe0", atm: "#0e9f6e"};
  const S = Object.entries(D.eq).map(([m, pts]) => ({m, pts}));
  const W0 = 1300, H = 320, pl = 80, pr = 12, pt = 12, pb = 26;
  const all = S.flatMap(s => s.pts.map(p => p[0])).sort();
  const t0 = new Date(all[0]).getTime(), t1 = new Date(all[all.length - 1]).getTime();
  const vals = S.flatMap(s => s.pts.map(p => p[1])).concat([0]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = d => pl + (new Date(d).getTime() - t0) / (t1 - t0) * (W0 - pl - pr);
  const y = v => pt + (hi - v) / (hi - lo || 1) * (H - pt - pb);
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4 || 1))) * 2;
  const ticks = []; for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  const yrs = []; for (let yv = new Date(all[0]).getFullYear() + 1; yv <= new Date(all[all.length - 1]).getFullYear(); yv++) yrs.push(yv);
  let svg = `<svg viewBox="0 0 ${W0} ${H}" width="100%" role="img" aria-label="Cumulative P&L">`;
  svg += ticks.map(v => `<line x1="${pl}" x2="${W0 - pr}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"${v === 0 ? ' stroke-width="2"' : ""}/><text x="${pl - 6}" y="${y(v) + 4}" text-anchor="end">${rs(v)}</text>`).join("");
  svg += yrs.map(yv => { const xx = x(yv + "-01-01"); return `<line x1="${xx}" x2="${xx}" y1="${pt}" y2="${H - pb}" stroke="var(--border)" stroke-dasharray="3,3"/><text x="${xx + 4}" y="${H - 8}">${yv}</text>`; }).join("");
  svg += S.map(s => `<path d="${s.pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="${COL[s.m]}" stroke-width="1.8"/>`).join("");
  document.getElementById("eq").innerHTML = svg + "</svg>";
  document.getElementById("lg").innerHTML = S.map(s => `<span><i style="background:${COL[s.m]}"></i>${D.modes[s.m]} (${rs(s.pts[s.pts.length - 1][1])})</span>`).join("");
})();
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    rows, grid = run()
    f = [r for r in rows if r["filled"]]
    print("days", len(rows), "filled", len(f), f"{len(f) / len(rows) * 100:.0f}%")
    print(pd.Series([r["entry_time"] for r in f]).value_counts().sort_index().head(12).to_dict())
    for g in grid:
        print(g["mode"], g["sl"], g["exit"], g["tgt"], g["n"], g["win"], g["avg"], g["total"], g["dd"], g["sl_rate"], g["a"], g["b"])
    d = simulate(rows, "15:15", 0.5, None)
    print(d.groupby("gap_fill").rs.agg(["count", "mean", "sum"]).round(0))
    print(d.groupby("dte0").rs.agg(["count", "mean", "sum"]).round(0))
    print(report(rows, grid)[0])
