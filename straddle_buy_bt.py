"""straddle_buy_bt.py -- the reverse of straddle_prevclose_bt: BUY the ATM
straddle (strike nearest yesterday's NIFTY close) when its combined price
reaches yesterday's closing straddle price (a buy-stop), same day exit.

Uses the day cache built by straddle_prevclose_bt.py
(results/straddle_prevclose/days.pkl).
Output: results/straddle_prevclose/buy_report.html
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

import charges
import straddle_prevclose_bt as B

OUT = B.OUT
LOT, SLIP = B.LOT, B.SLIP
EXITS = B.EXITS
SLS = [None, 0.2, 0.3, 0.4]          # premium DOWN this much from the buy price
TGTS = [None, 0.2, 0.3, 0.5, 1.0]    # premium UP this much


def simulate(rows, exit_t, sl, tgt, only=None):
    """only: None | 'gap' (triggered at the 09:15 open) | 'intraday' (triggered later)."""
    res = []
    for r in rows:
        if not r["filled"] or r["hh"][r["entry_i"]] >= exit_t:
            continue
        gap = r["hh"][r["entry_i"]] == r["hh"][0]
        if (only == "gap" and not gap) or (only == "intraday" and gap):
            continue
        e = r["entry_i"]
        paid = r["sold"] + 2 * SLIP
        path = np.array(r["comb"][e + 1:], dtype=float)
        pt = np.array(r["hh"][e + 1:])
        live = (pt <= exit_t) & np.isfinite(path)
        idx = np.flatnonzero(live)
        why, xi = "TIME", (idx[-1] if len(idx) else None)
        cand = []
        if sl is not None:
            h = np.flatnonzero(live & (path <= paid * (1 - sl)))
            if len(h):
                cand.append((h[0], "SL"))
        if tgt is not None:
            h = np.flatnonzero(live & (path >= paid * (1 + tgt)))
            if len(h):
                cand.append((h[0], "TGT"))
        if cand:
            xi, why = min(cand)
        got = (path[xi] if xi is not None else paid) - 2 * SLIP
        pts = got - paid
        ch = charges.order_charges("BUY", paid / 2, LOT) * 2 + charges.order_charges("SELL", got / 2, LOT) * 2
        res.append({"day": r["day"], "pts": pts, "rs": pts * LOT - ch, "why": why, "dte0": r["dte_is0"],
                    "gap_fill": gap, "gap": r["gap"], "entry_time": r["hh"][e]})
    return pd.DataFrame(res)


def run():
    rows = pickle.load(open(os.path.join(OUT, "days.pkl"), "rb"))
    grid = []
    for only in (None, "gap", "intraday"):
        for sl in SLS:
            for ex in EXITS:
                s = B.summary(simulate(rows, ex, sl, None, only))
                if s:
                    grid.append({"only": only, "sl": sl, "tgt": None, "exit": ex, **s})
            for tgt in TGTS[1:]:
                s = B.summary(simulate(rows, "15:25", sl, tgt, only))
                if s:
                    grid.append({"only": only, "sl": sl, "tgt": tgt, "exit": "15:25", **s})
    return rows, grid


def sim_open_trade(rows, delay, exit_t, gmin, at_open=False, slip=SLIP):
    """Gap-open days only (triggered at 09:15, |gap| > gmin): buy at the 09:15
    open print (at_open) or at the 1-min close `delay` minutes after 09:15;
    sell at the last close <= exit_t."""
    res = []
    for r in rows:
        if not r["filled"] or r["entry_i"] != 0 or r["gap"] is None or abs(r["gap"]) <= gmin:
            continue
        c, hh = np.array(r["comb"], dtype=float), r["hh"]
        if delay >= len(c) or not np.isfinite(c[delay]):
            continue
        paid = (r["sold"] if at_open else c[delay]) + 2 * slip
        k = max(i for i, h in enumerate(hh) if h <= exit_t)
        if k <= delay:
            continue
        got = c[k] - 2 * slip
        ch = charges.order_charges("BUY", paid / 2, LOT) * 2 + charges.order_charges("SELL", got / 2, LOT) * 2
        res.append({"day": r["day"], "rs": (got - paid) * LOT - ch, "gap": r["gap"]})
    return pd.DataFrame(res)


def _s(d):
    return {"n": int(len(d)), "win": round(float((d.rs > 0).mean() * 100), 1), "avg": round(float(d.rs.mean())),
            "total": round(float(d.rs.sum())), "a": round(float(d[d.day < "2024"].rs.mean())),
            "b": round(float(d[d.day >= "2024"].rs.mean())), "worst": round(float(d.rs.min()))}


def report(rows, grid):
    base = simulate(rows, "15:25", None, None)
    by_exit = [{"exit": g["exit"], **{k: g[k] for k in ("n", "win", "avg", "total", "a", "b")}}
               for g in grid if g["only"] is None and g["sl"] is None and g["tgt"] is None]
    gapday = simulate(rows, "09:45", 0.2, None, "gap")
    gaps = gapday.assign(g=pd.cut(gapday.gap.abs(), [-1, 50, 100, 150, 1e9], labels=["under 50", "50–100", "100–150", "over 150"]))
    by_gap = [{"g": str(k), **_s(x)} for k, x in gaps.groupby("g", observed=True)]
    decay = []
    for gmin in (100, 150):
        for label, dl, op in (("09:15 open print", 0, True), ("09:15 close", 0, False), ("09:16 close", 1, False),
                              ("09:17 close", 2, False), ("09:20 close", 5, False)):
            for ex in ("09:45", "12:00", "15:25"):
                decay.append({"gap": gmin, "entry": label, "exit": ex, **_s(sim_open_trade(rows, dl, ex, gmin, op))})
    slip = [{"slip": s_, **_s(sim_open_trade(rows, 0, "09:45", 100, True, s_))} for s_ in (0.5, 1.0, 2.0)]
    best = max(grid, key=lambda g: g["avg"])
    data = {"by_exit": by_exit, "by_gap": by_gap, "decay": decay, "slip": slip,
            "all": _s(base.assign(rs=base.rs)), "best": {k: best[k] for k in ("only", "sl", "tgt", "exit", "n", "avg", "total", "a", "b")},
            "n_cfg": len(grid)}
    path = os.path.join(OUT, "buy_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path, data


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Straddle Buy Trigger</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--best:#e9f8f1;--grp:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--best:#16302a;--grp:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
.rules{display:grid;grid-template-columns:120px 1fr;gap:6px 14px;font-size:13.5px}.rules b{color:var(--muted)}
</style></head><body><div class="wrap">
<h1>Buying the straddle at yesterday's closing price</h1>
<p class="note" style="margin-bottom:16px">Same 1,231 days and data as the selling test · 1-min option prices · 0.5 pt slippage per order unless stated · all charges · 1 lot (65).</p>
<div class="card"><h2>Rule</h2><div class="rules">
<b>Strike</b><span>50-pt strike nearest yesterday's NIFTY close (nearest weekly expiry).</span>
<b>Entry</b><span>Buy-stop at yesterday's closing straddle price: bought when the combined price reaches it (a gap open above it buys at the open), 09:15–14:30.</span>
<b>Exit</b><span>Same day, clock time; stop = premium down 20/30/40%; targets +20% to +100% also tried (${N} variants in all).</span>
</div></div>
<div class="card"><h2>Verdict</h2><ul id="verdict"></ul></div>
<div class="card"><h2>The key test — how fast the edge disappears after the open</h2>
<p class="note">Gap-open days only. The profit exists only if you buy at the very first 09:15 print; buying at the 1-min close a few minutes later removes it.</p>
<div class="scroll"><table id="tDecay"></table></div></div>
<div class="card"><h2>Gap-open days by gap size (exit 09:45, SL 20%, bought at the open print)</h2><div class="scroll"><table id="tGap"></table></div></div>
<div class="card"><h2>All triggered days, no stop — by exit time</h2><div class="scroll"><table id="tExit"></table></div></div>
<div class="card"><h2>Slippage sensitivity (gap &gt; 100, open print, exit 09:45)</h2><div class="scroll"><table id="tSlip"></table></div>
<p class="note">Slippage only moves the price by whole points; it cannot model an opening print that simply is not available to fill at.</p></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
document.querySelector(".rules").innerHTML = document.querySelector(".rules").innerHTML.replace("${N}", D.n_cfg);
const dk = (g, e, x) => D.decay.find(r => r.gap === g && r.entry === e && r.exit === x);
const B = D.best, e0 = dk(100, "09:15 open print", "09:45"), e1 = dk(100, "09:16 close", "09:45"), e5 = dk(100, "09:20 close", "09:45");
document.getElementById("verdict").innerHTML = [
  `<b>On all triggered days the buy is roughly break-even and loses when held:</b> ${rs(D.all.avg)} a trade held to 15:25 (${D.all.n} trades). Early exits look slightly positive only because of the gap-open days below.`,
  `<b>The one strong-looking pattern — big gap opens — is not tradeable.</b> On days NIFTY opened more than 100 pts from yesterday's close, buying at the 09:15 open print and selling at 09:45 made ${rs(e0.avg)} a trade (${e0.n} days, both periods positive). But buying at the 09:16 close instead gives ${rs(e1.avg)}, and at 09:20 ${rs(e5.avg)}. The whole edge is the first minute's price discovery: the straddle's opening print lags the gap and catches up within 1–2 minutes.`,
  `<b>You cannot reliably buy at that opening print.</b> The first option trades of a gap day are thin and often off-market; a real order at 09:15 fills at the ask of a wide spread, closer to the 09:15–09:16 close than to the print. Treat the open-print numbers as a data effect, not an edge.`,
  `<b>The best of ${D.n_cfg} variants</b> (${B.only || "all days"}, ${B.sl == null ? "no SL" : "SL " + B.sl * 100 + "%"}, exit ${B.exit}) made ${rs(B.avg)} a trade — built on those same open-print fills, and picking the best of many variants overstates it further.`,
  `<b>Conclusion:</b> neither selling nor buying the straddle on the yesterday's-close trigger has a usable edge. The trigger does identify the moving days (sellers lose there), but the move is priced in within a couple of minutes of the open.`,
].map(x => `<li>${x}</li>`).join("");
const row = r => `<td>${r.n}</td><td>${r.win}</td><td class="${cl(r.avg)}"><b>${rs(r.avg)}</b></td><td class="${cl(r.total)}">${rs(r.total)}</td><td class="${cl(r.a)}">${rs(r.a)}</td><td class="${cl(r.b)}">${rs(r.b)}</td>`;
const hd = `<th>n</th><th>Win %</th><th>₹ / trade</th><th>Total</th><th>₹/trade 2021–23</th><th>₹/trade 2024–26</th>`;
let h = `<tr><th>Bought at</th><th>Exit</th>${hd}</tr>`;
[100, 150].forEach(g => {
  h += `<tr class="grp"><td colspan="8">Gap more than ${g} pts</td></tr>`;
  D.decay.filter(r => r.gap === g).forEach(r => { h += `<tr><td>${r.entry}</td><td>${r.exit}</td>${row(r)}</tr>`; });
});
document.getElementById("tDecay").innerHTML = h;
document.getElementById("tGap").innerHTML = `<tr><th>|Gap| (pts)</th>${hd}</tr>` + D.by_gap.map(r => `<tr><td>${r.g}</td>${row(r)}</tr>`).join("");
document.getElementById("tExit").innerHTML = `<tr><th>Exit</th><th>n</th><th>Win %</th><th>₹ / trade</th><th>Total</th><th>2021–23 total</th><th>2024–26 total</th></tr>` +
  D.by_exit.map(r => `<tr><td>${r.exit}</td><td>${r.n}</td><td>${r.win}</td><td class="${cl(r.avg)}"><b>${rs(r.avg)}</b></td><td class="${cl(r.total)}">${rs(r.total)}</td><td class="${cl(r.a)}">${rs(r.a)}</td><td class="${cl(r.b)}">${rs(r.b)}</td></tr>`).join("");
document.getElementById("tSlip").innerHTML = `<tr><th>Slippage / order</th>${hd}</tr>` + D.slip.map(r => `<tr><td>${r.slip} pt</td>${row(r)}</tr>`).join("");
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    rows, grid = run()
    g = pd.DataFrame(grid)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(g[["only", "sl", "tgt", "exit", "n", "win", "avg", "total", "dd", "a", "b", "sl_rate", "tgt_rate"]]
          .sort_values("avg", ascending=False).head(25).to_string())
    print(g[(g.tgt.isna()) & (g.sl.isna()) & (g.only.isna())][["exit", "n", "win", "avg", "total", "a", "b"]].to_string())
    pickle.dump(grid, open(os.path.join(OUT, "buy_grid.pkl"), "wb"))
    print(report(rows, grid)[0])
